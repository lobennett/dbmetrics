"""Read verified OCR evidence for one local video."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from datetime import datetime, timezone

FIELDS = ("added_text", "scene_text", "added_logo", "scene_logo")

class ConflictError(ValueError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def finite_number(value, minimum, maximum):
    if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError("Invalid numeric value")
    return value


def fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def read_file(path, root, maximum):
    path = Path(path)
    if not path.is_relative_to(root):
        raise ValueError("File is outside the input directory")
    for component in (path, *path.parents):
        if component == root:
            break
        if component.is_symlink():
            raise ValueError("Symbolic links are not supported")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise ValueError("Invalid input file size or type")
        content = source.read(maximum + 1)
        if len(content) > maximum or fingerprint(before) != fingerprint(os.fstat(source.fileno())):
            raise ValueError("Input file changed during verification")
        return content, fingerprint(before)


class EvidenceVideo:
    def __init__(self, media_path, report_path, *, duration_seconds=None):
        media = Path(media_path).absolute()
        report = Path(report_path).absolute()
        self.batch = report.parent
        report_bytes, _ = read_file(report, self.batch, 16 * 1024 * 1024)
        start = json.loads(report_bytes.splitlines()[0])
        self.clip_id = start["clip_id"]
        media_bytes, identity = read_file(media, media.parent, 512 * 1024 * 1024)
        self.clips = {self.clip_id: {
            "clip_id": self.clip_id, "media": media, "identity": identity,
            "media_sha256": hashlib.sha256(media_bytes).hexdigest(),
            "duration_seconds": duration_seconds,
        }}
        self._load_report(report)

    def _load_report(self, path):
        content, unused_identity = read_file(path, self.batch, 16 * 1024 * 1024)
        events = [json.loads(line) for line in content.splitlines()]
        if len(events) < 4 or events[0].get("type") != "start" or events[-1].get("type") != "summary":
            raise ValueError("Incomplete evidence journal")
        clip = self.clips.get(events[0].get("clip_id"))
        if clip is None or "evidence" in clip:
            raise ValueError("Unknown or duplicate evidence clip")
        provenance = [event for event in events if event.get("type") == "provenance"]
        if len(provenance) != 1 or provenance[0]["media"]["sha256"] != clip["media_sha256"]:
            raise ValueError("Media SHA-256 does not match the OCR evidence")
        summary = events[-1]
        raw_frames = [event for event in events if event.get("type") == "frame"]
        if not 1 <= len(raw_frames) <= 1800:
            raise ValueError("Invalid frame count")
        if (summary.get("clip_id") != clip["clip_id"] or summary.get("processing_status") != "complete"
                or summary.get("coverage") != "full_decoded_stream" or summary.get("decode_eof") is not True
                or summary.get("decoded_frames") != len(raw_frames) or summary.get("inferred_frames") != len(raw_frames)):
            raise ValueError("Only complete, verified pilot evidence is supported")
        frames = []
        width, height = raw_frames[0]["width"], raw_frames[0]["height"]
        finite_number(width, 1, 10000)
        finite_number(height, 1, 10000)
        previous = 0
        for index, frame in enumerate(raw_frames):
            timestamp = finite_number(frame["relative_seconds"], previous, 60)
            if frame.get("index") != index or frame.get("inference_status") != "measured" or (frame["width"], frame["height"]) != (width, height):
                raise ValueError("Inconsistent measured frames")
            detections = frame["detections"]
            if not isinstance(detections, list) or len(detections) > 500:
                raise ValueError("Invalid detection list")
            for detection in detections:
                polygon = detection.get("polygon")
                if not isinstance(polygon, list) or len(polygon) != 4:
                    raise ValueError("Invalid text polygon")
                for point in polygon:
                    if len(point) != 2:
                        raise ValueError("Invalid polygon point")
                    finite_number(point[0], 0, width)
                    finite_number(point[1], 0, height)
                if detection.get("score") is not None:
                    finite_number(detection["score"], 0, 1)
            frames.append({"index": index, "time_seconds": timestamp, "detections": detections})
            previous = timestamp
        positive = sum(bool(frame["detections"]) for frame in frames)
        if summary.get("positive_frames") != positive:
            raise ValueError("Detection count does not match frame evidence")
        clip["report_sha256"] = hashlib.sha256(content).hexdigest()
        clip["evidence"] = {"clip_id": clip["clip_id"], "width": width, "height": height,
                            "frames": frames, "coverage": summary["coverage"],
                            "inferred_frames": len(frames), "positive_frames": positive,
                            "duration_seconds": clip["duration_seconds"],
                            "model_name": provenance[0]["checkpoint"]["model_name"],
                            "score_semantics": "Raw detector score, not calibrated clip confidence."}

    def get(self, clip_id):
        if clip_id not in self.clips:
            raise KeyError("Unknown clip")
        return self.clips[clip_id]

    def open_media(self, clip_id):
        clip = self.get(clip_id)
        source = open(os.open(clip["media"], os.O_RDONLY | os.O_NOFOLLOW), "rb", closefd=True)
        if fingerprint(os.fstat(source.fileno())) != clip["identity"]:
            source.close()
            raise ConflictError("Media changed; restart after verifying the source")
        return source
