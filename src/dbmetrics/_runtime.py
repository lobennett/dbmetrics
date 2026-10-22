from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import time

PROBE_FIELDS = ("duration_seconds", "width_px", "height_px", "frame_rate", "audio_track_present")
VISUAL_FIELDS = ("brightness_mean", "contrast_mean", "colorfulness_mean", "temporal_change_mean")
AUDIO_FIELDS = ("audio_rms_amplitude", "audio_silence_fraction")
TECHNICAL_FIELDS = ("rgb_chroma_mean", "near_gray_pixel_fraction", "frame_interval_cv", "optical_flow_mean_px_per_pair")
ALL_FIELDS = PROBE_FIELDS + VISUAL_FIELDS + AUDIO_FIELDS + TECHNICAL_FIELDS + ("hard_cut_count",)
STAGE_FIELDS = {"probe": PROBE_FIELDS, "visual": VISUAL_FIELDS, "audio": AUDIO_FIELDS, "technical": TECHNICAL_FIELDS, "ffmpeg_cuts": ("hard_cut_count",)}
MAX_PROBE_BYTES = 131072
MAX_WORKER_BYTES = 262144
MAX_MEDIA_BYTES = 2 * 1024 ** 3
MAX_CUT_FRAMES = 18000


def strict_json(raw):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant")

    try:
        return json.loads(raw, object_pairs_hook=unique_object, parse_constant=invalid_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("invalid JSON encoding or nesting") from error


def signature(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise ValueError("per-video timeout exceeded")
    return seconds


def hash_fd(descriptor, deadline, max_bytes):
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= max_bytes:
        raise ValueError("input must be a nonempty regular file within the byte limit")
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    total = 0
    while total < before.st_size:
        remaining(deadline)
        chunk = os.read(descriptor, min(1024 * 1024, before.st_size - total))
        if not chunk:
            raise ValueError("input changed during hashing")
        total += len(chunk)
        digest.update(chunk)
    if signature(before) != signature(os.fstat(descriptor)):
        raise ValueError("input changed during hashing")
    remaining(deadline)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest(), signature(before)


@contextmanager
def open_media(root, relative):
    descriptors = []
    try:
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(directory)
        for part in relative.parts[:-1]:
            directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            descriptors.append(directory)
        descriptor = os.open(relative.name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory)
        descriptors.append(descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("media must be a regular file")
        yield descriptor
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def run_bounded(command, deadline, max_output, pass_fds=(), process_group=True, capture_stderr=False):
    remaining(deadline)
    with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, pass_fds=pass_fds,
                          start_new_session=process_group) as process:
        stdout = bytearray()
        stderr = bytearray()
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, stdout)
                selector.register(process.stderr, selectors.EVENT_READ, stderr)
                while selector.get_map():
                    events = selector.select(min(remaining(deadline), 0.1))
                    for key, mask in events:
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        if len(stdout) + len(stderr) + len(chunk) > max_output:
                            raise ValueError("subprocess output limit exceeded")
                        key.data.extend(chunk)
            process.wait(timeout=remaining(deadline))
            if process.returncode != 0 or (stderr.strip() and not capture_stderr):
                raise ValueError("subprocess failed or reported errors: " + stderr[:512].decode("utf-8", "replace"))
            return (bytes(stdout), bytes(stderr)) if capture_stderr else bytes(stdout)
        finally:
            try:
                if process_group:
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.poll() is None:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()


def result(value=None, status="missing", **evidence):
    return {"value": value, "processing_status": status, "evidence": [evidence]}


def parse_probe(raw, audio_only=False):
    if len(raw) > MAX_PROBE_BYTES:
        raise ValueError("ffprobe output limit exceeded")
    payload = strict_json(raw)
    if not isinstance(payload, dict) or "error" in payload:
        raise ValueError("ffprobe reported invalid media")
    streams = payload.get("streams")
    if (not isinstance(streams, list) or not streams or len(streams) > 64
            or any(not isinstance(stream, dict) or stream.get("codec_type") not in
                   {"video", "audio", "subtitle", "data", "attachment"} for stream in streams)):
        raise ValueError("ffprobe stream inventory missing or malformed")
    stream_indices = set()
    for stream in streams:
        index = stream.get("index")
        disposition = stream.get("disposition", {})
        if (type(index) is not int or index < 0 or index in stream_indices
                or not isinstance(disposition, dict)
                or type(disposition.get("attached_pic", 0)) is not int
                or disposition.get("attached_pic", 0) not in (0, 1)):
            raise ValueError("ffprobe stream identity or disposition is malformed")
        stream_indices.add(index)
    audio = result(any(stream["codec_type"] == "audio" for stream in streams), "complete", source="streams.codec_type")
    if audio_only:
        return {"audio_track_present": audio}
    videos = [stream for stream in streams if stream["codec_type"] == "video"
              and stream.get("disposition", {}).get("attached_pic", 0) == 0]
    if len(videos) != 1:
        raise ValueError("expected exactly one non-attached video stream")
    video = videos[0]
    container = payload.get("format", {})
    if not isinstance(container, dict):
        raise ValueError("malformed ffprobe format")
    raw_values = {"duration_seconds": container.get("duration"), "width_px": video.get("width"),
                  "height_px": video.get("height"), "frame_rate": video.get("avg_frame_rate")}
    parsed = {}
    for field, raw_value in raw_values.items():
        source = "format.duration" if field == "duration_seconds" else "streams.video." + {
            "width_px": "width", "height_px": "height", "frame_rate": "avg_frame_rate"}[field]
        if raw_value is None or raw_value in ("", "N/A") or (field == "frame_rate" and raw_value == "0/0"):
            parsed[field] = result(reason="metadata_unavailable", source=source)
            continue
        try:
            if type(raw_value) is bool:
                raise ValueError("boolean is not numeric metadata")
            if field in ("width_px", "height_px"):
                if type(raw_value) is not int:
                    raise ValueError("pixel dimensions must be integers")
                value = raw_value
            elif field == "frame_rate":
                if not isinstance(raw_value, str) or not re.fullmatch(r"[0-9]{1,10}/[0-9]{1,10}", raw_value):
                    raise ValueError("expected average-frame-rate rational")
                value = float(Fraction(raw_value))
            else:
                value = float(raw_value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("metadata must be finite and positive")
            parsed[field] = result(value, "complete", source=source, raw_value=raw_value)
        except (TypeError, ValueError, OverflowError, ZeroDivisionError):
            parsed[field] = result(status="error", reason="invalid_numeric_metadata", source=source)
    parsed["audio_track_present"] = audio
    return parsed


def empty_records(clip_id, media_path, media_kind, status="missing", reason="not_requested", fields=ALL_FIELDS):
    records = []
    for field in fields:
        method = ("ffmpeg_technical" if field in TECHNICAL_FIELDS else "ffprobe" if field in PROBE_FIELDS else "ffmpeg_audio" if field in AUDIO_FIELDS else "ffmpeg_visual" if field in VISUAL_FIELDS
                  else "scene_detector")
        records.append({"clip_id": clip_id, "field_id": field, **result(status=status, reason=reason),
                        "method": method, "review_status": "unvalidated", "media_sha256": None,
                        "media_path": str(media_path), "media_kind": media_kind,
                        "model_id": None, "model_revision": None})
    return records


def extract_clip(clip, media_root, media_kind, ffprobe, deadline, max_media_bytes, stage,
                 filename, ffmpeg="ffmpeg", ffmpeg_cut_threshold=0.3):
    relative = Path(filename)
    path = Path(media_root) / relative
    records = empty_records(clip["clip_id"], path, media_kind,
                            fields=STAGE_FIELDS[stage])
    requested = STAGE_FIELDS[stage]
    measured = {}
    digest = None
    try:
        with open_media(media_root, relative) as descriptor:
            digest, before = hash_fd(descriptor, deadline, max_media_bytes)
            descriptor_path = "/dev/fd/" + str(descriptor)
            if stage == "audio":
                from ._audio import measure_audio

                os.lseek(descriptor, 0, os.SEEK_SET)
                measured.update(measure_audio(descriptor_path, ffmpeg=ffmpeg, deadline=deadline,
                                              timeout=min(300, remaining(deadline)), pass_fds=(descriptor,),
                                              run_process=run_bounded, has_audio=None))
            else:
                command = [str(ffprobe), "-v", "error", "-max_alloc", "67108864",
                           "-protocol_whitelist", "file,pipe", "-f", "mov",
                           "-show_error", "-show_entries",
                           "format=duration:stream=index,codec_type,width,height,avg_frame_rate,nb_frames:stream_disposition=attached_pic",
                           "-of", "json", "-i", descriptor_path]
                raw = run_bounded(command, deadline, MAX_PROBE_BYTES, (descriptor,), process_group=False)
                measured = parse_probe(raw)
            if stage == "visual":
                from ._visual import measure_visual

                dimensions = (measured["width_px"]["value"], measured["height_px"]["value"])
                if any(value is None for value in dimensions) or dimensions[0] * dimensions[1] > 8388608:
                    raise ValueError("visual decode requires known dimensions of at most 8388608 pixels")
                os.lseek(descriptor, 0, os.SEEK_SET)
                measured.update(measure_visual(descriptor_path, pass_fds=(descriptor,), ffmpeg=ffmpeg,
                                               timeout=min(300, remaining(deadline)), run_process=run_bounded))
            if stage == "technical":
                from ._technical import measure_technical

                dimensions = (measured["width_px"]["value"], measured["height_px"]["value"])
                if (any(value is None for value in dimensions) or max(dimensions) > 4096
                        or dimensions[0] * dimensions[1] > 8388608):
                    raise ValueError("technical decode requires known dimensions <=4096 and <=8388608 pixels")
                os.lseek(descriptor, 0, os.SEEK_SET)
                measured.update(measure_technical(descriptor_path, pass_fds=(descriptor,), ffmpeg=ffmpeg,
                    timeout=min(300, remaining(deadline)), deadline=deadline, run_process=run_bounded))
            if stage == "ffmpeg_cuts":
                from ._cuts import detect_ffmpeg_cuts

                os.lseek(descriptor, 0, os.SEEK_SET)
                measured["hard_cut_count"] = detect_ffmpeg_cuts(
                    descriptor_path, raw, ffmpeg=ffmpeg, timeout=min(300, remaining(deadline)),
                    threshold=ffmpeg_cut_threshold, run_process=run_bounded, pass_fds=(descriptor,))
            remaining(deadline)
            with open_media(media_root, relative) as current:
                if signature(os.fstat(descriptor)) != before or signature(os.fstat(current)) != before:
                    raise ValueError("media changed during extraction")
    except Exception as error:
        status = "missing" if isinstance(error, FileNotFoundError) and digest is None else "error"
        for record in records:
            if record["field_id"] in requested:
                record.update(result(status=status, reason=str(error)[:512]))
        return records, True
    failed = False
    for record in records:
        if record["field_id"] not in requested:
            continue
        record["media_sha256"] = digest
        field = record["field_id"]
        if field in measured:
            record.update(measured[field])
            if field in PROBE_FIELDS:
                record["evidence"][0]["ffprobe_binary"] = str(ffprobe)
            failed = failed or record["processing_status"] != "complete"
    return records, failed
