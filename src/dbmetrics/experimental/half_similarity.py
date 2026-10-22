"""Experimental, offline first/second-half visual similarity; no registry writes."""

from bisect import bisect_left
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess


MODEL_ID = "facebook/vjepa2-vitl-fpc64-256"
MODEL_REVISION = "b3c1679b7c34d3255ef3547f27c7b226aefab26f"
PIPELINE_VERSION = "half-similarity-v2"


@dataclass(frozen=True)
class Config:
    frames_per_half: int = 64
    bins: int = 16
    vjepa: bool = False
    device: str = "cpu"
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    timeout_seconds: float = 120
    max_duration_seconds: float = 600

    def __post_init__(self):
        for name, upper in (("frames_per_half", 128), ("bins", 256)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer in [1, {upper}]")
        if self.vjepa and self.frames_per_half != 64:
            raise ValueError("Pinned V-JEPA checkpoint requires 64 frames per half")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda, or mps")
        for value in (self.timeout_seconds, self.max_duration_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("limits must be finite and positive")


def media_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command, timeout):
    result = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{Path(command[0]).name}: {result.stderr.decode(errors='replace')[-1000:]}")
    return result.stdout


def provenance(path, config, backend=None):
    versions = {"python": platform.python_version()}
    for package in ("torch", "torchvision", "transformers", "numpy", "safetensors", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    for name in ("ffmpeg", "ffprobe"):
        try:
            versions[name] = _run([getattr(config, name), "-version"], 10).decode().splitlines()[0]
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            versions[name] = "unavailable"
    return {
        "pipeline_version": PIPELINE_VERSION,
        "media_sha256": media_hash(path),
        "config": asdict(config),
        "runtime": versions,
        "vjepa": backend.provenance if backend is not None else VJEPABackend.provenance,
        "baseline": "rgb-channel-histogram-v1; concatenate channel counts; cosine",
        "preprocessing": "no autorotation; RGB24; bilinear shortest edge 292; center crop 256",
        "pooling": "V-JEPA encoder tokens: arithmetic mean across space and time",
    }


def request_key(evidence):
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sample_halves(timestamps, duration, count):
    if not timestamps or not math.isfinite(duration) or duration <= 0 or type(count) is not int or count < 1:
        raise ValueError("Invalid timeline or sample count")
    if any(not math.isfinite(value) for value in timestamps) or timestamps != sorted(timestamps):
        raise ValueError("Frame timestamps must be finite and ordered")
    halves = []
    for half in range(2):
        start, end = half * duration / 2, (half + 1) * duration / 2
        candidates = [index for index, value in enumerate(timestamps) if start <= value < end]
        if not candidates:
            raise ValueError("A half has no decodable frames")
        times = [timestamps[index] for index in candidates]
        requested = [start + (index + 0.5) * (end - start) / count for index in range(count)]
        selected = []
        for target in requested:
            position = bisect_left(times, target)
            neighbors = {max(0, position - 1), min(len(times) - 1, position)}
            selected.append(candidates[min(neighbors, key=lambda index: (abs(times[index] - target), index))])
        halves.append({"requested_seconds": requested, "frame_indices": selected,
                       "timestamps_seconds": [timestamps[index] for index in selected],
                       "unique_frame_count": len(set(selected))})
    return tuple(halves)


def _selection_expression(indices):
    terms = [f"eq(n\\,{index})" for index in indices]
    if not terms:
        raise ValueError("No frame indices selected")
    while len(terms) > 1:
        terms = [f"({terms[offset]}+{terms[offset + 1]})" if offset + 1 < len(terms)
                 else terms[offset] for offset in range(0, len(terms), 2)]
    return terms[0]


def decode_media(path, config, *, interval=None):
    path = str(Path(path).resolve())
    base = [config.ffprobe, "-v", "error", "-select_streams", "v:0"]
    metadata = json.loads(_run(base + ["-show_entries", "stream=duration,start_time", "-of", "json", path],
                               config.timeout_seconds))
    stream = metadata["streams"][0]
    duration = float(stream["duration"])
    if not math.isfinite(duration) or not 0 < duration <= config.max_duration_seconds:
        raise ValueError("Video duration is missing, invalid, or exceeds configured limit")
    timeline = json.loads(_run(base + ["-show_frames", "-show_entries", "frame=best_effort_timestamp_time",
                                      "-of", "json", path], config.timeout_seconds))
    timestamps = [float(frame["best_effort_timestamp_time"]) for frame in timeline["frames"]]
    origin = float(stream.get("start_time", timestamps[0]))
    relative = [value - origin for value in timestamps]
    start, end = interval if interval is not None else (0.0, duration)
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration):
        raise ValueError("Requested comparison window extends outside the video timeline")
    halves = sample_halves([value - start for value in relative], end - start, config.frames_per_half)
    for half in halves:
        for name in ("requested_seconds", "timestamps_seconds"):
            half[name] = [value + start for value in half[name]]
    indices = sorted(set(index for half in halves for index in half["frame_indices"]))
    expression = _selection_expression(indices)
    filters = f"select={expression},scale=292:292:force_original_aspect_ratio=increase:flags=bilinear,crop=256:256,setsar=1"
    raw = _run([config.ffmpeg, "-v", "error", "-nostdin", "-noautorotate", "-i", path,
                "-map", "0:v:0", "-an", "-sn", "-dn", "-vf", filters, "-fps_mode", "passthrough",
                "-frames:v", str(len(indices)), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
               config.timeout_seconds)
    frame_size = 256 * 256 * 3
    if len(raw) != len(indices) * frame_size:
        raise ValueError("Decoded frame count/size differs from probed timeline")
    frames = {index: raw[offset * frame_size:(offset + 1) * frame_size] for offset, index in enumerate(indices)}
    for half in halves:
        half["frames"] = [frames[index] for index in half["frame_indices"]]
    return {"duration_seconds": duration, "timestamp_origin_seconds": origin,
            "comparison_interval_seconds": [start, end], "split_seconds": (start + end) / 2,
            "timestamp_reference": "seconds relative to video stream start_time; intervals are left-closed, right-open",
            "halves": list(halves)}


def histogram(frames, bins):
    counts = [0] * (3 * bins)
    if not frames:
        raise ValueError("No frames")
    for frame in frames:
        if not frame or len(frame) % 3:
            raise ValueError("Invalid RGB frame")
        for channel in range(3):
            for value in frame[channel::3]:
                counts[channel * bins + value * bins // 256] += 1
    return counts


def cosine(left, right):
    if not left or len(left) != len(right) or not all(math.isfinite(value) for value in (*left, *right)):
        raise ValueError("Invalid embedding shape or nonfinite values")
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    if not left_norm or not right_norm or not math.isfinite(left_norm * right_norm):
        raise ValueError("Zero or invalid embedding norm")
    value = math.fsum((first / left_norm) * (second / right_norm) for first, second in zip(left, right))
    return max(-1.0, min(1.0, value))


class VJEPABackend:
    provenance = {"model_id": MODEL_ID, "revision": MODEL_REVISION,
                  "transformers_version": "4.57.5", "weights": "safetensors",
                  "local_files_only": True, "representation": "get_vision_features; mean token pooling"}

    def __init__(self, device="cpu"):
        self.device = device
        self.model = None

    def embed(self, frames):
        import torch
        from transformers import AutoVideoProcessor, VJEPA2Model

        if importlib.metadata.version("transformers") != "4.57.5":
            raise RuntimeError("V-JEPA adapter requires transformers==4.57.5")
        if len(frames) != 64 or any(len(frame) != 256 * 256 * 3 for frame in frames):
            raise ValueError("V-JEPA requires 64 RGB 256x256 frames")
        if self.model is None:
            options = {"local_files_only": True, "trust_remote_code": False}
            self.processor = AutoVideoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, **options)
            self.model = VJEPA2Model.from_pretrained(MODEL_ID, use_safetensors=True,
                                                    revision=MODEL_REVISION, torch_dtype=torch.float32, **options).to(self.device).eval()
        video = torch.frombuffer(bytearray(b"".join(frames)), dtype=torch.uint8).reshape(64, 256, 256, 3)
        inputs = self.processor(videos=[video.permute(0, 3, 1, 2)], do_sample_frames=False,
                                do_resize=False, do_center_crop=False, return_tensors="pt")
        pixels = inputs["pixel_values_videos"].to(self.device)
        with torch.inference_mode():
            tokens = self.model.get_vision_features(pixels)
        if tokens.ndim != 3 or tokens.shape[0] != 1:
            raise ValueError("Unexpected V-JEPA encoder output shape")
        return tokens.mean(dim=1)[0].float().cpu().tolist()


def _unknown(reason):
    return {"value": None, "status": "unknown", "reason": reason}


def measure(path, config=None, *, backend=None, decoder=None):
    config = config or Config()
    requested = config.vjepa or backend is not None
    result = {"schema_version": 1, "path": str(Path(path).resolve()), "request_key": None,
              "provenance": None, "sampling": None,
              "visual_histogram_cosine": _unknown("not processed"),
              "vjepa_cosine": _unknown("not processed") if requested else
              {"value": None, "status": "not_requested", "reason": "V-JEPA disabled"}}
    try:
        evidence = provenance(path, config, backend)
        result["provenance"] = evidence
        result["request_key"] = request_key(evidence)
        decoded = (decoder or decode_media)(path, config)
        halves = [half["frames"] for half in decoded["halves"]]
        result["sampling"] = {**decoded, "halves": [
            {key: value for key, value in half.items() if key != "frames"} for half in decoded["halves"]]}
        if media_hash(path) != evidence["media_sha256"]:
            raise ValueError("Media changed during decoding")
    except Exception as error:
        result["visual_histogram_cosine"] = _unknown(f"{type(error).__name__}: {error}")
        if requested:
            result["vjepa_cosine"] = _unknown(f"{type(error).__name__}: {error}")
        return result
    for field, embed in (("visual_histogram_cosine", lambda frames: histogram(frames, config.bins)),
                         ("vjepa_cosine", (backend or VJEPABackend(config.device)).embed)):
        if field == "vjepa_cosine" and not requested:
            continue
        try:
            value = cosine(embed(halves[0]), embed(halves[1]))
            result[field] = {"value": value, "status": "ok", "reason": None}
        except Exception as error:
            result[field] = _unknown(f"{type(error).__name__}: {error}")
    return result


def measure_scene_window(path, candidate_seconds, *, window_seconds=1.0, config=None, backend=None):
    """Compare visual context around an externally supplied candidate, not detect cuts."""
    if (isinstance(candidate_seconds, bool) or isinstance(window_seconds, bool) or
            not math.isfinite(candidate_seconds) or not math.isfinite(window_seconds) or
            candidate_seconds < 0 or window_seconds <= 0):
        raise ValueError("Candidate and window must be finite; candidate >= 0 and window > 0")
    comparison = {"kind": "scene_candidate_windows", "candidate_seconds": candidate_seconds,
                  "window_seconds_each_side": window_seconds,
                  "timestamp_reference": "video stream start_time", "boundary_policy": "reject_not_clip"}

    def decode_window(media_path, options):
        return decode_media(media_path, options,
                            interval=(candidate_seconds - window_seconds, candidate_seconds + window_seconds))

    result = measure(path, config, backend=backend, decoder=decode_window)
    result["comparison"] = comparison
    if result["provenance"] is not None:
        result["provenance"]["comparison"] = comparison
        result["request_key"] = request_key(result["provenance"])
    available = result["vjepa_cosine"]["status"] == "ok"
    result["assessment"] = {
        "confirmed_cut": None,
        "uncertainty": "uncalibrated" if available else "embedding_unavailable",
        "reason": ("Visual embedding similarity has no calibrated semantic/cut decision threshold."
                   if available else "No successful requested visual embedding; baseline alone cannot confirm a semantic cut."),
    }
    return result
