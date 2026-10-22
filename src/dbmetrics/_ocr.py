"""Bounded, offline, single-clip text-detection evidence; no selection decisions.

Use run_benchmark for the killable process-group deadline, including native
inference. measure_clip is the lower-level streaming seam for trusted callers.
Only the optional Paddle adapter imports third-party packages.
"""

from collections import deque
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid


SCHEMA = "dbp-study-text-benchmark-v1"
MODEL = "PP-OCRv5_mobile_det"
MODEL_FILES = ("inference.json", "inference.pdiparams", "inference.yml")
MAX_REPORT_BYTES = 32 * 1024 * 1024
MAX_BOXES = 128


@dataclass(frozen=True)
class Limits:
    max_frames: int = 600
    max_seconds: float = 15
    timeout: float = 120
    threads: int = 1
    every_nth: int = 1
    max_pixels: int = 8_847_360
    detector_side: int = 960

    def __post_init__(self):
        for name, maximum in (("max_frames", 1800), ("threads", 4), ("every_nth", 1800),
                              ("max_pixels", 8_847_360), ("detector_side", 1920)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer in 1..{maximum}")
        for name, maximum in (("max_seconds", 60), ("timeout", 600)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
                raise ValueError(f"{name} must be finite and in (0, {maximum}]")


@dataclass(frozen=True)
class Frame:
    index: int
    pts: int
    time_base: str
    timestamp_seconds: float
    relative_seconds: float
    width: int
    height: int
    rgb: bytes


def local_file(path, maximum):
    path = Path(path).expanduser().absolute()
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
        raise ValueError(f"expected nonempty regular file, no symlink, at most {maximum} bytes: {path}")
    return path


def fingerprint(path, maximum):
    path = local_file(path, maximum)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("input changed during hashing")
    return {"path": str(path), "size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
            "sha256": digest.hexdigest()}


def ffmpeg_provenance(binary, deadline):
    process = subprocess.Popen([str(binary), "-version"], stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = bytearray()
    try:
        for name, chunk in pipe_chunks(process, deadline, 16384, 16384):
            if name == "stdout":
                output.extend(chunk)
        if process.returncode != 0 or not output:
            raise ValueError("could not identify FFmpeg version")
    finally:
        stop_process(process)
    return {"ffmpeg_version": output.decode("utf-8", "replace").splitlines()[0],
            "ffmpeg_sha256": fingerprint(Path(binary).resolve(), 512 * 1024 * 1024)["sha256"]}


def checkpoint_provenance(model_dir, source, revision):
    if not source or not revision:
        raise ValueError("explicit checkpoint source and revision are required")
    directory = Path(model_dir).expanduser().absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("model_dir must be an existing local directory, not a symlink")
    files = [fingerprint(directory / name, 128 * 1024 * 1024) for name in MODEL_FILES]
    return {"model_name": MODEL, "model_dir": str(directory), "source": source,
            "revision": revision, "source_revision_operator_supplied": True, "files": files}


def ffmpeg_command(media, limits, ffmpeg):
    return [str(ffmpeg), "-nostdin", "-hide_banner", "-nostats", "-loglevel", "info",
            "-xerror", "-err_detect", "explode", "-threads", str(limits.threads),
            "-filter_threads", str(limits.threads), "-noautorotate", "-copyts",
            "-protocol_whitelist", "file,pipe", "-f", "mov", "-i", str(media),
            "-map", "0:V:0", "-an", "-sn", "-dn", "-vf", "format=rgb24,showinfo",
            "-vsync", "0", "-frames:v", str(limits.max_frames + 1),
            "-c:v", "ppm", "-threads", str(limits.threads), "-f", "image2pipe", "pipe:1"]


def pipe_chunks(process, deadline, stdout_limit, stderr_limit):
    counts = {"stdout": 0, "stderr": 0}
    with selectors.DefaultSelector() as selector:
        for name in counts:
            stream = getattr(process, name)
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("wall-clock deadline")
            for key, mask in selector.select(min(remaining, 0.1)):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                name = key.data
                counts[name] += len(chunk)
                if counts[name] > (stdout_limit if name == "stdout" else stderr_limit):
                    raise ValueError(f"{name} byte budget exceeded")
                yield name, chunk
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("wall-clock deadline")
        process.wait(timeout=remaining)


def stop_process(process, group=False):
    if group:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        process.kill()
    process.wait(timeout=5)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            stream.close()


def decoded_frames(media, limits, ffmpeg, deadline):
    process = subprocess.Popen(ffmpeg_command(media, limits, ffmpeg), stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pixels = bytearray()
    diagnostics = bytearray()
    timestamps = deque()
    time_base = None
    first_time = None
    previous_time = None
    metadata_count = 0
    frame_count = 0
    tail = b""
    try:
        maximum = (limits.max_pixels * 3 + 128) * (limits.max_frames + 1)
        for name, chunk in pipe_chunks(process, deadline, maximum, 2 * 1024 * 1024):
            if name == "stdout":
                pixels.extend(chunk)
            else:
                diagnostics.extend(chunk)
                tail = (tail + chunk)[-2048:]
                while b"\n" in diagnostics:
                    line, _, rest = diagnostics.partition(b"\n")
                    diagnostics = bytearray(rest)
                    if b"showinfo" not in line:
                        continue
                    base = re.search(rb"config in time_base:\s*(\d+/\d+)", line)
                    if base:
                        new_base = base[1].decode("ascii")
                        if Fraction(new_base) <= 0 or (time_base and new_base != time_base):
                            raise ValueError("invalid or changing time base")
                        time_base = new_base
                    match = re.search(rb"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:", line)
                    if match:
                        if time_base is None or int(match[1]) != metadata_count:
                            raise ValueError("missing or nonconsecutive frame metadata")
                        metadata_count += 1
                        if metadata_count > limits.max_frames + 1:
                            raise ValueError("frame metadata budget exceeded")
                        timestamps.append((int(match[2]), time_base))
            while pixels and timestamps:
                header = re.match(rb"P6\s+(\d+)\s+(\d+)\s+255\n", pixels)
                if header is None:
                    if len(pixels) > 128:
                        raise ValueError("invalid PPM frame header")
                    break
                width, height = int(header[1]), int(header[2])
                if min(width, height) < 1 or width * height > limits.max_pixels:
                    raise ValueError("decoded frame pixel budget exceeded")
                end = header.end() + width * height * 3
                if len(pixels) < end:
                    break
                pts, base = timestamps.popleft()
                timestamp = Fraction(pts) * Fraction(base)
                if previous_time is not None and timestamp <= previous_time:
                    raise ValueError("non-increasing presentation timestamps")
                if first_time is None:
                    first_time = timestamp
                rgb = bytes(pixels[header.end():end])
                del pixels[:end]
                yield Frame(frame_count, pts, base, float(timestamp), float(timestamp - first_time),
                            width, height, rgb)
                previous_time = timestamp
                frame_count += 1
            if len(pixels) > limits.max_pixels * 3 + 65536:
                raise ValueError("unpaired frame buffer budget exceeded")
        if process.returncode != 0:
            raise ValueError("FFmpeg decode failed: " + tail.decode("utf-8", "replace"))
        if pixels or timestamps or frame_count == 0:
            raise ValueError("empty, truncated, or unpaired frame stream")
    finally:
        stop_process(process)


def paddle_detections(result, frame):
    if "dt_polys" not in result:
        raise ValueError("detector did not return dt_polys")
    polygons = result["dt_polys"]
    scores = result.get("dt_scores")
    if len(polygons) > MAX_BOXES or (scores is not None and len(scores) != len(polygons)):
        raise ValueError("invalid detection count or score count")
    detections = []
    for index, polygon in enumerate(polygons):
        if len(polygon) != 4 or any(len(point) != 2 for point in polygon):
            raise ValueError("expected quadrilateral text polygon")
        points = [[float(coordinate) for coordinate in point] for point in polygon]
        for horizontal, vertical in points:
            if not (math.isfinite(horizontal) and math.isfinite(vertical)
                    and 0 <= horizontal <= frame.width and 0 <= vertical <= frame.height):
                raise ValueError("nonfinite or out-of-frame polygon")
        area_twice = sum(point[0] * following[1] - following[0] * point[1]
                         for point, following in zip(points, points[1:] + points[:1]))
        if abs(area_twice) == 0:
            raise ValueError("degenerate polygon")
        score = None if scores is None or scores[index] is None else float(scores[index])
        if score is not None and (not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("invalid detector score")
        detections.append({"polygon": points, "score": score})
    return detections


class PaddleDetector:
    def __init__(self, config):
        if not config.get("model_dir"):
            raise ValueError("an explicit local detection checkpoint is required")
        validate_config(config)
        from paddleocr import TextDetection
        import cv2
        import numpy

        cv2.setNumThreads(config["limits"]["threads"])
        self.numpy = numpy
        self.model = TextDetection(
            model_name=MODEL, model_dir=config["model_dir"], device=config["device"],
            enable_hpi=False, use_tensorrt=False, enable_mkldnn=False,
            cpu_threads=config["limits"]["threads"], limit_type="max",
            limit_side_len=config["limits"]["detector_side"],
            thresh=0.3, box_thresh=0.6, unclip_ratio=1.5)

    def detect(self, frame):
        array = self.numpy.frombuffer(frame.rgb, dtype=self.numpy.uint8).reshape(frame.height, frame.width, 3)
        output = iter(self.model.predict(array[:, :, ::-1].copy(), batch_size=1))
        result = next(output)
        if next(output, None) is not None:
            raise ValueError("expected one detection result per frame")
        return paddle_detections(result, frame)


def summarize(frames, stop_reason, every_nth, detector_enabled, elapsed, diagnostic=None):
    inferred = [frame for frame in frames if frame["inference_status"] == "measured"]
    positive = [frame for frame in inferred if frame["detections"]]
    eof = stop_reason == "eof"
    complete = eof and bool(frames)
    full = complete and len(inferred) == len(frames) and every_nth == 1
    coverage = ("full_decoded_stream" if full else "sampled" if every_nth > 1 else "dense_prefix")
    if not inferred:
        coverage = "not_measured"
    evidence = ("detected" if positive else "no_detections" if full else
                "no_detections_in_processed_frames" if inferred else "not_measured")
    times = [frame["relative_seconds"] for frame in inferred]
    gaps = [later - earlier for earlier, later in zip(times, times[1:])]
    status = "complete" if complete else "incomplete" if stop_reason in (
        "frame_budget", "duration_budget", "deadline") else "error"
    return {"type": "summary", "processing_status": status, "stop_reason": stop_reason,
            "coverage": coverage, "decode_eof": eof, "evidence_result": evidence,
            "detector_enabled": detector_enabled, "decoded_frames": len(frames),
            "inferred_frames": len(inferred), "positive_frames": len(positive),
            "inferred_fraction_of_stream": len(inferred) / len(frames) if complete else None,
            "observed_span_seconds": frames[-1]["relative_seconds"] if frames else None,
            "first_inferred_seconds": times[0] if times else None,
            "last_inferred_seconds": times[-1] if times else None,
            "max_inferred_gap_seconds": max(gaps) if gaps else None,
            "elapsed_seconds": elapsed,
            "inference_seconds": sum(frame.get("inference_seconds", 0) for frame in inferred),
            "diagnostic": diagnostic,
            "annotation": {"text_present": None, "logo_present": None,
                           "text_scope": "unadjudicated_all_visible_text_candidates",
                           "evidence_status": "machine_candidates_require_review"}}


def measure_clip(media, limits, detector, emit, ffmpeg="ffmpeg"):
    started = time.monotonic()
    deadline = started + limits.timeout
    frames = []
    reason = "error"
    diagnostic = None
    stream = decoded_frames(media, limits, ffmpeg, deadline)
    try:
        for frame in stream:
            if frame.index >= limits.max_frames:
                reason = "frame_budget"
                break
            if frame.relative_seconds >= limits.max_seconds:
                reason = "duration_budget"
                break
            event = {key: value for key, value in asdict(frame).items() if key != "rgb"}
            event.update(type="frame", inference_status="skipped" if detector else "not_requested",
                         detections=[], inference_seconds=0)
            if detector is not None and frame.index % limits.every_nth == 0:
                inference_started = time.monotonic()
                try:
                    found = detector.detect(frame)
                    event["detections"] = paddle_detections(
                        {"dt_polys": [item["polygon"] for item in found],
                         "dt_scores": [item.get("score") for item in found]}, frame)
                except Exception:
                    event["inference_status"] = "failed"
                    frames.append(event)
                    emit(event)
                    raise
                event["inference_seconds"] = time.monotonic() - inference_started
                event["inference_status"] = "measured"
            frames.append(event)
            emit(event)
            if time.monotonic() >= deadline:
                raise TimeoutError("wall-clock deadline")
        else:
            reason = "eof"
    except (TimeoutError, subprocess.TimeoutExpired) as error:
        reason, diagnostic = "deadline", str(error)[:2048]
    except Exception as error:
        reason, diagnostic = "error", f"{type(error).__name__}: {error}"[:2048]
    finally:
        stream.close()
    return summarize(frames, reason, limits.every_nth, detector is not None,
                     time.monotonic() - started, diagnostic)


def package_versions():
    versions = {}
    for package in ("paddleocr", "paddlex", "paddlepaddle", "paddlepaddle-gpu", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def validate_config(config, dry_run=False):
    if len(json.dumps(config, allow_nan=False).encode()) >= 4096:
        raise ValueError("configuration must fit within 4095 bytes")
    Limits(**config["limits"])
    if not isinstance(config["clip_id"], str) or not 1 <= len(config["clip_id"]) <= 256:
        raise ValueError("clip_id must contain 1..256 characters")
    if config["detector"] not in ("paddle", "decode-only"):
        raise ValueError("unknown detector")
    if not re.fullmatch(r"cpu|gpu:\d+", config["device"]):
        raise ValueError("device must be cpu or one explicit gpu:N")
    local_file(config["media"], 512 * 1024 * 1024)
    if config.get("model_dir"):
        directory = Path(config["model_dir"])
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("model_dir must be a local directory, not a symlink")
        for name in MODEL_FILES:
            local_file(directory / name, 128 * 1024 * 1024)
        if not config.get("model_source") or not config.get("model_revision"):
            raise ValueError("model_source and model_revision must be explicit")
    elif config["detector"] == "paddle" and not dry_run:
        raise ValueError("--model-dir, --model-source and --model-revision required; no downloads")


def private_directory(path):
    path = Path(path).expanduser().absolute()
    if not path.exists():
        path.mkdir(mode=0o700)
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077
            or info.st_uid != os.getuid()):
        raise ValueError("output directory must be owned by this user, private (0700), and not a symlink")
    return path


def offline_audit(event, arguments):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                 "socket.gethostbyaddr", "socket.getnameinfo", "socket.sendto"):
        raise RuntimeError("offline benchmark refuses Python network access")


def worker_main():
    config = json.loads(sys.stdin.readline(65536))
    destination = sys.stdout

    def emit(event):
        destination.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
        destination.flush()

    sys.addaudithook(offline_audit)
    with redirect_stdout(sys.stderr):
        try:
            validate_config(config)
            decoder_provenance = ffmpeg_provenance(config["ffmpeg"], time.monotonic() + config["limits"]["timeout"])
            media = fingerprint(config["media"], 512 * 1024 * 1024)
            checkpoint = (checkpoint_provenance(config["model_dir"], config["model_source"],
                                                config["model_revision"])
                          if config["detector"] == "paddle" else None)
            emit({"type": "provenance", "media": media, "checkpoint": checkpoint, **decoder_provenance,
                  "packages": package_versions(), "python": sys.version,
                  "module_sha256": fingerprint(__file__, 1024 * 1024)["sha256"],
                  "ffmpeg_command": ffmpeg_command(config["media"], Limits(**config["limits"]),
                                                    config["ffmpeg"]),
                  "device_requested": config["device"], "gpu_execution_verified": False,
                  "representation": "full unrotated RGB24 frame; BGR ndarray to detector; no crop",
                  "detector_parameters": {"limit_type": "max", "limit_side_len": config["limits"]["detector_side"],
                                          "thresh": 0.3, "box_thresh": 0.6, "unclip_ratio": 1.5},
                  "score_semantics": "raw detector score, not calibrated clip confidence"})
            detector = PaddleDetector(config) if config["detector"] == "paddle" else None
            result = measure_clip(config["media"], Limits(**config["limits"]), detector, emit, config["ffmpeg"])
            after = Path(config["media"]).stat()
            if after.st_size != media["size_bytes"] or after.st_mtime_ns != media["mtime_ns"]:
                result.update(stop_reason="error", diagnostic="media changed during measurement")
            emit({"type": "done", "stop_reason": result["stop_reason"], "diagnostic": result["diagnostic"]})
        except Exception as error:
            emit({"type": "done", "stop_reason": "error", "diagnostic": f"{type(error).__name__}: {error}"[:2048]})


def resolve_ffmpeg(binary):
    found = shutil.which(os.path.expanduser(os.fspath(binary)))
    return str(Path(found).resolve(strict=True)) if found else None


def run_benchmark(config, output_dir):
    """Create an exclusive private JSONL journal; supervise one clip, never a job pool."""
    config = dict(config)
    config["media"] = os.path.abspath(os.path.expanduser(config["media"]))
    if config.get("model_dir"):
        config["model_dir"] = os.path.abspath(os.path.expanduser(config["model_dir"]))
    config["ffmpeg"] = resolve_ffmpeg(config["ffmpeg"])
    if config["ffmpeg"] is None:
        raise ValueError("FFmpeg unavailable; supply an existing executable")
    validate_config(config)
    directory = private_directory(output_dir)
    run_id = str(uuid.uuid4())
    report_path = directory / f"{run_id}.jsonl"
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND, 0o600)
    frames = []
    reason, diagnostic = "error", "worker did not finish"
    started = time.monotonic()
    limits = Limits(**config["limits"])
    with os.fdopen(descriptor, "w", encoding="utf-8") as journal:
        def append(event):
            journal.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
            journal.flush()

        append({"type": "start", "schema": SCHEMA, "run_id": run_id, "clip_id": config["clip_id"],
                "created_at": datetime.now(timezone.utc).isoformat(), "config": config})
        with tempfile.TemporaryDirectory(prefix=".text-worker-", dir=directory) as scratch:
            environment = dict(os.environ)
            for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                environment[variable] = str(limits.threads)
            for variable in ("HOME", "XDG_CACHE_HOME", "PADDLE_PDX_CACHE_HOME", "PADDLE_HOME",
                             "PADDLE_EXTENSION_DIR", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "HF_HOME",
                             "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_ASSETS_CACHE", "HUGGINGFACE_ASSETS_CACHE",
                             "HF_XET_CACHE", "TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE",
                             "PYTORCH_PRETRAINED_BERT_CACHE", "MODELSCOPE_CACHE", "MPLCONFIGDIR",
                             "CUDA_CACHE_PATH", "TRITON_CACHE_DIR", "NUMBA_CACHE_DIR",
                             "OPENCV_OPENCL_CACHE_DIR", "TMPDIR", "TMP", "TEMP"):
                environment[variable] = scratch
            environment.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                               PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True", PYTHONDONTWRITEBYTECODE="1")
            process = None
            pending = bytearray()
            tail = b""
            done = False
            try:
                process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve())],
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           cwd=scratch, env=environment, start_new_session=True)
                process.stdin.write((json.dumps(config) + "\n").encode())
                process.stdin.close()
                for name, chunk in pipe_chunks(process, started + limits.timeout, MAX_REPORT_BYTES, 256 * 1024):
                    if name == "stderr":
                        tail = (tail + chunk)[-2048:]
                        continue
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line, _, rest = pending.partition(b"\n")
                        pending = bytearray(rest)
                        event = json.loads(line)
                        if done:
                            raise ValueError("worker emitted data after completion")
                        if event["type"] == "frame":
                            if len(frames) >= limits.max_frames or event["index"] != len(frames):
                                raise ValueError("worker frame budget or ordering violation")
                            frames.append(event)
                            append(event)
                        elif event["type"] == "provenance":
                            append(event)
                        elif event["type"] == "done":
                            done = True
                            reason, diagnostic = event["stop_reason"], event["diagnostic"]
                        else:
                            raise ValueError("unexpected worker event")
                    if len(pending) > 128 * 1024:
                        raise ValueError("worker record byte budget exceeded")
                if process.returncode != 0 or pending or not done:
                    raise ValueError("worker exited without a clean completion: " + tail.decode("utf-8", "replace"))
            except (TimeoutError, subprocess.TimeoutExpired) as error:
                reason, diagnostic = "deadline", str(error)[:2048]
            except Exception as error:
                reason, diagnostic = "error", f"{type(error).__name__}: {error}"[:2048]
            finally:
                if process is not None:
                    stop_process(process, group=True)
        summary = summarize(frames, reason, limits.every_nth, config["detector"] == "paddle",
                            time.monotonic() - started, diagnostic)
        summary.update(schema=SCHEMA, run_id=run_id, clip_id=config["clip_id"])
        append(summary)
        os.fsync(journal.fileno())
    return {"report_path": str(report_path), **summary}


if __name__ == "__main__":
    worker_main()
