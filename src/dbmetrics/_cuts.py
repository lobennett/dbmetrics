"""Bounded full-clip FFmpeg scene-score candidates, not semantic boundaries.

The caller owns media discovery, the trusted held descriptor and its SHA256,
and must bind probe_raw to that same unchanged file. This module neither opens
nor hashes input itself; input_path and pass_fds pass unchanged to FFmpeg.
No ML or application modules are imported. Probe validation and the default
bounded subprocess runner are imported lazily from pilot_extract so the parent
can later import this backend lazily without a startup circular dependency.

Every decoded frame produces a paired metadata index, PTS and scene score.
Completion requires precisely the indexed ffprobe nb_frames count; unknown
counts, partial decode, malformed scores and excess frames are errors, never
zero cuts. The extra frame budget detects underestimated indexed counts.
At most 2048 candidate indices/PTS are retained; overflow fails explicitly.
Integer PTS are exact but have no inferred time base; pts_time is FFmpeg's
rounded presentation timestamp, not an exact frame-rate-derived time. Optional
interval scoring defaults to timestamps reset to the first video frame, or can
preserve zero-origin container timestamps after caller validation. Both modes
validate the complete indexed decode and retain transitions only when both
adjacent observed frame timestamps are inside the requested half-open interval.

run_process(command, deadline, max_output, pass_fds=(), process_group=False,
capture_stderr=True) must enforce the deadline and combined byte limit, raise
on unsuccessful exit, and return (stdout_bytes, stderr_bytes). FFmpeg remains
inside the caller's killable worker process group. This is not a worker or CLI.
"""

import json
import math
import os
import re
import subprocess
import time


MAX_CUTS = 2048
MAX_RESULT_BYTES = 131072
MAX_PIXELS = 8388608
PROBE_DURATION_TOLERANCE_SECONDS = 0.001
TRANSFORM = ("scale=160:-2:flags=area,format=yuv420p,select='gte(scene,0)',"
             "metadata=mode=print:key=lavfi.scene_score")
INTERVAL_TRANSFORM = "setpts=PTS-STARTPTS," + TRANSFORM
METADATA_LINE = re.compile(r"^(\[Parsed_metadata_[0-9]+ @ [^\]\r\n]+\])\s+\[info\]\s+(.*)$")
FRAME_LINE = re.compile(r"frame:\s*([0-9]{1,8})\s+pts:\s*(-?[0-9]{1,20})\s+pts_time:\s*(\S{1,32})\s*")
SCORE_LINE = re.compile(r"lavfi\.scene_score=(\S{1,32})\s*")


def _probe_bounds(raw):
    from ._runtime import MAX_CUT_FRAMES, parse_probe, strict_json

    if not isinstance(raw, (str, bytes)):
        raise ValueError("invalid probe type")
    parsed = parse_probe(raw)
    payload = strict_json(raw)
    video = next(stream for stream in payload["streams"] if stream["codec_type"] == "video"
                 and stream.get("disposition", {}).get("attached_pic", 0) == 0)
    raw_frames = video.get("nb_frames")
    if type(raw_frames) is int:
        expected = raw_frames
    elif isinstance(raw_frames, str) and re.fullmatch(r"[0-9]{1,8}", raw_frames):
        expected = int(raw_frames)
    else:
        raise ValueError("indexed frame count unavailable")
    if not 1 <= expected <= MAX_CUT_FRAMES:
        raise ValueError("indexed frame count exceeds bounds")
    required = ("duration_seconds", "width_px", "height_px")
    if any(parsed[field]["processing_status"] != "complete" for field in required):
        raise ValueError("incomplete probe metadata")
    duration, width, height = (parsed[field]["value"] for field in required)
    resized_height_bound = max(2, math.ceil(height * 80 / width) * 2)
    if duration > 600 or width * height > MAX_PIXELS or 160 * resized_height_bound > MAX_PIXELS:
        raise ValueError("decode dimensions or duration exceed bounds")
    return expected, duration, width, height


def _parse_scores(stderr, expected, threshold, interval=None):
    text = stderr.decode("utf-8", "strict")
    if re.search(r"\[(?:error|fatal|panic)\]", text, re.IGNORECASE):
        raise ValueError("FFmpeg reported a decode error")
    versions = re.findall(r"^(?:\[info\]\s+)?ffmpeg version (\S{1,128})\s", text, re.MULTILINE)
    if len(versions) != 1:
        raise ValueError("FFmpeg version unavailable")
    pending = None
    previous_pts = None
    previous_time = None
    first_time = None
    metadata_prefix = None
    decoded = 0
    indices = []
    cut_pts = []
    cut_times = []
    interval_frames = 0
    for line in text.splitlines():
        if "Parsed_metadata_" not in line:
            continue
        entry = METADATA_LINE.fullmatch(line)
        if not entry:
            raise ValueError("malformed frame metadata")
        prefix, body = entry.groups()
        if metadata_prefix is None:
            metadata_prefix = prefix
        if prefix != metadata_prefix:
            raise ValueError("multiple metadata streams")
        frame = FRAME_LINE.fullmatch(body)
        score = SCORE_LINE.fullmatch(body)
        if frame:
            if pending is not None or int(frame[1]) != decoded or decoded >= expected:
                raise ValueError("noncontiguous or excess frame metadata")
            pts = int(frame[2])
            seconds = float(frame[3])
            if (not -(2 ** 63) < pts < 2 ** 63 or not math.isfinite(seconds)
                    or (previous_pts is not None and pts <= previous_pts)
                    or (previous_time is not None and seconds <= previous_time)):
                raise ValueError("invalid or nonmonotonic presentation timestamps")
            if first_time is None:
                first_time = seconds
            if seconds - first_time > 600:
                raise ValueError("observed clip exceeds duration bound")
            pending = (pts, seconds)
        elif score and pending is not None:
            value = float(score[1])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid scene score")
            pts, seconds = pending
            if interval is not None and interval[0] <= seconds < interval[1]:
                interval_frames += 1
            if (decoded and value >= threshold
                    and (interval is None or (interval[0] <= previous_time < interval[1]
                                              and interval[0] <= seconds < interval[1]))):
                if len(indices) >= MAX_CUTS:
                    raise ValueError("candidate evidence limit exceeded")
                indices.append(decoded)
                cut_pts.append(pts)
                cut_times.append(seconds)
            previous_pts, previous_time = pts, seconds
            pending = None
            decoded += 1
        else:
            raise ValueError("unpaired or malformed scene metadata")
    if pending is not None or decoded != expected:
        raise ValueError("incomplete indexed frame coverage")
    observations = {"software_version": versions[0], "decoded_frames": decoded,
                    "cut_frame_indices": indices, "cut_pts": cut_pts,
                    "cut_pts_time_seconds": cut_times,
                    "observed_pts_time_span_seconds": previous_time - first_time}
    if interval is not None:
        observations["interval_observed_frames"] = interval_frames
    return observations


def detect_ffmpeg_cuts(input_path, probe_raw, ffmpeg="ffmpeg", timeout=30,
                       threshold=0.3, run_process=None, pass_fds=(), interval=None,
                       timestamp_origin="first_frame"):
    """Return one hard_cut_count record; invalid caller controls raise ValueError."""
    if (type(timeout) not in (int, float) or not math.isfinite(timeout)
            or not 0 < timeout <= 300):
        raise ValueError("timeout must be finite and in (0, 300]")
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 <= threshold <= 1):
        raise ValueError("threshold must be finite and in [0, 1]")
    if (timestamp_origin not in ("first_frame", "container")
            or timestamp_origin == "container" and interval is None):
        raise ValueError("container timestamps require an interval; otherwise use first_frame")
    if interval is not None:
        if (type(interval) is not tuple or len(interval) != 2
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       for value in interval)
                or not 0 <= interval[0] < interval[1] <= 600):
            raise ValueError("interval must be a finite numeric (start, end) with 0 <= start < end <= 600")
    deadline = time.monotonic() + timeout
    evidence = {
        "detector": "FFmpeg.scene_score", "algorithm": "dbp-ffmpeg-cuts-v1",
        "threshold": threshold, "candidate_only": True,
        "warning": "UNVALIDATED threshold-based cut candidates, not semantic event boundaries or scientific ground truth",
        "frame_count_source": "ffprobe.stream.nb_frames",
        "timing_basis": "decoded_frame_indices_not_event_bounds; zero-based index of first frame after candidate transition",
        "timestamp_basis": "observed integer PTS and FFmpeg rounded pts_time seconds; FFmpeg default timestamp origin, no inferred FPS grid",
        "resize": "Unrotated area-resampled width 160, aspect ratio preserved to even height; yuv420p encoded pixels, no HDR tone mapping",
        "transform": TRANSFORM, "stream_selection": "0:V:0 (non-attached video)",
        "timeout_seconds": timeout, "max_retained_cuts": MAX_CUTS,
        "coverage": "not_established",
    }
    if interval is not None:
        interval_evidence = {
            "interval_seconds": {"start": interval[0], "end": interval[1]},
            "interval_boundary_rule": "half-open [start,end); count only when both previous and current observed frame timestamps are inside",
            "probe_duration_rounding_tolerance_seconds": PROBE_DURATION_TOLERANCE_SECONDS,
        }
        if timestamp_origin == "container":
            interval_evidence.update({
                "algorithm": "dbp-ffmpeg-cuts-container-interval-v1",
                "interval_time_basis": "zero-origin container pts_time preserved with -copyts; caller verifies format start_time 0; first video frame is not rebased to zero",
                "timestamp_basis": "observed integer PTS and FFmpeg rounded pts_time seconds preserved with -copyts from a zero-origin container; first video frame may begin after zero, no inferred FPS grid",
                "transform": TRANSFORM,
            })
        else:
            interval_evidence.update({
                "algorithm": "dbp-ffmpeg-cuts-interval-v1",
                "interval_time_basis": "decoded pts_time after setpts=PTS-STARTPTS; first decoded video frame is time zero",
                "timestamp_basis": "observed integer PTS and FFmpeg rounded pts_time seconds after setpts=PTS-STARTPTS; origin is first decoded video frame, no inferred FPS grid",
                "transform": INTERVAL_TRANSFORM,
            })
        evidence.update(interval_evidence)

    def unknown(reason, status="error"):
        return {"value": None, "processing_status": status,
                "evidence": [dict(evidence, reason=reason)]}

    try:
        expected, duration, width, height = _probe_bounds(probe_raw)
    except (ValueError, TypeError, OverflowError):
        return unknown("invalid_or_unbounded_probe")
    if interval is not None and interval[1] > duration + PROBE_DURATION_TOLERANCE_SECONDS:
        raise ValueError("interval end exceeds probed duration")
    max_output = 32768 + (expected + 1) * 512
    evidence.update({"expected_frames": expected, "decode_frame_budget": expected + 1,
                     "source_duration_seconds": duration, "source_width": width,
                     "source_height": height, "max_combined_output_bytes": max_output})
    command = [os.fspath(ffmpeg), "-loglevel", "level+info", "-nostdin", "-nostats", "-xerror",
               "-hwaccel", "none", "-threads", "1", "-noautorotate",
               *(["-copyts"] if timestamp_origin == "container" else []),
               "-protocol_whitelist", "file,pipe", "-f", "mov", "-i", os.fspath(input_path),
               "-map", "0:V:0", "-an", "-sn", "-dn", "-filter_threads", "1",
               "-vf", INTERVAL_TRANSFORM if interval is not None and timestamp_origin == "first_frame" else TRANSFORM,
               "-frames:v", str(expected + 1), "-threads:v", "1",
               "-vsync", "0", "-f", "null", "pipe:1"]
    try:
        if run_process is None:
            from ._runtime import run_bounded

            run_process = run_bounded
        if time.monotonic() >= deadline:
            raise ValueError("deadline exceeded")
        stdout, stderr = run_process(command, deadline, max_output, pass_fds=pass_fds,
                                     process_group=False, capture_stderr=True)
        if (not isinstance(stdout, bytes) or not isinstance(stderr, bytes)
                or stdout or len(stderr) > max_output):
            raise ValueError("invalid or excessive FFmpeg output")
        observations = _parse_scores(stderr, expected, threshold, interval)
        result_evidence = dict(evidence, **observations, coverage="complete_indexed_clip")
        if interval is not None and observations["interval_observed_frames"] == 0:
            completed = {"value": None, "processing_status": "missing",
                         "evidence": [dict(result_evidence,
                                           reason="no_observed_frames_in_interval")]}
        else:
            completed = {"value": len(observations["cut_frame_indices"]),
                         "processing_status": "complete", "evidence": [result_evidence]}
        if len(json.dumps(completed, allow_nan=False).encode("utf-8")) > MAX_RESULT_BYTES:
            return unknown("evidence_limit_exceeded")
        if time.monotonic() >= deadline:
            raise ValueError("deadline exceeded")
        return completed
    except FileNotFoundError:
        return unknown("ffmpeg_unavailable", "missing")
    except (OSError, ValueError, TypeError, OverflowError, subprocess.SubprocessError):
        return unknown("decode_failed")
