"""Opt-in, bounded direct measurements of a consecutive decoded video prefix.

The caller owns the held, hashed media descriptor and source-dimension gate.
Reuse the visual decoder's bounded pipe runner and integer-PTS parser; never
open media here or use OpenCV VideoCapture. FFmpeg is forced to local MOV/MP4,
one CPU thread, no autorotation, area-resized square RGB24, no frame resampling.
Defaults: 120 consecutive frames, 160px square, first 10 presentation seconds,
9,216,000 raw bytes, and 30 wall seconds. Whichever prefix limit comes first
applies; full-clip coverage is never asserted. Source decoding is not an address
space sandbox; extraction's killable process group enforces the total deadline.

RGB channel spread and near-gray pixels describe encoded decoder output, not
color tags, grayscale production history, calibrated gamut or HDR appearance.
PTS interval CV needs at least two intervals and describes only this prefix,
not a VFR class or altered event speed. Flow uses lazy NumPy/OpenCV Farneback,
CPU arrays and one OpenCV thread, averaging magnitudes equally across pixels
then adjacent pairs. Units are resized pixels per pair, not pixels/second or
source-resolution pixels. Cuts are not masked; this is not camera/object
decomposition. Missing/failed flow dependencies invalidate only flow; decode or
total timeout invalidates all four fields. No partial flow scores are emitted.
"""

import math
import os
import subprocess
import time

from ._visual import MAX_RAW_BYTES, MAX_STDERR_BYTES, _remaining, _run_bounded, _source_timestamps


ALGORITHM = "dbp-technical-prefix-rgb-flow-v1"
DEFINITIONS = {
    "rgb_chroma_mean": "Mean pixel RGB channel spread (max(R,G,B)-min(R,G,B))/255 over decoded prefix frames; encoded 0..1, not declared chroma or gamut.",
    "near_gray_pixel_fraction": "Fraction of decoded prefix pixels with max(R,G,B)-min(R,G,B) <= 2 encoded RGB levels; 0..1, not a grayscale class or production history.",
    "frame_interval_cv": "Population standard deviation / mean of adjacent observed presentation-time intervals in the consecutive decoded prefix; needs >=3 frames, not a whole-clip VFR class or event speed.",
    "optical_flow_mean_px_per_pair": "Mean Farneback flow magnitude over resized pixels then adjacent decoded prefix pairs; resized pixels/pair, not camera/object decomposition, source pixels, or pixels/second; cuts unmasked.",
}
FLOW_FIELD = "optical_flow_mean_px_per_pair"
FLOW_PARAMETERS = {"pyr_scale": 0.5, "levels": 3, "winsize": 15, "iterations": 3,
                   "poly_n": 5, "poly_sigma": 1.2, "flags": 0}
SAMPLING_BASIS = {
    "coverage": "bounded_consecutive_prefix_not_full_clip", "source_duration_seconds": None,
    "representation": "FFmpeg rgb24 encoded 8-bit RGB; no linearization or HDR tone mapping",
    "color_basis": "decoder-converted encoded RGB, not verified color tags; absent tags remain unknown",
    "transform": "unrotated area-resampled square; aspect ratio stretched; no padding",
    "stream_selection": "0:V:0 (first non-attached-picture video); audio/subtitles/data discarded",
    "timestamp_basis": "observed source-frame integer PTS * showinfo time base, relative to first decoded frame after setpts=PTS-STARTPTS",
    "cadence_basis": "consecutive source frames in presentation order; no selection, fps filter or duplicates",
    "near_gray_threshold_rgb_levels": 2,
    "flow_units": "resized pixels per adjacent decoded frame pair",
    "flow_grayscale": "OpenCV COLOR_RGB2GRAY on encoded RGB24",
    "flow_threads": 1, "cut_masking": "none; cuts can inflate flow",
}


def _limits(timeout, max_frames, width, window_seconds):
    for name, value, maximum in (("timeout", timeout, 300), ("window_seconds", window_seconds, 60)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(name + " must be finite and within its positive bound")
    if type(max_frames) is not int or not 1 <= max_frames <= 600:
        raise ValueError("max_frames must be an integer in 1..600")
    if type(width) is not int or not 1 <= width <= 320:
        raise ValueError("width must be an integer in 1..320")
    maximum = width * width * 3 * max_frames
    if maximum > MAX_RAW_BYTES:
        raise ValueError("technical raw byte limit exceeded")
    return maximum, maximum + MAX_STDERR_BYTES + max_frames * 512


def _interval_cv(timestamps):
    intervals = [current - previous for previous, current in zip(timestamps, timestamps[1:])]
    if len(intervals) < 2:
        return None
    mean = math.fsum(intervals) / len(intervals)
    return math.sqrt(math.fsum((interval - mean) ** 2 for interval in intervals) / len(intervals)) / mean


def _load_flow_backend():
    import numpy
    import cv2

    return numpy, cv2


def _flow_pair(previous, current, numpy, cv2):
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(previous_gray, current_gray, None, **FLOW_PARAMETERS)
    if flow is None or flow.shape != (*previous_gray.shape, 2):
        raise ValueError("invalid optical flow shape")
    magnitudes = numpy.hypot(flow[:, :, 0].astype(numpy.float64), flow[:, :, 1].astype(numpy.float64))
    if not numpy.isfinite(magnitudes).all() or (magnitudes > math.hypot(*previous_gray.shape)).any():
        raise ValueError("nonfinite or beyond-frame-diagonal optical flow")
    return float(magnitudes.mean())


def measure_technical(input_path, pass_fds=(), ffmpeg="ffmpeg", timeout=30,
                      max_frames=120, width=160, window_seconds=10, run_process=None, deadline=None):
    """Return four value/status/evidence records; invalid controls raise before launch.

    Hard ceilings: 600 frames, 320px square, 64 MiB RGB, 60 presentation seconds,
    300 wall seconds. An optional enclosing deadline can only shorten runtime.
    run_process has pilot_extract.run_bounded's signature and must enforce both
    streams' combined byte limit and deadline. The built-in runner does so.
    """
    maximum, combined = _limits(timeout, max_frames, width, window_seconds)
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("deadline must be finite")
    deadline = min(deadline, time.monotonic() + timeout) if deadline is not None else time.monotonic() + timeout
    binary = os.fspath(ffmpeg)
    sampling = dict(SAMPLING_BASIS, max_frames=max_frames, width=width, height=width,
                    requested_window_seconds=window_seconds, timeout_seconds=timeout,
                    max_raw_bytes=maximum, max_combined_output_bytes=combined,
                    frame_count=None, frame_limit_reached=None, source_timestamps_seconds=[],
                    observed_sample_span_seconds=None, interval_min_seconds=None, interval_max_seconds=None)
    aggregates = {}

    def records(values=None, status="error", reason=None, diagnostic=None):
        output = {}
        for field, definition in DEFINITIONS.items():
            evidence = {"algorithm": ALGORITHM, "definition": definition, "ffmpeg_binary": binary,
                        "sampling": dict(sampling), "aggregates": dict(aggregates)}
            if field != "frame_interval_cv":
                evidence["sampling"].pop("source_timestamps_seconds")
            if reason:
                evidence["reason"] = reason
            if diagnostic:
                evidence["diagnostic"] = diagnostic[:1024]
            output[field] = {"value": values[field] if values else None,
                             "processing_status": status, "evidence": [evidence]}
        return output

    transform = (f"setpts=PTS-STARTPTS,trim=duration={window_seconds},trim=end_frame={max_frames},"
                 f"scale={width}:{width}:flags=area,setsar=1,format=rgb24,showinfo=checksum=0")
    command = [binary, "-hide_banner", "-loglevel", "level+info", "-nostdin", "-nostats", "-xerror",
               "-max_alloc", "67108864", "-hwaccel", "none", "-threads", "1", "-noautorotate",
               "-protocol_whitelist", "file,pipe", "-f", "mov", "-i", os.fspath(input_path),
               "-map", "0:V:0", "-an", "-sn", "-dn", "-filter_threads", "1", "-vf", transform,
               "-t", str(window_seconds), "-frames:v", str(max_frames), "-threads:v", "1",
               "-vsync", "0", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    try:
        _remaining(deadline)
        raw, stderr = (run_process or _run_bounded)(command, deadline, combined, pass_fds=pass_fds,
                                                  process_group=False, capture_stderr=True)
        _remaining(deadline)
        frame_bytes = width * width * 3
        if len(raw) > maximum or len(raw) % frame_bytes or len(stderr) > combined - maximum:
            raise ValueError("technical raw or diagnostic output limit/alignment exceeded")
        count = len(raw) // frame_bytes
        timestamps = _source_timestamps(stderr, count, width, 0, window_seconds)
        sampling.update(frame_count=count, frame_limit_reached=count == max_frames,
                        source_timestamps_seconds=timestamps,
                        observed_sample_span_seconds=timestamps[-1] if count else None)
        intervals = [current - previous for previous, current in zip(timestamps, timestamps[1:])]
        if intervals:
            sampling.update(interval_min_seconds=min(intervals), interval_max_seconds=max(intervals))
        if not count:
            return records(status="missing", reason="no_decoded_prefix_frames")
        chroma_sum = near_gray_count = 0
        for frame_offset in range(0, len(raw), frame_bytes):
            _remaining(deadline)
            frame = memoryview(raw)[frame_offset:frame_offset + frame_bytes]
            for offset in range(0, frame_bytes, 3):
                pixel = frame[offset:offset + 3]
                spread = max(pixel) - min(pixel)
                chroma_sum += spread
                near_gray_count += spread <= 2
        pixels = count * width * width
        aggregates.update(pixel_count=pixels, chroma_sum_levels=chroma_sum, near_gray_pixel_count=near_gray_count)
        values = {"rgb_chroma_mean": chroma_sum / (pixels * 255),
                  "near_gray_pixel_fraction": near_gray_count / pixels,
                  "frame_interval_cv": _interval_cv(timestamps), FLOW_FIELD: None}
        output = records(values, "complete")
        if count < 3:
            output["frame_interval_cv"]["processing_status"] = "missing"
            output["frame_interval_cv"]["evidence"][0]["reason"] = "needs_two_presentation_intervals"
        if count < 2:
            output[FLOW_FIELD]["processing_status"] = "missing"
            output[FLOW_FIELD]["evidence"][0]["reason"] = "needs_two_decoded_frames"
        else:
            try:
                numpy, cv2 = _load_flow_backend()
                _remaining(deadline)
                previous_threads = cv2.getNumThreads()
                try:
                    cv2.setNumThreads(1)
                    frames = numpy.frombuffer(raw, dtype=numpy.uint8).reshape(count, width, width, 3)
                    pair_means = []
                    for index in range(1, count):
                        _remaining(deadline)
                        pair_means.append(_flow_pair(frames[index - 1], frames[index], numpy, cv2))
                finally:
                    cv2.setNumThreads(previous_threads)
                _remaining(deadline)
                output[FLOW_FIELD]["value"] = math.fsum(pair_means) / len(pair_means)
                flow_evidence = output[FLOW_FIELD]["evidence"][0]
                flow_evidence.update(opencv_version=cv2.__version__, flow_parameters=dict(FLOW_PARAMETERS))
                flow_evidence["aggregates"].update(flow_pair_count=len(pair_means),
                    flow_pair_mean_sum=math.fsum(pair_means),
                    flow_pair_mean_min=min(pair_means), flow_pair_mean_max=max(pair_means))
            except Exception as error:
                output[FLOW_FIELD]["processing_status"] = "error"
                output[FLOW_FIELD]["value"] = None
                output[FLOW_FIELD]["evidence"][0].update(reason="flow_backend_failed", diagnostic=str(error)[:1024])
        _remaining(deadline)
        return output
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return records(reason="decode_or_budget_failed", diagnostic=str(error))


def validate_evidence(field, annotation):
    """Validate retained definitions, limits and numeric aggregates without ML imports."""
    if len(annotation["evidence"]) != 1:
        raise ValueError("Technical measurements need one evidence record")
    evidence = annotation["evidence"][0]
    sampling, aggregates = evidence.get("sampling"), evidence.get("aggregates")
    binary = evidence.get("ffmpeg_binary")
    if (evidence.get("algorithm") != ALGORITHM or evidence.get("definition") != DEFINITIONS[field]
            or not isinstance(binary, str) or not 0 < len(binary) <= 4096
            or any(ord(character) < 32 for character in binary)
            or not isinstance(sampling, dict) or not isinstance(aggregates, dict)
            or annotation.get("model_id") is not None or annotation.get("model_revision") is not None):
        raise ValueError("Invalid technical definition or provenance")
    for key, expected in SAMPLING_BASIS.items():
        if type(sampling.get(key)) is not type(expected) or sampling.get(key) != expected:
            raise ValueError("Technical sampling differs from registered definition")
    width, count, maximum = sampling.get("width"), sampling.get("frame_count"), sampling.get("max_frames")
    window = sampling.get("requested_window_seconds")
    raw_bytes, combined = _limits(sampling.get("timeout_seconds"), maximum, width, window)
    if type(count) is not int or not 1 <= count <= maximum:
        raise ValueError("Invalid technical frame count")
    for key, expected in (("height", width), ("max_raw_bytes", raw_bytes),
                          ("max_combined_output_bytes", combined), ("frame_limit_reached", count == maximum)):
        if type(sampling.get(key)) is not type(expected) or sampling[key] != expected:
            raise ValueError("Invalid technical sample arithmetic")
    span = sampling.get("observed_sample_span_seconds")
    minimum_interval, maximum_interval = sampling.get("interval_min_seconds"), sampling.get("interval_max_seconds")
    if type(span) not in (int, float) or not math.isfinite(span) or not 0 <= span < window:
        raise ValueError("Invalid prefix presentation span")
    if count == 1:
        if span != 0 or minimum_interval is not None or maximum_interval is not None:
            raise ValueError("One frame has no presentation intervals")
    elif (any(type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < window
              for value in (minimum_interval, maximum_interval))
          or minimum_interval > maximum_interval
          or not minimum_interval - 1e-12 <= span / (count - 1) <= maximum_interval + 1e-12):
        raise ValueError("Invalid prefix interval summary")
    interval_cv = None
    if field == "frame_interval_cv":
        timestamps = sampling.get("source_timestamps_seconds")
        if (not isinstance(timestamps, list) or len(timestamps) != count or count < 3
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < window for value in timestamps)
                or timestamps[0] != 0 or any(current <= previous for previous, current in zip(timestamps, timestamps[1:]))
                or span != timestamps[-1]):
            raise ValueError("Invalid consecutive-prefix PTS")
        intervals = [current - previous for previous, current in zip(timestamps, timestamps[1:])]
        if minimum_interval != min(intervals) or maximum_interval != max(intervals):
            raise ValueError("PTS do not match interval summary")
        interval_cv = _interval_cv(timestamps)
    pixels = count * width * width
    chroma, gray = aggregates.get("chroma_sum_levels"), aggregates.get("near_gray_pixel_count")
    if (type(aggregates.get("pixel_count")) is not int or aggregates["pixel_count"] != pixels
            or type(chroma) is not int or not 0 <= chroma <= pixels * 255
            or type(gray) is not int or not 0 <= gray <= pixels
            or not (pixels - gray) * 3 <= chroma <= gray * 2 + (pixels - gray) * 255):
        raise ValueError("Invalid technical pixel aggregates")
    expected = {"rgb_chroma_mean": chroma / (pixels * 255), "near_gray_pixel_fraction": gray / pixels,
                "frame_interval_cv": interval_cv}.get(field)
    if field == FLOW_FIELD:
        version, parameters = evidence.get("opencv_version"), evidence.get("flow_parameters")
        pairs = aggregates.get("flow_pair_count")
        total = aggregates.get("flow_pair_mean_sum")
        minimum, maximum = aggregates.get("flow_pair_mean_min"), aggregates.get("flow_pair_mean_max")
        if (not isinstance(version, str) or not 0 < len(version) <= 128 or any(ord(character) < 32 for character in version)
                or not isinstance(parameters, dict) or set(parameters) != set(FLOW_PARAMETERS)
                or any(type(parameters[key]) is not type(value) or parameters[key] != value for key, value in FLOW_PARAMETERS.items())
                or count < 2 or type(pairs) is not int or pairs != count - 1
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or not 0 <= value <= math.hypot(width, width) for value in (minimum, maximum))
                or minimum > maximum or type(total) not in (int, float) or not math.isfinite(total)
                or not (pairs - 1) * minimum + maximum - 1e-12 <= total <= (pairs - 1) * maximum + minimum + 1e-12):
            raise ValueError("Invalid fixed optical-flow method or pair aggregates")
        expected = total / pairs
    value = annotation.get("value")
    if (expected is None or type(value) not in (int, float) or not math.isfinite(value)
            or not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("Technical value does not match retained aggregates")
