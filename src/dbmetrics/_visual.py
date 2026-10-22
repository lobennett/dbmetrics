"""Bounded CPU-only visual proxies on FFmpeg's sampled, encoded RGB bytes.

No application imports, media discovery, hashing, or file ownership: the caller
holds and validates the input descriptor throughout this call. Its path (for
example /dev/fd/17) is passed unchanged to FFmpeg with pass_fds=(17,).
Input is forced through the MOV/MP4 demuxer with only file and pipe protocols.

Sampling resets the first video PTS to zero, then selects distinct source frames
at least max(0.5, 1/sample_fps) seconds apart, without duplicating frames. Bounded
showinfo supplies observed integer PTS and time base, paired with RGB output.
Reported source timestamps are relative to the first decoded frame, not the
container's absolute origin. Cadence is observed and variable, not necessarily
the requested sample_fps. Source duration and full-clip coverage are not
established; no ffprobe is run.
Frames are area-resampled to width-by-width squares without padding, stretching
aspect ratio. Autorotation is disabled. Measurements describe this transformed
rgb24 representation, not linear-light luminance, HDR, or calibrated color.

Brightness is mean encoded Rec.709-weighted RGB luma / 255. Contrast is the
mean within-frame population luma standard deviation / 255 (actual maximum
0.5, accepted field range 0..1). Colorfulness is the Hasler-Suesstrunk RGB
opponent proxy: (sqrt(var(R-G)+var((R+G)/2-B)) +
0.3*sqrt(mean(R-G)**2+mean((R+G)/2-B)**2)) / 255, averaged per frame.
It is nonnegative and NOT clipped to one. Temporal change is mean absolute
encoded-luma difference between consecutive sampled frames / 255, averaged
over pairs; it is neither optical flow, camera/object motion, nor cut count.

All four records have value, processing_status, and evidence. Empty output is
missing; decode/timeout failures are error with null values, never partial
measurements. Invalid resource settings raise ValueError before launching.
The optional run_process callback has pilot_extract.run_bounded's signature
and returns (stdout_bytes, stderr_bytes including showinfo); process_group=False keeps FFmpeg in
the extractor worker's killable process group. It must enforce the supplied
deadline and combined output limit. The built-in runner does so independently.
"""

from fractions import Fraction
import math
import os
import re
import selectors
import signal
import subprocess
import time


DEFINITIONS = {
    "brightness_mean": "Mean per-frame encoded luma: (0.2126*R + 0.7152*G + 0.0722*B)/255; range 0..1.",
    "contrast_mean": "Mean per-frame population standard deviation of encoded luma/255; range 0..0.5 within field 0..1.",
    "colorfulness_mean": "Mean Hasler-Suesstrunk RGB opponent proxy/255: sqrt(var(R-G)+var((R+G)/2-B)) + 0.3*sqrt(mean(R-G)**2+mean((R+G)/2-B)**2), then divide by 255; nonnegative, not clipped to one.",
    "temporal_change_mean": "Mean per-pair absolute encoded luma/255 change at corresponding sampled pixels; range 0..1; not optical flow, camera/object motion, or hard-cut count.",
}
MAX_STDERR_BYTES = 32768
MAX_RAW_BYTES = 64 * 1024 * 1024
LUMA_SCALE = 2550000


def _remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise ValueError("visual measurement timeout exceeded")
    return seconds


def _run_bounded(command, deadline, max_output, pass_fds=(),
                 process_group=False, capture_stderr=True):
    _remaining(deadline)
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
                    for key, mask in selector.select(min(_remaining(deadline), 0.1)):
                        chunk = os.read(key.fileobj.fileno(), 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        if len(stdout) + len(stderr) + len(chunk) > max_output:
                            raise ValueError("subprocess output limit exceeded")
                        key.data.extend(chunk)
            process.wait(timeout=_remaining(deadline))
            if process.returncode != 0:
                raise ValueError("FFmpeg exited unsuccessfully: " +
                                 stderr[:1024].decode("utf-8", "replace"))
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


def _frame_measurements(frame):
    lumas = []
    red_green_sum = 0
    red_green_squares = 0
    yellow_blue_sum = 0
    yellow_blue_squares = 0
    for offset in range(0, len(frame), 3):
        red, green, blue = frame[offset:offset + 3]
        lumas.append(2126 * red + 7152 * green + 722 * blue)
        red_green = red - green
        yellow_blue = red + green - 2 * blue
        red_green_sum += red_green
        red_green_squares += red_green * red_green
        yellow_blue_sum += yellow_blue
        yellow_blue_squares += yellow_blue * yellow_blue
    pixels = len(lumas)
    mean_luma = sum(lumas) / pixels
    contrast = math.sqrt(sum((value - mean_luma) ** 2 for value in lumas) / pixels)
    mean_red_green = red_green_sum / pixels
    mean_yellow_blue = yellow_blue_sum / pixels / 2
    variance_red_green = max(0.0, red_green_squares / pixels - mean_red_green ** 2)
    variance_yellow_blue = max(0.0, yellow_blue_squares / pixels / 4 - mean_yellow_blue ** 2)
    colorfulness = (math.sqrt(variance_red_green + variance_yellow_blue) +
                   0.3 * math.hypot(mean_red_green, mean_yellow_blue)) / 255
    return lumas, mean_luma / LUMA_SCALE, contrast / LUMA_SCALE, colorfulness


def _source_timestamps(diagnostics, frame_count, width, spacing, window, interval=None):
    text = diagnostics.decode("utf-8", "strict")
    if re.search(r"\[(?:error|fatal|panic)\]", text, re.IGNORECASE):
        raise ValueError("FFmpeg reported a decode error")
    lines = [line for line in text.splitlines()
             if "showinfo" in line and re.search(r"\bn:\s*", line)]
    if len(lines) != frame_count:
        raise ValueError("decoded frames cannot be paired with observed PTS")
    if not frame_count:
        return []
    time_bases = re.findall(r"showinfo[^\n]*config in time_base:\s*([0-9]+/[0-9]+)", text)
    if len(time_bases) != 1:
        raise ValueError("missing or ambiguous showinfo time base")
    try:
        time_base = Fraction(time_bases[0])
    except ZeroDivisionError as error:
        raise ValueError("invalid showinfo time base") from error
    if time_base <= 0:
        raise ValueError("invalid showinfo time base")
    timestamps = []
    for index, line in enumerate(lines):
        match = re.search(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:(\S+).*?"
                          r"fmt:(\S+).*?\bs:(\d+)x(\d+)", line)
        if (not match or int(match.group(1)) != index
                or match.group(4, 5, 6) != ("rgb24", str(width), str(width))):
            raise ValueError("malformed or reordered showinfo frame")
        actual = float(int(match.group(2)) * time_base)
        reported = float(match.group(3))
        start, end = interval if interval is not None else (0, window)
        if (not math.isfinite(reported) or not start <= actual < end
                or not math.isclose(actual, reported, rel_tol=1e-5, abs_tol=1e-6)
                or (not timestamps and interval is None and actual != 0)
                or (timestamps and (actual <= timestamps[-1]
                    or actual - timestamps[-1] + 1e-9 < spacing))):
            raise ValueError("missing, duplicate, too-close, or out-of-window observed PTS")
        timestamps.append(actual)
    return timestamps


def measure_visual(input_path, pass_fds=(), ffmpeg="/usr/bin/ffmpeg", timeout=30,
                   max_frames=120, sample_fps=2, width=160, run_process=None, interval=None,
                   timestamp_origin="first_frame"):
    """Return four metric records, with shared sampling details in each evidence.

    Defaults cap raw output at 9,216,000 bytes plus 94,208 diagnostic bytes,
    120 square 160px frames, a requested 60-second prefix, and 30 seconds
    wall time including metric computation. Hard configurable ceilings: 600
    frames, 320px square, 64 MiB raw RGB, 600 prefix seconds, 300 wall seconds,
    and 0.1..30 requested samples/second, clamped to at most two observations/sec.
    Minimum spacing is max(0.5, 1/sample_fps); the prefix is max_frames * spacing.
    Diagnostic allowance is 32 KiB plus 512 bytes per requested frame.
    These are output/work limits, not an
    address-space cap on FFmpeg's source-resolution decoder. The outer worker
    remains responsible for media validation and OS-level resource isolation.
    Optional interval=(start, end) selects a half-open source-frame interval after
    resetting the first decoded video PTS to zero, before sampling. Observed PTS
    remain relative to that original frame, not the first retained segment frame.
    The complete requested interval must fit the sampling window; no silent prefix
    truncation is allowed. The caller validates media origin, duration and identity.
    timestamp_origin="container" instead preserves input PTS with copyts; the
    caller must verify a zero-origin container. Video-stream delay is retained.
    """
    for name, value, lower, upper in (("timeout", timeout, 0, 300),
                                      ("sample_fps", sample_fps, 0, 30)):
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not lower < value <= upper):
            raise ValueError(name + " must be finite and within its positive bound")
    if type(max_frames) is not int or not 1 <= max_frames <= 600:
        raise ValueError("max_frames must be an integer in 1..600")
    if type(width) is not int or not 1 <= width <= 320:
        raise ValueError("width must be an integer in 1..320")
    frame_bytes = width * width * 3
    max_raw_bytes = frame_bytes * max_frames
    spacing = max(0.5, 1 / sample_fps)
    window = max_frames * spacing
    max_diagnostics = MAX_STDERR_BYTES + max_frames * 512
    if sample_fps < 0.1 or window > 600 or max_raw_bytes > MAX_RAW_BYTES:
        raise ValueError("sampling exceeds the cadence, duration, or raw byte limit")
    if timestamp_origin not in ("first_frame", "container") or timestamp_origin == "container" and interval is None:
        raise ValueError("container timestamps require an interval; otherwise use first_frame")
    if interval is not None:
        if (type(interval) not in (tuple, list) or len(interval) != 2
                or any(type(value) not in (int, float) for value in interval)
                or not 0 <= interval[0] < interval[1] <= 600
                or interval[1] - interval[0] > window):
            raise ValueError("interval must fit in 0..600 seconds and the full sampling budget")
        interval = tuple(float(value) for value in interval)
    deadline = time.monotonic() + timeout
    binary = os.fspath(ffmpeg)
    cadence = f"isnan(prev_selected_t)+gte(t-prev_selected_t,{spacing})"
    selection = (f"select='gte(t,{interval[0]})*lt(t,{interval[1]})*({cadence})'"
                 if interval is not None else f"trim=duration={window},select='{cadence}'")
    origin = "setpts=PTS" if timestamp_origin == "container" else "setpts=PTS-STARTPTS"
    transform = (f"{origin},{selection},"
                 f"trim=end_frame={max_frames},scale={width}:{width}:flags=area,"
                 "setsar=1,format=rgb24,showinfo=checksum=0")
    command = [binary, "-hide_banner", "-loglevel", "level+info", "-nostdin", "-nostats", "-xerror",
               "-hwaccel", "none", "-threads", "1", "-noautorotate",
               *(["-copyts"] if timestamp_origin == "container" else []),
               "-protocol_whitelist", "file,pipe", "-f", "mov",
               "-i", os.fspath(input_path), "-map", "0:V:0", "-an", "-sn", "-dn",
               "-filter_threads", "1", "-vf", transform,
               *(["-t", str(window)] if interval is None else []),
               "-frames:v", str(max_frames), "-threads:v", "1", "-vsync", "0",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    sampling = {
        "sample_fps": sample_fps, "max_frames": max_frames, "width": width,
        "height": width, "frame_count": None, "frame_limit_reached": None,
        "timeout_seconds": timeout, "requested_window_seconds": window,
        "max_raw_bytes": max_raw_bytes, "max_combined_output_bytes": max_raw_bytes + max_diagnostics,
        "representation": "FFmpeg rgb24, encoded 8-bit RGB; no linearization or HDR tone mapping",
        "transform": "Unrotated area-resampled square, aspect ratio stretched; no padding; " + transform,
        "stream_selection": "0:V:0 (first non-attached-picture video); audio/subtitles/data discarded",
        "timestamp_basis": "observed source-frame integer PTS * showinfo time base, relative to first decoded frame after setpts=PTS-STARTPTS",
        "cadence_basis": "observed variable spacing between distinct source frames; sample_fps is requested, not exact observed cadence; no resampler duplicates",
        "minimum_spacing_seconds": spacing, "observed_sample_span_seconds": None,
        "source_timestamps_seconds": [], "source_duration_seconds": None,
        "coverage": "not_established",
    }
    if interval is not None:
        sampling.update(requested_interval_seconds=list(interval),
                        requested_window_seconds=interval[1] - interval[0],
                        boundary_rule="start inclusive, end exclusive; source-frame timestamps retained",
                        coverage="sampled_interval_not_full_frame_coverage")
    if timestamp_origin == "container":
        sampling["timestamp_basis"] = "observed source-frame PTS preserved with copyts; caller verifies zero-origin container; video delay is retained"

    def records(values=None, status="error", reason=None, diagnostic=None):
        result = {}
        for field, definition in DEFINITIONS.items():
            evidence = {"definition": definition, "sampling": dict(sampling),
                        "ffmpeg_binary": binary, "algorithm": "dbp-visual-rgb-container-interval-v1"
                        if timestamp_origin == "container" else "dbp-visual-rgb-interval-v2" if interval is not None else "dbp-visual-rgb-v2"}
            if reason:
                evidence["reason"] = reason
            if diagnostic:
                evidence["diagnostic"] = diagnostic[:1024]
            result[field] = {"value": values[field] if values else None,
                             "processing_status": status, "evidence": [evidence]}
        return result

    try:
        raw, stderr = (run_process or _run_bounded)(
            command, deadline, max_raw_bytes + max_diagnostics, pass_fds=pass_fds,
            process_group=False, capture_stderr=True)
        _remaining(deadline)
        if len(stderr) > max_diagnostics:
            raise ValueError("FFmpeg diagnostic output limit exceeded")
        if len(raw) > max_raw_bytes or len(raw) % frame_bytes:
            return records(reason="invalid_raw_output")
        frame_count = len(raw) // frame_bytes
        timestamps = _source_timestamps(stderr, frame_count, width, spacing, window, interval)
        sampling.update({
            "frame_count": frame_count, "frame_limit_reached": frame_count == max_frames,
            "source_timestamps_seconds": timestamps,
            "observed_sample_span_seconds": timestamps[-1] - timestamps[0] if frame_count else None,
        })
        if not frame_count:
            return records(status="missing", reason="no_sampled_frames")
        brightness_total = 0.0
        contrast_total = 0.0
        colorfulness_total = 0.0
        temporal_total = 0.0
        previous = None
        for offset in range(0, len(raw), frame_bytes):
            _remaining(deadline)
            lumas, brightness, contrast, colorfulness = _frame_measurements(
                memoryview(raw)[offset:offset + frame_bytes])
            brightness_total += brightness
            contrast_total += contrast
            colorfulness_total += colorfulness
            if previous is not None:
                temporal_total += sum(abs(current - earlier) for current, earlier in
                                      zip(lumas, previous)) / len(lumas) / LUMA_SCALE
            previous = lumas
        _remaining(deadline)
        values = {
            "brightness_mean": min(1.0, max(0.0, brightness_total / frame_count)),
            "contrast_mean": min(0.5, max(0.0, contrast_total / frame_count)),
            "colorfulness_mean": colorfulness_total / frame_count,
            "temporal_change_mean": min(1.0, temporal_total / (frame_count - 1)) if frame_count > 1 else None,
        }
        result = records(values, "complete")
        if frame_count == 1:
            result["temporal_change_mean"]["processing_status"] = "missing"
            result["temporal_change_mean"]["evidence"][0]["reason"] = "insufficient_sampled_frames"
        return result
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return records(reason="decode_failed", diagnostic=str(error))
