"""Bounded CPU-only amplitude measurements on the first decoded audio stream.

The caller validates and holds the media descriptor, supplies its /dev/fd path
and pass_fds, and owns hashing and before/after identity checks. This module
does not discover media, probe it, import the application, or load models.
Input uses only the MOV/MP4 demuxer and local file/pipe protocols.

FFmpeg's layout-aware mono rematrix precedes measurement at 16000 Hz, signed
little-endian 16-bit PCM with no dither or asynchronous timeline filling.
These are normalized decoded mono amplitudes, not loudness or LUFS. Opposite
channel phases can cancel; other streams and video-only time are not measured.
RMS pools squared samples and divides by 32768. Successful evidence retains the
exact integer squared_sample_sum so RMS can be independently recomputed.
Silence is window RMS <= 0.001
in fixed nonoverlapping 320-sample blocks starting at the first decoded sample.
The final shorter window is included and weighted by its sample count.

At most 60 seconds plus one sentinel sample are retained. Reaching the sentinel
rejects the whole measurement, never returning prefix values. A successful
decode below that limit establishes full_decoded_audio coverage, not container
duration or PTS continuity. Empty/no-track input is missing, never silence;
malformed PCM, decoder errors and timeouts are error with null values. Invalid
resource controls raise ValueError before launching any subprocess.

run_process follows pilot_extract.run_bounded(command, deadline, max_output,
pass_fds=(), process_group=False, capture_stderr=True), returning stdout/stderr
bytes and raising on nonzero exit or resource limits. It must not silently
truncate output. The standalone fallback enforces the same combined byte and
absolute monotonic time caps. process_group=False preserves the parent worker's
killable group. The deadline includes metric computation and is never extended.
"""

import math
import os
import selectors
import signal
import struct
import subprocess
import time


ALGORITHM = "dbp-audio-pcm-v1"
SAMPLE_RATE_HZ = 16000
WINDOW_SAMPLES = 320
SILENCE_THRESHOLD_RMS = 0.001
NORMALIZATION_DIVISOR = 32768
MAX_DURATION_SECONDS = 60
MAX_STDERR_BYTES = 32768
DOWNMIX = ("FFmpeg default layout-aware mono rematrix, then s16 quantization; "
           "phase cancellation and clipping possible; not per-channel energy, "
           "not loudness or LUFS; no dither or asynchronous gap filling")
DEFINITIONS = {
    "audio_rms_amplitude": "sqrt(sum(s16_sample**2)/sample_count)/32768 on decoded mono 16000 Hz PCM; range 0..1; not loudness.",
    "audio_silence_fraction": "Fraction of decoded samples in nonoverlapping 320-sample (20 ms) blocks with normalized RMS <= 0.001; final partial block weighted by sample count; range 0..1.",
}


def _remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise ValueError("audio measurement timeout exceeded")
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
            if process.returncode != 0 or (stderr.strip() and not capture_stderr):
                raise ValueError("subprocess failed or reported errors: " +
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


def _no_audio_error(error):
    diagnostic = str(error).strip()
    prefix = "subprocess failed or reported errors: "
    if diagnostic.startswith(prefix):
        diagnostic = diagnostic[len(prefix):]
    return diagnostic == "Output file #0 does not contain any stream"


def measure_audio(media_path, *, ffmpeg="ffmpeg", deadline=None, pass_fds=(),
                  process_group=False, run_process=None, has_audio=None,
                  timeout=30, max_duration_seconds=60):
    """Return both amplitude records; has_audio is a probe bool or None.

    timeout is in (0, 300] seconds; max_duration_seconds is in [1/16000, 60].
    The sample budget is floor(max_duration_seconds * 16000), plus one sentinel.
    The default combined output cap is 1,952,770 bytes including diagnostics.
    Caller deadlines are absolute time.monotonic values, not wall-clock dates.
    """
    if (type(timeout) not in (int, float) or not math.isfinite(timeout)
            or not 0 < timeout <= 300):
        raise ValueError("timeout must be finite and in (0, 300]")
    if (type(max_duration_seconds) not in (int, float)
            or not math.isfinite(max_duration_seconds)
            or not 1 / SAMPLE_RATE_HZ <= max_duration_seconds <= MAX_DURATION_SECONDS):
        raise ValueError("max_duration_seconds must be finite and in [1/16000, 60]")
    if (deadline is not None and (type(deadline) not in (int, float)
                                 or not math.isfinite(deadline))):
        raise ValueError("deadline must be a finite monotonic timestamp or None")
    if has_audio is not None and type(has_audio) is not bool:
        raise ValueError("has_audio must be a boolean or None")
    if type(process_group) is not bool:
        raise ValueError("process_group must be a boolean")
    local_deadline = time.monotonic() + timeout
    deadline = local_deadline if deadline is None else min(deadline, local_deadline)
    max_samples = math.floor(max_duration_seconds * SAMPLE_RATE_HZ)
    max_raw_bytes = (max_samples + 1) * 2
    max_output = max_raw_bytes + MAX_STDERR_BYTES
    binary = os.fspath(ffmpeg)
    sampling = {
        "sample_rate_hz": SAMPLE_RATE_HZ, "channels": 1, "sample_format": "s16le",
        "normalization_divisor": NORMALIZATION_DIVISOR,
        "stream_selection": "0:a:0 (first audio stream)", "downmix": DOWNMIX,
        "silence_threshold_rms": SILENCE_THRESHOLD_RMS, "silence_comparison": "<=",
        "window_samples": WINDOW_SAMPLES, "window_seconds": WINDOW_SAMPLES / SAMPLE_RATE_HZ,
        "partial_window_weighting": "sample_count", "sample_count": None,
        "silent_samples": None, "squared_sample_sum": None,
        "window_count": None, "partial_window_samples": None,
        "decoded_duration_seconds": None, "coverage": "not_established",
        "sample_limit_reached": False, "max_duration_seconds": max_duration_seconds,
        "max_samples": max_samples, "sentinel_samples": 1, "max_raw_bytes": max_raw_bytes,
        "max_combined_output_bytes": max_output, "timeout_seconds": timeout,
    }

    def records(values=None, status="error", reason=None, diagnostic=None):
        result = {}
        for field, definition in DEFINITIONS.items():
            evidence = {"algorithm": ALGORITHM, "definition": definition,
                        "ffmpeg_binary": binary, "sampling": dict(sampling)}
            if reason:
                evidence["reason"] = reason
            if diagnostic:
                evidence["diagnostic"] = diagnostic[:1024]
            result[field] = {"value": values[field] if values is not None else None,
                             "processing_status": status, "evidence": [evidence]}
        return result

    if has_audio is False:
        return records(status="missing", reason="no_audio_track")
    transform = ("aresample=16000:async=0:dither_method=none,"
                 "aformat=sample_fmts=s16:channel_layouts=mono,"
                 f"atrim=end_sample={max_samples + 1},asetpts=N/SR/TB")
    command = [binary, "-hide_banner", "-loglevel", "error", "-nostdin", "-nostats", "-xerror",
               "-max_alloc", "67108864", "-threads", "1", "-err_detect", "explode",
               "-protocol_whitelist", "file,pipe", "-f", "mov", "-i", os.fspath(media_path),
               "-map", "0:a:0" if has_audio is True else "0:a:0?", "-vn", "-sn", "-dn",
               "-filter_threads", "1", "-af", transform,
               "-t", f"{(max_samples + 1) / SAMPLE_RATE_HZ:.9f}",
               "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-threads:a", "1",
               "-f", "s16le", "pipe:1"]
    try:
        _remaining(deadline)
        raw, stderr = (run_process or _run_bounded)(
            command, deadline, max_output, pass_fds=pass_fds,
            process_group=process_group, capture_stderr=True)
        _remaining(deadline)
        if not isinstance(stderr, bytes) or len(stderr) > MAX_STDERR_BYTES:
            raise ValueError("invalid or excessive FFmpeg diagnostic output")
        if stderr.strip():
            raise ValueError(stderr[:1024].decode("utf-8", "replace"))
        if not isinstance(raw, bytes) or len(raw) % 2:
            return records(reason="invalid_pcm_output")
        sample_count = len(raw) // 2
        if sample_count > max_samples:
            sampling["sample_limit_reached"] = True
            return records(status="missing", reason="sample_limit_exceeded")
        if not sample_count:
            return records(status="missing", reason="empty_audio" if has_audio else "no_decoded_audio")
        squared_total = 0
        silent_samples = 0
        for offset in range(0, len(raw), WINDOW_SAMPLES * 2):
            _remaining(deadline)
            block = raw[offset:offset + WINDOW_SAMPLES * 2]
            block_count = len(block) // 2
            block_squares = sum(sample[0] ** 2 for sample in struct.iter_unpack("<h", block))
            squared_total += block_squares
            if block_squares <= block_count * (SILENCE_THRESHOLD_RMS * NORMALIZATION_DIVISOR) ** 2:
                silent_samples += block_count
        _remaining(deadline)
        sampling.update({
            "sample_count": sample_count, "silent_samples": silent_samples,
            "squared_sample_sum": squared_total,
            "window_count": (sample_count + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES,
            "partial_window_samples": sample_count % WINDOW_SAMPLES,
            "decoded_duration_seconds": sample_count / SAMPLE_RATE_HZ,
            "coverage": "full_decoded_audio",
        })
        return records({
            "audio_rms_amplitude": math.sqrt(squared_total / sample_count) / NORMALIZATION_DIVISOR,
            "audio_silence_fraction": silent_samples / sample_count,
        }, status="complete")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        if has_audio is None and _no_audio_error(error):
            return records(status="missing", reason="no_audio_track")
        return records(reason="decode_failed", diagnostic=str(error))
