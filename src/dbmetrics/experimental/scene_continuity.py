"""CPU-only, consecutive-frame diagnostics; candidates are not proven scene cuts."""

from dataclasses import asdict, dataclass
import hashlib
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
import time

import numpy as np


SCHEMA = 'dbp-scene-continuity-v1'
METHOD = 'adjacent-encoded-luma-local-contrast-v1'


@dataclass(frozen=True)
class Config:
    width: int = 96
    height: int = 54
    max_frames: int = 1800
    timeout: float = 30.0
    baseline_radius: int = 6
    minimum_baseline: int = 3
    absolute_change: float = 0.15
    excess_change: float = 0.10
    contrast_ratio: float = 3.0
    baseline_floor: float = 0.01
    sustained_transitions: int = 3
    return_change: float = 0.04
    media_scope: str = 'unknown'

    def __post_init__(self):
        for name, lower, upper in (('width', 8, 512), ('height', 8, 512),
                                   ('max_frames', 2, 20000), ('baseline_radius', 3, 120),
                                   ('minimum_baseline', 1, 238),
                                   ('sustained_transitions', 3, 100)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f'Invalid {name}')
        for name, lower, upper in (('timeout', 0.01, 3600), ('absolute_change', 0.0001, 1),
                                   ('excess_change', 0, 1), ('contrast_ratio', 1, 1000),
                                   ('baseline_floor', 0.0001, 1), ('return_change', 0, 1)):
            value = getattr(self, name)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not lower <= value <= upper):
                raise ValueError(f'Invalid {name}')
        if (self.width * self.height * (self.max_frames + 1) > 64 * 1024 * 1024
                or self.minimum_baseline > 2 * (self.baseline_radius - 1)):
            raise ValueError('Frame memory budget or baseline configuration exceeded')
        if self.media_scope not in ('full', 'prefix', 'unknown'):
            raise ValueError('Invalid media_scope')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_signature(path):
    info = Path(path).stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def media_identity(path):
    path = Path(path).resolve(strict=True)
    before = file_signature(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('Media must be a regular local file')
    digest = file_sha256(path)
    if file_signature(path) != before:
        raise ValueError('Media changed while hashing')
    return {'path': str(path), 'sha256': digest, 'size_bytes': before[2]}


def run_bounded(command, timeout, stdout_limit, stderr_limit=8 * 1024 * 1024):
    """Drain both pipes with byte caps and a wall deadline; kill/reap on failure."""
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    buffers = [bytearray(), bytearray()]
    try:
        with selectors.DefaultSelector() as selector:
            for index, stream in enumerate((process.stdout, process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, index)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Decoder wall timeout')
                for key, _ in selector.select(remaining):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    buffer = buffers[key.data]
                    if len(buffer) + len(block) > (stdout_limit, stderr_limit)[key.data]:
                        raise ValueError('Decoder output byte limit exceeded')
                    buffer.extend(block)
            try:
                returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as error:
                raise TimeoutError('Decoder wall timeout') from error
        if returncode:
            message = buffers[1][-2000:].decode('utf-8', errors='replace')
            raise ValueError(f'Decoder failed ({returncode}): {message}')
        return bytes(buffers[0]), bytes(buffers[1])
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        process.stdout.close()
        process.stderr.close()


def make_policy(config, ffmpeg='ffmpeg'):
    executable = shutil.which(str(ffmpeg))
    if not executable:
        raise ValueError('ffmpeg executable not found')
    version, _ = run_bounded([executable, '-version'], config.timeout, 65536)
    policy = {'schema_version': SCHEMA, 'method': METHOD, 'config': asdict(config),
              'ffmpeg_path': str(Path(executable).resolve()),
              'ffmpeg_version': version.decode('utf-8', errors='replace').strip(),
              'numpy_version': np.__version__, 'implementation_sha256': file_sha256(__file__)}
    return {**policy, 'policy_sha256': hashlib.sha256(canonical(policy)).hexdigest()}


def analyze_frames(frames, timestamps, config):
    """Diagnose encoded (not linear-light) luma MAE using actual adjacent frames."""
    frames = np.asarray(frames)
    timestamps = np.asarray(timestamps, dtype=float)
    if (frames.ndim != 3 or frames.dtype != np.uint8 or len(frames) < 1
            or frames.shape[1] == 0 or frames.shape[2] == 0
            or timestamps.shape != (len(frames),) or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0)):
        raise ValueError('Expected uint8 luma frames and finite, strictly increasing timestamps')
    changes = []
    returns = []
    previous = frames[0].astype(np.int16)
    for index in range(1, len(frames)):
        current = frames[index].astype(np.int16)
        changes.append(float(np.abs(current - previous).mean() / 255))
        if index >= 2:
            returns.append(float(np.abs(current - frames[index - 2].astype(np.int16)).mean() / 255))
        previous = current
    sustained = set()
    runs = []
    start = 0
    while start < len(changes):
        if changes[start] < config.absolute_change:
            start += 1
            continue
        end = start + 1
        while end < len(changes) and changes[end] >= config.absolute_change:
            end += 1
        if end - start >= config.sustained_transitions:
            sustained.update(range(start, end))
            runs.append({'start_seconds': float(timestamps[start]),
                         'end_seconds': float(timestamps[end]), 'transition_count': end - start})
        start = end
    flicker = set()
    flicker_returns = []
    for index, return_change in enumerate(returns, start=1):
        if (changes[index - 1] >= config.absolute_change
                and changes[index] >= config.absolute_change and return_change <= config.return_change):
            flicker.update((index - 1, index))
            flicker_returns.append({'start_seconds': float(timestamps[index - 1]),
                                    'return_seconds': float(timestamps[index + 1]),
                                    'two_frame_change': return_change})
    trace = []
    candidates = []
    for index, change in enumerate(changes):
        neighbors = [changes[other] for other in range(max(0, index - config.baseline_radius),
                     min(len(changes), index + config.baseline_radius + 1)) if abs(other - index) > 1]
        baseline = float(np.median(neighbors)) if neighbors else None
        supported = len(neighbors) >= config.minimum_baseline
        ratio = change / max(config.baseline_floor, baseline) if baseline is not None else None
        excess = change - baseline if baseline is not None else None
        candidate = (supported and change >= config.absolute_change
                     and excess >= config.excess_change and ratio >= config.contrast_ratio
                     and index not in sustained)
        row = {'frame_index': index + 1, 'previous_timestamp_seconds': float(timestamps[index]),
               'timestamp_seconds': float(timestamps[index + 1]),
               'interval_seconds': float(timestamps[index + 1] - timestamps[index]),
               'change': change, 'local_baseline': baseline, 'baseline_count': len(neighbors),
               'baseline_supported': supported, 'excess_change': excess, 'contrast_ratio': ratio,
               'sustained_change': index in sustained, 'flicker_ambiguous': index in flicker,
               'abrupt_change_candidate': bool(candidate)}
        trace.append(row)
        if candidate:
            candidates.append(dict(row))
    supported_count = sum(row['baseline_supported'] for row in trace)
    return {'status': 'ok' if supported_count else 'insufficient_evidence',
            'trace': trace, 'candidates': candidates, 'sustained_change_runs': runs,
            'flicker_returns': flicker_returns,
            'summary': {'transition_count': len(changes), 'candidate_count': len(candidates),
                        'supported_transition_count': supported_count,
                        'unsupported_transition_count': len(changes) - supported_count,
                        'mean_change': float(np.mean(changes)) if changes else None,
                        'max_change': max(changes) if changes else None,
                        'interpretation': 'diagnostic_only_not_proven_scene_cuts'}}


def measure(path, config=None, ffmpeg='ffmpeg', *, _policy=None):
    config = config or Config()
    policy = _policy or make_policy(config, ffmpeg)
    media = media_identity(path)
    before = file_signature(media['path'])
    frame_bytes = config.width * config.height
    filters = (f'format=yuv420p,extractplanes=y,scale={config.width}:{config.height}:flags=area,'
               'showinfo')
    command = [policy['ffmpeg_path'], '-hide_banner', '-nostdin', '-loglevel', 'info',
               '-xerror', '-err_detect', 'explode', '-threads', '1', '-filter_threads', '1',
               '-protocol_whitelist', 'file,pipe', '-copyts', '-i', media['path'],
               '-map', '0:v:0', '-an', '-sn', '-dn', '-vf', filters, '-fps_mode', 'passthrough',
               '-frames:v', str(config.max_frames + 1), '-threads', '1',
               '-pix_fmt', 'gray', '-f', 'rawvideo', 'pipe:1']
    raw, stderr = run_bounded(command, config.timeout, frame_bytes * (config.max_frames + 1))
    if file_signature(media['path']) != before or media_identity(media['path']) != media:
        raise ValueError('Media changed during decoding')
    if not raw or len(raw) % frame_bytes:
        raise ValueError('No complete decoded video frames')
    count = len(raw) // frame_bytes
    matches = re.findall(rb'\bn:\s*(\d+)\s+pts:\s*\S+\s+pts_time:([^\s]+)', stderr)
    if len(matches) < count or any(int(number) != index for index, (number, _) in enumerate(matches[:count])):
        raise ValueError('Missing or unaligned frame timestamps')
    timestamps = [float(value) for _, value in matches[:count]]
    if not all(math.isfinite(value) for value in timestamps) or any(
            right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError('Invalid or non-increasing presentation timestamps')
    prefix = count > config.max_frames
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(count, config.height, config.width)
    used = min(count, config.max_frames)
    result = analyze_frames(frames[:used], timestamps[:used], config)
    return {'schema_version': SCHEMA, 'provenance': policy, 'media': media,
            'coverage': {'kind': 'prefix' if prefix else 'full', 'frame_count': used,
                         'decoded_frame_count': count, 'consecutive': True, 'stride': 1,
                         'stop_reason': 'max_frames' if prefix else 'decoder_eof',
                         'first_timestamp_seconds': timestamps[0],
                         'last_timestamp_seconds': timestamps[used - 1],
                         'lookahead_timestamp_seconds': timestamps[-1] if prefix else None,
                         'timestamp_basis': 'source_presentation_seconds',
                         'declared_media_scope': config.media_scope,
                         'source_scope': 'prefix' if prefix else config.media_scope},
            **result}
