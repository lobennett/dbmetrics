"""Single-video CPU measurements using the existing bounded algorithms."""

import hashlib
import json
import math
from pathlib import Path
import sys
import time

from . import _runtime

METRICS = _runtime.ALL_FIELDS


def measure_video(path: str | Path, *, clip_id: str | None = None,
                  stages: tuple[str, ...] = tuple(_runtime.STAGE_FIELDS),
                  timeout: float = 30, ffmpeg: str = 'ffmpeg',
                  ffprobe: str = 'ffprobe', cut_threshold: float = 0.3) -> list[dict]:
    """Return value, status and evidence for each requested metric.

    Timeout applies to each stage, including hashing. Inputs are read-only;
    missing or failed measurements have null values, never fabricated zeros.
    This function neither imports results into the website nor launches jobs.
    """
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError('timeout must be finite and in (0, 300]')
    if type(cut_threshold) not in (int, float) or not math.isfinite(cut_threshold) or not 0 <= cut_threshold <= 1:
        raise ValueError('cut_threshold must be finite and in [0, 1]')
    if isinstance(stages, str) or not stages or any(stage not in _runtime.STAGE_FIELDS for stage in stages):
        raise ValueError('choose probe, visual, audio, technical or ffmpeg_cuts stages')
    if len(set(stages)) != len(stages):
        raise ValueError('stages must not contain duplicates')
    path = Path(path).absolute()
    clip_id = path.stem if clip_id is None else clip_id
    if not isinstance(clip_id, str) or not clip_id or len(clip_id) > 240:
        raise ValueError('clip_id must be a nonempty string of at most 240 characters')
    records = []
    code_hash = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob('*.py')):
        code_hash.update(source.name.encode())
        code_hash.update(source.read_bytes())
    provenance = {'package': 'dbp-metrics', 'version': '0.3.0',
                  'code_sha256': code_hash.hexdigest(), 'accuracy_validated': False}
    for stage in stages:
        options = {'clip': {'clip_id': clip_id}, 'media_root': str(path.parent),
                   'media_kind': 'original', 'filename': path.name, 'stage': stage,
                   'ffprobe': str(ffprobe), 'ffmpeg': str(ffmpeg), 'timeout': timeout,
                   'max_media_bytes': _runtime.MAX_MEDIA_BYTES,
                   'ffmpeg_cut_threshold': cut_threshold}
        command = [sys.executable, '-I', '-B', str(Path(__file__).with_name('_worker.py')),
                   json.dumps(options)]
        try:
            raw, diagnostics = _runtime.run_bounded(
                command, time.monotonic() + timeout, _runtime.MAX_WORKER_BYTES,
                capture_stderr=True)
            batch = _runtime.strict_json(raw)['records']
            if diagnostics and batch:
                batch[0]['evidence'][0]['worker_diagnostics'] = diagnostics[:8192].decode('utf-8', 'replace')
        except Exception as error:
            batch = _runtime.empty_records(clip_id, path, 'original', 'error', str(error)[:512],
                                           fields=_runtime.STAGE_FIELDS[stage])
        for record in batch:
            record['evidence'][0]['run_provenance'] = dict(provenance)
        records.extend(batch)
    return records
