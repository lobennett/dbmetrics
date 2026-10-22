"""Write standalone scene-continuity JSON or resumable manifest-shard JSONL."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys


from dbmetrics.experimental.scene_continuity import Config, SCHEMA, canonical, make_policy, measure, media_identity


def load_work(options):
    if options.clip:
        return [{'clip_id': str(options.clip.resolve()), 'media_path': options.clip}], None
    raw = options.manifest.read_bytes()
    rows = []
    identifiers = set()
    for line in raw.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (not isinstance(row, dict) or not isinstance(row.get('clip_id'), str)
                or not row['clip_id'] or row['clip_id'] in identifiers
                or not isinstance(row.get('media_path'), str) or not row['media_path']
                or set(row) != {'clip_id', 'media_path'}):
            raise ValueError('Manifest needs unique clip_id and media_path strings')
        identifiers.add(row['clip_id'])
        path = Path(row['media_path'])
        rows.append({'clip_id': row['clip_id'], 'media_path': path if path.is_absolute()
                     else options.manifest.resolve().parent / path})
    if not rows:
        raise ValueError('Empty manifest shard')
    return rows, hashlib.sha256(raw).hexdigest()


def run(options):
    config = Config(max_frames=options.max_frames, width=options.width, height=options.height,
                    timeout=options.timeout, media_scope=options.media_scope)
    policy = make_policy(config, options.ffmpeg)
    rows, manifest_sha256 = load_work(options)
    output_path = options.output.resolve()
    if any(row['media_path'].resolve() == output_path for row in rows) or (
            options.manifest and options.manifest.resolve() == output_path):
        raise ValueError('Output must not be an input')
    with options.output.open('a+b') as output:
        fcntl.flock(output, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.seek(0)
        existing = output.read()
        if existing and not existing.endswith(b'\n'):
            raise ValueError('Incomplete output tail; preserve and recover to a new output path')
        completed = [json.loads(line) for line in existing.splitlines()] if options.manifest else (
            [json.loads(existing)] if existing else [])
        if len(completed) > len(rows):
            raise ValueError('Output has unexpected records')
        for row, result in zip(rows, completed):
            if (not isinstance(result, dict) or result.get('schema_version') != SCHEMA
                    or result.get('provenance') != policy
                    or result.get('manifest_sha256') != manifest_sha256
                    or result.get('clip_id') != row['clip_id']
                    or result.get('media') != media_identity(row['media_path'])):
                raise ValueError('Existing output has different policy, manifest, or media')
        for row in rows[len(completed):]:
            result = measure(row['media_path'], config, _policy=policy)
            result.update(clip_id=row['clip_id'], manifest_sha256=manifest_sha256)
            output.write(canonical(result) + b'\n')
            output.flush()
            os.fsync(output.fileno())
    return {'output': str(options.output), 'completed': len(rows), 'resumed': len(completed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--clip', type=Path)
    inputs.add_argument('--manifest', type=Path, help='JSONL shard: clip_id, media_path')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--max-frames', type=int, default=1800)
    parser.add_argument('--width', type=int, default=96)
    parser.add_argument('--height', type=int, default=54)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--media-scope', choices=('full', 'prefix', 'unknown'), default='unknown')
    options = parser.parse_args(argv)
    try:
        print(json.dumps(run(options), sort_keys=True))
    except (ValueError, OSError, TimeoutError) as error:
        print(f'scene continuity: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
