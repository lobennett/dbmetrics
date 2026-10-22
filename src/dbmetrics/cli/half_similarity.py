"""Offline single-file / append-only resumable JSONL similarity runner."""

import argparse
import json
import os
from pathlib import Path
import sys


from dbmetrics.experimental.half_similarity import Config, VJEPABackend, measure, provenance, request_key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("media", nargs="?", type=Path)
    source.add_argument("--manifest", type=Path, help='JSONL records: {"id": "unique", "path": "clip.mp4"}')
    parser.add_argument("--output", type=Path, help="JSONL results (default stdout for single clip)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--vjepa", action="store_true", help="Use pinned, already-cached weights; never download")
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--frames-per-half", type=int, default=64)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    try:
        config = Config(frames_per_half=args.frames_per_half, bins=args.bins, vjepa=args.vjepa,
                        device=args.device, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe)
        if (args.manifest or args.resume) and not args.output:
            raise ValueError("--manifest and --resume require --output")
        if args.manifest:
            rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
            if any(not isinstance(row, dict) or not isinstance(row.get("id"), str) or
                   not row["id"] or not isinstance(row.get("path"), str) or not row["path"] for row in rows):
                raise ValueError("Each manifest record requires nonempty string id and path")
            if len({row["id"] for row in rows}) != len(rows):
                raise ValueError("Manifest ids must be unique")
            rows = [{"id": row["id"], "path": args.manifest.parent / row["path"]} for row in rows]
        else:
            rows = [{"id": str(args.media), "path": args.media}]
        if args.output and args.output.resolve() in {Path(row["path"]).resolve() for row in rows} | (
                {args.manifest.resolve()} if args.manifest else set()):
            raise ValueError("Output must not overwrite input media or manifest")
        completed = set()
        if args.resume and args.output.exists():
            with args.output.open("rb+") as previous:
                while True:
                    offset = previous.tell()
                    line = previous.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        previous.truncate(offset)
                        break
                    record = json.loads(line)
                    if record["visual_histogram_cosine"]["status"] == "ok" and record["vjepa_cosine"]["status"] in ("ok", "not_requested"):
                        completed.add((record["id"], record["request_key"]))
        output = args.output.open("a" if args.resume else "x") if args.output else sys.stdout
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    backend = VJEPABackend(config.device) if config.vjepa else None
    failed = False
    try:
        for row in rows:
            if completed:
                try:
                    key = request_key(provenance(row["path"], config, backend))
                except Exception:
                    key = None
                if (row["id"], key) in completed:
                    continue
            record = {"id": row["id"], **measure(row["path"], config, backend=backend)}
            output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
            if args.output:
                os.fsync(output.fileno())
            failed |= any(record[field]["status"] == "unknown" for field in ("visual_histogram_cosine", "vjepa_cosine"))
    finally:
        if args.output:
            output.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
