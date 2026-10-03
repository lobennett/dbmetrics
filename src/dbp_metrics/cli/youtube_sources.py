"""Export archived counts or explicitly retrieve current YouTube source statistics."""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import time

from dbp_metrics.source_statistics import METRICS, catalog_record, read_catalog, youtube_record


def fetch_youtube(source_id):
    from yt_dlp import YoutubeDL
    from yt_dlp.version import __version__

    options = {
        "quiet": True, "skip_download": True, "noplaylist": True, "cachedir": False,
        "socket_timeout": 20, "retries": 0, "extractor_retries": 0,
        "ignore_no_formats_error": True, "getcomments": False,
        "extractor_args": {"youtube": {"skip": ["hls", "dash"]}},
    }
    with YoutubeDL(options) as extractor:
        info = extractor.extract_info("https://www.youtube.com/watch?v=" + source_id, download=False)
    return info, __version__


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _existing(output, provider, catalog_sha256, source_ids):
    seen = set()
    if not output.exists():
        return seen
    with output.open() as handle:
        for line in handle:
            row = json.loads(line)
            if (type(row) is not dict or row.get("schema_version") != "dbp-source-statistics-v1"
                    or row.get("scope") != "source_video"
                    or row.get("source_id") not in source_ids or row["source_id"] in seen
                    or row.get("provenance", {}).get("provider") != provider
                    or row["provenance"].get("catalog_sha256") != catalog_sha256
                    or row.get("status") not in ("ok", "error")
                    or set(row.get("metrics", {})) != set(METRICS)):
                raise ValueError("Existing output has different provenance or invalid records; choose a new output")
            seen.add(row["source_id"])
    if seen and output.read_bytes()[-1:] != b"\n":
        raise ValueError("Existing output ends without a newline; inspect it before resuming")
    return seen


def wait_for_request(clock, delay):
    """Serialize request starts across workers, without serializing HTTP calls."""
    clock.parent.mkdir(parents=True, exist_ok=True)
    with clock.open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        previous = float(handle.read() or '0')
        remaining = previous + delay - time.time()
        if remaining > 0:
            time.sleep(remaining)
        handle.seek(0)
        handle.truncate()
        handle.write(str(time.time()))
        handle.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", choices=("catalog", "youtube"), default="catalog")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=10, help="New source lookups; default 10")
    selection.add_argument("--all", action="store_true", help="Process every remaining source")
    parser.add_argument("--delay", type=float, default=2, help="Seconds between YouTube source lookups")
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--resume-from', type=Path, help='Read-only checkpoint shared by all shards')
    parser.add_argument('--request-clock', type=Path, help='Shared request-start clock for parallel workers')
    parser.add_argument('--stop-file', type=Path, help='Shared halt signal on access or rate-limit responses')
    args = parser.parse_args(argv)
    if args.limit < 1 or not math.isfinite(args.delay) or args.delay < 0:
        parser.error("limit must be positive and delay must be finite and nonnegative")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error('shard-index must be between zero and shard-count minus one')
    if args.output.resolve() == args.catalog.resolve():
        parser.error("Output must not be the catalog")
    if args.provider == "youtube":
        try:
            import yt_dlp
        except ImportError:
            parser.error("Install the optional youtube dependency: uv sync --extra youtube")
    rows = read_catalog(args.catalog)
    digest = _digest(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    processed = 0
    with args.output.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another collector is writing this output; leave that run active") from None
        seen = _existing(args.output, args.provider, digest, {row["source_id"] for row in rows})
        if args.resume_from:
            seen |= _existing(args.resume_from, args.provider, digest, {row['source_id'] for row in rows})
        pending = [row for row in rows if row["source_id"] not in seen
                   and int(hashlib.sha256(row['source_id'].encode()).hexdigest(), 16)
                   % args.shard_count == args.shard_index]
        if not args.all:
            pending = pending[:args.limit]
        for row in pending:
            if args.stop_file and args.stop_file.exists():
                errors += 1
                break
            if args.request_clock and args.provider == 'youtube':
                wait_for_request(args.request_clock, args.delay)
            elif processed and args.provider == "youtube":
                time.sleep(args.delay)
            if args.stop_file and args.stop_file.exists():
                errors += 1
                break
            observed_at = datetime.now(timezone.utc).isoformat()
            stop = False
            try:
                if args.provider == "catalog":
                    record = catalog_record(row, catalog_sha256=digest)
                else:
                    info, version = fetch_youtube(row["source_id"])
                    record = youtube_record(row["source_id"], info, observed_at=observed_at,
                                            extractor_version=version)
                    record["provenance"]["catalog_sha256"] = digest
            except Exception as error:
                errors += 1
                message = str(error)[:500]
                stop = any(marker in str(error).lower() for marker in (
                    "429", "too many requests", "http error 403", "forbidden", "captcha", "rate limit",
                    "sign in", "not a bot"))
                record = {
                    "schema_version": "dbp-source-statistics-v1", "source_id": row["source_id"],
                    "scope": "source_video", "channel_id": row["channel_id"],
                    "metrics": {metric: None for metric in METRICS}, "uploaded_on": None,
                    "observed_at": observed_at, "status": "error", "error": message,
                    "provenance": {"provider": args.provider, "catalog_sha256": digest},
                }
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            processed += 1
            if args.provider == "youtube":
                print(json.dumps({"source_id": row["source_id"], "status": record["status"],
                                  "metrics": record["metrics"]}), flush=True)
            if stop:
                if args.stop_file:
                    args.stop_file.write_text('Stopped on a YouTube access/rate-limit response.\n')
                print("Stopped on a YouTube access/rate-limit response; no bypass or repeated retries.")
                break
    print(json.dumps({"output": str(args.output), "sources": len(rows), "already_recorded": len(seen),
                      "processed": processed, "errors": errors,
                      "remaining": len(pending) - processed, 'shard_index': args.shard_index,
                      'shard_count': args.shard_count}), flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
