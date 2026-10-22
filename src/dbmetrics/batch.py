"""One-GPU, offline VLM supervisor with exclusive, hash-checked checkpoints.

The worker retains verified weights between clips, not video tensors. A deadline
or malformed protocol kills its process group before another worker is created.
Checkpoints are private execution evidence, never independent validation labels.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
import fcntl
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "dbmetrics"

from .legacy import (MAX_MEDIA_BYTES, MAX_WORKER_BYTES, SEMANTIC_FIELDS,
                           bounded_number, empty_records, extract_clip,
                           load_manifest, relative_media_path, remaining, strict_json)


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def write_checkpoint(path, payload):
    content = canonical({"payload": payload, "sha256": hashlib.sha256(canonical(payload)).hexdigest()})
    if len(content) > MAX_WORKER_BYTES:
        raise ValueError("checkpoint exceeds size bound")
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=Path(path).parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        os.unlink(temporary)


@contextmanager
def run_lock(directory):
    descriptor = os.open(directory / ".run.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("this output directory is already running") from error
        yield
    finally:
        os.close(descriptor)


def read_checkpoint(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("checkpoint must be a regular file")
        raw = stream.read(MAX_WORKER_BYTES + 1)
    if len(raw) > MAX_WORKER_BYTES:
        raise ValueError("checkpoint exceeds size bound")
    wrapped = strict_json(raw)
    if (not isinstance(wrapped, dict) or set(wrapped) != {"payload", "sha256"}
            or hashlib.sha256(canonical(wrapped["payload"])).hexdigest() != wrapped["sha256"]):
        raise ValueError("checkpoint hash mismatch")
    return wrapped["payload"]


class JsonWorker:
    def __init__(self, command):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, start_new_session=True, bufsize=0)

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        self.close()

    def close(self):
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def request(self, payload, deadline):
        request = canonical(payload) + b"\n"
        if len(request) > 8192:
            raise ValueError("worker request exceeds bound")
        self.process.stdin.write(request)
        stdout, stderr = bytearray(), bytearray()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ, stdout)
            selector.register(self.process.stderr, selectors.EVENT_READ, stderr)
            while True:
                for key, unused in selector.select(remaining(deadline)):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        if key.fileobj is self.process.stdout:
                            raise ValueError("worker exited before a complete response: " + stderr[-1000:].decode("utf-8", "replace"))
                        continue
                    key.data.extend(chunk)
                    if len(stdout) + len(stderr) > MAX_WORKER_BYTES:
                        raise ValueError("worker response/diagnostics exceeded byte limit")
                    if b"\n" in stdout:
                        line, trailing = stdout.split(b"\n", 1)
                        if trailing.strip():
                            raise ValueError("unexpected additional worker response")
                        response = strict_json(line)
                        if not isinstance(response, dict):
                            raise ValueError("worker response must be an object")
                        return response


def worker_main(config):
    from ._vlm import VLMRuntime

    runtime = VLMRuntime()
    for raw in sys.stdin:
        if len(raw.encode("utf-8")) > 8192:
            raise ValueError("oversized worker input")
        request = strict_json(raw)
        options = dict(config)
        options["clip"] = request["clip"]
        options["deadline"] = time.monotonic() + request["timeout"]
        options["vlm_options"] = dict(options["vlm_options"], runtime=runtime)
        started = time.monotonic()
        with redirect_stdout(sys.stderr):
            records, failed = extract_clip(**options)
        records = [record for record in records if record["field_id"] in SEMANTIC_FIELDS]
        response = {"records": records, "failed": failed, "elapsed_seconds": time.monotonic() - started}
        content = canonical(response)
        if len(content) > MAX_WORKER_BYTES - 16384:
            raise ValueError("worker response exceeds bound")
        print(content.decode("utf-8"), flush=True)


def run_batch(arguments):
    from ._vlm import PROMPT, sample_times, snapshot_path, validate_device

    bounded_number(arguments.timeout, 1, 600, "timeout")
    bounded_number(arguments.wall_seconds, 1, 172800, "wall-seconds")
    bounded_number(arguments.limit, 1, 10000, "limit")
    bounded_number(arguments.offset, 0, 9999, "offset")
    validate_device(arguments.device)
    sample_times(1, arguments.frames)
    model = snapshot_path(arguments.local_vlm_dir, arguments.model_revision, arguments.model_id)
    root = arguments.media_root.resolve(strict=True)
    output = arguments.output_dir.resolve(strict=True)
    metadata = output.stat()
    if (not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700 or root == output or root in output.parents
            or any((ancestor / ".git").exists() for ancestor in (output, *output.parents))):
        raise ValueError("use an owned mode-0700 output directory outside Git and media")
    with run_lock(output):
        return run_locked_batch(arguments, model, root, output)


def run_locked_batch(arguments, model, root, output):
    from ._vlm import PROMPT, validate_local_model

    deadline = time.monotonic() + arguments.wall_seconds
    verified = validate_local_model(model, arguments.model_revision,
                                    min(deadline, time.monotonic() + arguments.timeout), arguments.model_id)
    clips, manifest_hash = load_manifest(arguments.manifest)
    selected = clips[arguments.offset:arguments.offset + arguments.limit]
    if not selected:
        raise ValueError("empty manifest selection")
    config = {"media_root": str(root), "media_kind": "rendition", "ffprobe": arguments.ffprobe,
              "ffmpeg": arguments.ffmpeg, "max_media_bytes": MAX_MEDIA_BYTES, "cuts": False,
              "cut_threshold": 27.0, "stage": "vlm",
              "vlm_options": {"local_model_dir": str(model), "revision": arguments.model_revision,
                              "model_id": arguments.model_id, "device": arguments.device,
                              "ffmpeg": arguments.ffmpeg, "frame_count": arguments.frames,
                              "expected_file_sha256": verified["file_sha256"]}}
    code_files = sorted(Path(__file__).parent.rglob('*.py'))
    identity = {"schema_version": "dbt1-vlm-batch-v1", "manifest_sha256": manifest_hash, "config": config,
                "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "timeout": arguments.timeout,
                "code_sha256": {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in code_files}}
    fingerprint = hashlib.sha256(canonical(identity)).hexdigest()
    identity_path = output / "identity.json"
    if identity_path.exists():
        if read_checkpoint(identity_path) != identity:
            raise ValueError("resume identity differs; use a new output directory")
    else:
        write_checkpoint(identity_path, identity)
    worker, consecutive_errors = None, 0
    completed = errors = skipped = 0
    try:
        for offset, clip in enumerate(selected, arguments.offset):
            checkpoint = output / f"clip-{offset:05d}.json"
            if checkpoint.exists():
                saved = read_checkpoint(checkpoint)
                if saved.get("fingerprint") != fingerprint or saved.get("clip_id") != clip["clip_id"]:
                    raise ValueError("resume checkpoint identity mismatch")
                if type(saved.get("failed")) is not bool:
                    raise ValueError("resume checkpoint needs explicit failure status")
                errors += int(saved["failed"])
                skipped += 1
                continue
            remaining(deadline)
            if worker is None:
                command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--worker", canonical(config).decode()]
                worker = JsonWorker(command)
            started = time.monotonic()
            try:
                response = worker.request({"clip": clip, "timeout": arguments.timeout},
                                          min(deadline, started + arguments.timeout))
                records = response["records"]
                if (type(response.get("failed")) is not bool or not isinstance(records, list)
                        or len(records) != len(SEMANTIC_FIELDS)
                        or {record.get("field_id") for record in records} != set(SEMANTIC_FIELDS)
                        or any(record.get("clip_id") != clip["clip_id"] for record in records)):
                    raise ValueError("worker clip/field identity mismatch")
            except (OSError, ValueError, KeyError, TypeError) as error:
                path = root / relative_media_path(clip["clip_id"], "rendition")
                records = [record for record in empty_records(clip["clip_id"], path, "rendition", "error", str(error)[:512])
                           if record["field_id"] in SEMANTIC_FIELDS]
                response = {"failed": True}
            failed = response["failed"]
            payload = {"schema_version": "dbt1-pilot-measurements-v1", "manifest_sha256": manifest_hash,
                       "run_id": "vlm-batch-" + fingerprint[:16], "fingerprint": fingerprint,
                       "clip_id": clip["clip_id"], "records": records, "failed": failed,
                       "elapsed_seconds": time.monotonic() - started}
            write_checkpoint(checkpoint, payload)
            completed += 1
            errors += int(failed)
            consecutive_errors = consecutive_errors + 1 if failed else 0
            print(json.dumps({"offset": offset, "clip_id": clip["clip_id"], "failed": failed,
                              "elapsed_seconds": round(payload["elapsed_seconds"], 3),
                              "completed": completed, "errors": errors, "resumed": skipped}), flush=True)
            if failed:
                worker.close()
                worker = None
            if consecutive_errors >= 3:
                raise ValueError("three consecutive extraction failures; stop instead of consuming the sample")
    finally:
        if worker is not None:
            worker.close()
    return int(errors > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-vlm-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--wall-seconds", type=float, default=86400)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=5000)
    try:
        return run_batch(parser.parse_args())
    except (OSError, ValueError, BatchInterrupted) as error:
        print("Batch stopped: " + str(error), file=sys.stderr)
        return 2


class BatchInterrupted(Exception):
    pass


def interrupted(signum, frame):
    raise BatchInterrupted("batch interrupted")


if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "dbmetrics"
    signal.signal(signal.SIGTERM, interrupted)
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker_main(strict_json(sys.argv[2]))
    else:
        raise SystemExit(main())
