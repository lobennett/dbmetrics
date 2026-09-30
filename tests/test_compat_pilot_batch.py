"""Supervisor tests use a tiny protocol fixture, never a GPU or model."""

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import dbp_metrics.batch


class BatchTests(unittest.TestCase):
    def test_worker_reuses_process_and_bounds_output(self):
        program = 'import sys,json,os\nfor line in sys.stdin:\n print(json.dumps({"pid":os.getpid(),"echo":json.loads(line)}),flush=True)'
        with dbp_metrics.batch.JsonWorker([sys.executable, "-u", "-c", program]) as worker:
            first = worker.request({"clip": 1}, time.monotonic() + 2)
            second = worker.request({"clip": 2}, time.monotonic() + 2)
            self.assertEqual(first["pid"], second["pid"])
            self.assertEqual(second["echo"], {"clip": 2})
        self.assertIsNotNone(worker.process.poll())

    def test_worker_timeout_and_oversize_are_not_valid_predictions(self):
        programs = ('import sys,time\nsys.stdin.readline(); time.sleep(10)',
                    'import sys\nsys.stdin.readline(); print("x"*300000,flush=True)')
        for program in programs:
            with self.subTest(program=program):
                with dbp_metrics.batch.JsonWorker([sys.executable, "-u", "-c", program]) as worker:
                    with self.assertRaises(ValueError):
                        worker.request({}, time.monotonic() + 0.3)
                self.assertIsNotNone(worker.process.poll())

    def test_checkpoint_is_exclusive_and_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clip.json"
            payload = {"records": [], "fingerprint": "fixture"}
            dbp_metrics.batch.write_checkpoint(path, payload)
            self.assertEqual(dbp_metrics.batch.read_checkpoint(path), payload)
            with self.assertRaises(FileExistsError):
                dbp_metrics.batch.write_checkpoint(path, payload)
            wrapped = json.loads(path.read_text())
            wrapped["payload"]["fingerprint"] = "tampered"
            path.write_text(json.dumps(wrapped))
            with self.assertRaisesRegex(ValueError, "hash"):
                dbp_metrics.batch.read_checkpoint(path)

    def test_run_lock_prevents_concurrent_supervisors(self):
        with tempfile.TemporaryDirectory() as directory:
            with dbp_metrics.batch.run_lock(Path(directory)):
                with self.assertRaisesRegex(ValueError, "already running"):
                    with dbp_metrics.batch.run_lock(Path(directory)):
                        self.fail("a second supervisor must not run")
            with dbp_metrics.batch.run_lock(Path(directory)):
                pass

    def test_interruption_is_not_an_ordinary_clip_failure(self):
        self.assertFalse(issubclass(dbp_metrics.batch.BatchInterrupted, OSError))
        with self.assertRaises(dbp_metrics.batch.BatchInterrupted):
            dbp_metrics.batch.interrupted(15, None)

    def test_resume_retains_failure_exit_and_rejects_changed_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clip = {"clip_id": "fixture_part_00000000", "source_id": "fixture", "part": 0}
            arguments = SimpleNamespace(wall_seconds=10, timeout=2, model_revision="a" * 40,
                model_id="Qwen/Qwen3-VL-2B-Instruct", manifest=root / "manifest.json", offset=0,
                limit=1, ffprobe="ffprobe", ffmpeg="ffmpeg", frames=8, device="cuda:0")
            records = [record for record in dbp_metrics.batch.empty_records(clip["clip_id"], "/fixture.mp4", "rendition", "error", "fixture")
                       if record["field_id"] in dbp_metrics.batch.SEMANTIC_FIELDS]
            worker = Mock()
            worker.request.return_value = {"records": records, "failed": True}
            with patch("dbp_metrics.batch.load_manifest", return_value=([clip], "b" * 64)), \
                    patch("dbp_metrics._vlm.validate_local_model", return_value={"file_sha256": {"weights": "c" * 64}}) as verified, \
                    patch("dbp_metrics.batch.JsonWorker", return_value=worker) as factory:
                self.assertEqual(dbp_metrics.batch.run_locked_batch(arguments, root, root, root), 1)
                self.assertEqual(dbp_metrics.batch.run_locked_batch(arguments, root, root, root), 1)
                self.assertEqual(factory.call_count, 1)
                verified.return_value = {"file_sha256": {"weights": "d" * 64}}
                with self.assertRaisesRegex(ValueError, "resume identity"):
                    dbp_metrics.batch.run_locked_batch(arguments, root, root, root)
