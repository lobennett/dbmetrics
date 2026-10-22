import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dbmetrics.evidence import ConflictError, EvidenceVideo


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.media = self.root / "clip.mp4"
        self.media.write_bytes(b"fixture media")
        self.report = self.root / "report.jsonl"
        self.events = [
            {"type": "start", "clip_id": "clip"},
            {"type": "provenance", "media": {
                "sha256": hashlib.sha256(self.media.read_bytes()).hexdigest()},
             "checkpoint": {"model_name": "fixture"}},
            {"type": "frame", "index": 0, "relative_seconds": 0,
             "width": 10, "height": 10, "inference_status": "measured",
             "detections": []},
            {"type": "summary", "clip_id": "clip", "processing_status": "complete",
             "coverage": "full_decoded_stream", "decode_eof": True,
             "decoded_frames": 1, "inferred_frames": 1, "positive_frames": 0},
        ]
        self.write_report()

    def write_report(self):
        self.report.write_text("\n".join(json.dumps(row) for row in self.events))

    def test_reads_hash_bound_evidence_without_inventing_duration(self):
        video = EvidenceVideo(self.media, self.report)
        self.assertEqual(video.clip_id, "clip")
        self.assertIsNone(video.get("clip")["duration_seconds"])
        self.assertEqual(video.get("clip")["evidence"]["inferred_frames"], 1)
        with video.open_media("clip") as stream:
            self.assertEqual(stream.read(), b"fixture media")

    def test_rejects_hash_mismatch(self):
        self.media.write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            EvidenceVideo(self.media, self.report)

    def test_rejects_incomplete_report(self):
        self.events[-1]["decode_eof"] = False
        self.write_report()
        with self.assertRaisesRegex(ValueError, "complete"):
            EvidenceVideo(self.media, self.report)

    def test_rejects_mutated_media_after_loading(self):
        video = EvidenceVideo(self.media, self.report)
        self.media.write_bytes(b"different")
        with self.assertRaises(ConflictError):
            video.open_media("clip")

    def test_rejects_symlink_media(self):
        link = self.root / "link.mp4"
        link.symlink_to(self.media)
        with self.assertRaisesRegex(ValueError, "Symbolic"):
            EvidenceVideo(link, self.report)
