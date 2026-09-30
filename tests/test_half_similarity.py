import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import dbp_metrics.experimental.half_similarity as subject


class FakeBackend:
    provenance = {"model_id": "test-only", "revision": "fake-v1"}

    def embed(self, frames):
        return [1.0, float(frames[0][0])]


def fake_decode(path, config):
    return {
        "duration_seconds": 2.0,
        "timestamp_origin_seconds": 0.0,
        "halves": [
            {"requested_seconds": [0.5], "timestamps_seconds": [0.5],
             "frame_indices": [0], "frames": [bytes([0, 0, 0])]},
            {"requested_seconds": [1.5], "timestamps_seconds": [1.5],
             "frame_indices": [1], "frames": [bytes([255, 255, 255])]},
        ],
    }


class HalfSimilarityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.media = Path(self.temp.name) / "clip.mp4"
        self.media.write_bytes(b"fake media")

    def test_sampling_is_deterministic_and_never_crosses_midpoint(self):
        first, second = subject.sample_halves([0, 0.2, 0.9, 1, 1.8], 2, 4)
        self.assertEqual(first["requested_seconds"], [0.125, 0.375, 0.625, 0.875])
        self.assertTrue(all(value < 1 for value in first["timestamps_seconds"]))
        self.assertTrue(all(value >= 1 for value in second["timestamps_seconds"]))
        self.assertEqual((first, second), subject.sample_halves([0, 0.2, 0.9, 1, 1.8], 2, 4))
        with self.assertRaises(ValueError):
            subject.sample_halves([0], 2, 4)

    def test_scores_are_separate_and_provenance_is_explicit(self):
        result = subject.measure(self.media, backend=FakeBackend(), decoder=fake_decode)
        self.assertEqual(result["visual_histogram_cosine"]["value"], 0)
        self.assertAlmostEqual(result["vjepa_cosine"]["value"], 1 / math.sqrt(65026))
        self.assertEqual(result["provenance"]["vjepa"]["revision"], "fake-v1")
        self.assertEqual(len(result["provenance"]["media_sha256"]), 64)
        self.assertNotIn("frames", result["sampling"]["halves"][0])
        self.assertNotIn("confidence", json.dumps(result))
        json.dumps(result, allow_nan=False)

    def test_backend_failure_does_not_destroy_baseline(self):
        backend = FakeBackend()
        backend.embed = lambda frames: [float("nan")]
        result = subject.measure(self.media, backend=backend, decoder=fake_decode)
        self.assertEqual(result["visual_histogram_cosine"]["status"], "ok")
        self.assertEqual(result["vjepa_cosine"]["status"], "unknown")
        self.assertIsNone(result["vjepa_cosine"]["value"])

    def test_decode_failure_and_missing_file_are_unknown(self):
        with patch.object(subject, "decode_media", side_effect=ValueError("bad video")):
            result = subject.measure(self.media)
        self.assertIsNone(result["visual_histogram_cosine"]["value"])
        self.assertEqual(result["visual_histogram_cosine"]["status"], "unknown")
        self.assertIsNone(subject.measure(self.media.with_name("missing.mp4"))["vjepa_cosine"]["value"])

    def test_baseline_does_not_load_model_and_hash_changes(self):
        with patch.object(subject.VJEPABackend, "embed", side_effect=AssertionError):
            first = subject.measure(self.media, decoder=fake_decode)
        self.assertEqual(first["vjepa_cosine"]["status"], "not_requested")
        self.media.write_bytes(b"changed")
        second = subject.measure(self.media, decoder=fake_decode)
        self.assertNotEqual(first["request_key"], second["request_key"])

    def test_invalid_cosine_and_config(self):
        for left, right in [([0], [1]), ([1], [1, 2]), ([float("inf")], [1])]:
            with self.assertRaises(ValueError):
                subject.cosine(left, right)
        for kwargs in ({"frames_per_half": 0}, {"bins": 0}, {"device": "auto"}):
            with self.assertRaises(ValueError):
                subject.Config(**kwargs)

    def test_adapter_contract_is_offline_pinned_encoder_only(self):
        torch = MagicMock()
        transformers = MagicMock()
        model = transformers.VJEPA2Model.from_pretrained.return_value.to.return_value.eval.return_value
        tokens = model.get_vision_features.return_value
        tokens.ndim = 3
        tokens.shape = (1, 8192, 1024)
        tokens.mean.return_value.__getitem__.return_value.float.return_value.cpu.return_value.tolist.return_value = [1, 2]
        backend = subject.VJEPABackend()
        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
                patch.object(subject.importlib.metadata, "version", return_value="4.57.5"):
            self.assertEqual(backend.embed([bytes(256 * 256 * 3)] * 64), [1, 2])
        options = transformers.VJEPA2Model.from_pretrained.call_args.kwargs
        self.assertTrue(options["local_files_only"])
        self.assertTrue(options["use_safetensors"])
        self.assertFalse(options["trust_remote_code"])
        self.assertEqual(options["revision"], subject.MODEL_REVISION)
        tokens.mean.assert_called_once_with(dim=1)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_real_cpu_mp4_and_cli_resume(self):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=red:s=32x32:r=4:d=2", "-c:v", "mpeg4", str(self.media)], check=True)
        result = subject.measure(self.media, subject.Config(frames_per_half=4))
        self.assertEqual(result["visual_histogram_cosine"]["status"], "ok", result)
        self.assertAlmostEqual(result["visual_histogram_cosine"]["value"], 1)
        manifest = self.media.with_suffix(".jsonl")
        output = self.media.with_suffix(".results.jsonl")
        manifest.write_text(json.dumps({"id": "clip", "path": self.media.name}) + "\n")
        command = [sys.executable, "-m", "dbp_metrics.cli.half_similarity",
                   "--manifest", str(manifest), "--output", str(output), "--frames-per-half", "4"]
        subprocess.run(command, check=True, capture_output=True)
        initial = output.read_bytes()
        subprocess.run(command + ["--resume"], check=True, capture_output=True)
        self.assertEqual(output.read_bytes(), initial)
        with output.open("ab") as destination:
            destination.write(b'{"partial":')
        subprocess.run(command + ["--resume"], check=True, capture_output=True)
        self.assertEqual(output.read_bytes(), initial)
        subprocess.run(command + ["--resume", "--bins", "8"], check=True, capture_output=True)
        self.assertEqual(len(output.read_text().splitlines()), 2)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=blue:s=32x32:r=4:d=2", "-c:v", "mpeg4", str(self.media)], check=True)
        subprocess.run(command + ["--resume"], check=True, capture_output=True)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertNotEqual(records[0]["request_key"], records[-1]["request_key"])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_real_different_halves_and_offline_model_failure(self):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=red:s=32x32:r=4:d=1", "-f", "lavfi", "-i",
                        "color=c=blue:s=32x32:r=4:d=1", "-filter_complex",
                        "[0:v][1:v]concat=n=2:v=1:a=0", "-c:v", "mpeg4", str(self.media)], check=True)
        with patch.object(subject.VJEPABackend, "embed", side_effect=RuntimeError("cache unavailable")):
            result = subject.measure(self.media, subject.Config(vjepa=True))
        self.assertEqual(result["visual_histogram_cosine"]["status"], "ok", result)
        self.assertLess(result["visual_histogram_cosine"]["value"], 0.5)
        self.assertIsNone(result["vjepa_cosine"]["value"])
        for half, bounds in zip(result["sampling"]["halves"], ((0, 1), (1, 2))):
            self.assertEqual(len(half["timestamps_seconds"]), 64)
            self.assertTrue(all(bounds[0] <= value < bounds[1] for value in half["timestamps_seconds"]))

    def test_unknown_manifest_entries_retry_and_outputs_cannot_clobber_inputs(self):
        manifest = self.media.with_suffix(".jsonl")
        output = self.media.with_suffix(".results.jsonl")
        manifest.write_text(json.dumps({"id": "bad", "path": self.media.name}) + "\n")
        command = [sys.executable, "-m", "dbp_metrics.cli.half_similarity",
                   "--manifest", str(manifest), "--output", str(output)]
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 1)
        self.assertEqual(subprocess.run(command + ["--resume"], capture_output=True).returncode, 1)
        self.assertEqual(len(output.read_text().splitlines()), 2)
        original = self.media.read_bytes()
        self.assertEqual(subprocess.run(command + ["--output", str(self.media), "--resume"],
                                        capture_output=True).returncode, 2)
        self.assertEqual(self.media.read_bytes(), original)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_scene_window_contract_and_uncertainty(self):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=red:s=32x32:r=8:d=4", "-c:v", "mpeg4", str(self.media)], check=True)
        config = subject.Config(frames_per_half=4)
        result = subject.measure_scene_window(self.media, 2, window_seconds=0.5, config=config,
                                              backend=FakeBackend())
        self.assertEqual(result["visual_histogram_cosine"]["status"], "ok", result)
        self.assertEqual(result["assessment"]["uncertainty"], "uncalibrated")
        self.assertIsNone(result["assessment"]["confirmed_cut"])
        self.assertEqual(result["provenance"]["comparison"]["candidate_seconds"], 2)
        for half, bounds in zip(result["sampling"]["halves"], ((1.5, 2), (2, 2.5))):
            self.assertTrue(all(bounds[0] <= value < bounds[1] for value in half["timestamps_seconds"]))
        other = subject.measure_scene_window(self.media, 2.5, window_seconds=0.5, config=config)
        self.assertNotEqual(result["request_key"], other["request_key"])
        self.assertEqual(other["assessment"]["uncertainty"], "embedding_unavailable")
        edge = subject.measure_scene_window(self.media, 0.1, window_seconds=0.5, config=config)
        self.assertIsNone(edge["visual_histogram_cosine"]["value"])
        self.assertEqual(edge["assessment"]["uncertainty"], "embedding_unavailable")

    @unittest.skipUnless(os.environ.get("DBP_TEST_SIMILARITY_MP4"), "Set DBP_TEST_SIMILARITY_MP4 for bundled media")
    def test_existing_bundled_mp4(self):
        result = subject.measure(os.environ["DBP_TEST_SIMILARITY_MP4"])
        self.assertEqual(result["visual_histogram_cosine"]["status"], "ok", result)
        self.assertTrue(math.isfinite(result["visual_histogram_cosine"]["value"]))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_default_and_maximum_selection_preserve_exact_frame_timestamps(self):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                        "color=c=red:s=32x32:r=60:d=19", "-c:v", "mpeg4", str(self.media)], check=True)
        timeline = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(self.media)]))
        timestamps = [float(frame["best_effort_timestamp_time"]) for frame in timeline["frames"]]
        for config in (subject.Config(), subject.Config(frames_per_half=128)):
            with self.subTest(frames=config.frames_per_half):
                decoded = subject.decode_media(self.media, config)
                expected = subject.sample_halves(timestamps, 19, config.frames_per_half)
                for actual, planned in zip(decoded["halves"], expected):
                    self.assertEqual(actual["frame_indices"], planned["frame_indices"])
                    self.assertEqual(actual["timestamps_seconds"], planned["timestamps_seconds"])
                    self.assertEqual(actual["unique_frame_count"], config.frames_per_half)
                    self.assertEqual(len(actual["frames"]), config.frames_per_half)


if __name__ == "__main__":
    unittest.main()
