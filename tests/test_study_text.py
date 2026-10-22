"""Real lossless video fixtures test transport and evidence, not OCR accuracy."""

import importlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
FFMPEG = os.environ.get("DBP_TEST_FFMPEG") or shutil.which("ffmpeg") or str(
    Path.home() / "Library/Caches/Cypress/15.20.1/Cypress.app/Contents/Resources/app/"
    "node_modules/@ffmpeg-installer/darwin-arm64/ffmpeg")


class FixtureDetector:
    def detect(self, frame):
        if not any(frame.rgb):
            return []
        return [{"polygon": [[1, 1], [30, 1], [30, 14], [1, 14]], "score": None}]


class StudyTextTests(unittest.TestCase):
    def setUp(self):
        try:
            self.subject = importlib.import_module("dbmetrics._ocr")
        except ModuleNotFoundError:
            self.fail("dbmetrics._ocr benchmark implementation is missing")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def video(self, name="clip.mov", text=True, offset=False, variable_pts=False):
        if not Path(FFMPEG).is_file():
            self.skipTest("set DBP_TEST_FFMPEG to an existing FFmpeg binary")
        black = bytes(32 * 16 * 3)
        glyph = bytearray(black)
        for row, pattern in enumerate(("11111", "00100", "00100", "00100", "00100")):
            for column, pixel in enumerate(pattern):
                if pixel == "1":
                    start = ((row + 3) * 32 + column + 3) * 3
                    glyph[start:start + 3] = b"\xff\xff\xff"
        path = self.directory / name
        command = [FFMPEG, "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "32x16", "-r", "4", "-i", "pipe:0"]
        if offset:
            command += ["-vf", "setpts=PTS+2/TB"]
        elif variable_pts:
            command += ["-vf", "setpts=N*(N+1)/2"]
        command += ["-vsync", "0", "-c:v", "png", "-threads", "1", str(path)]
        subprocess.run(command, input=black * 3 + (bytes(glyph) if text else black),
                       capture_output=True, check=True, timeout=10)
        return path

    def measure(self, path=None, detector=None, **limits):
        events = []
        result = self.subject.measure_clip(
            path or self.video(), self.subject.Limits(**limits),
            detector=detector or FixtureDetector(), emit=events.append, ffmpeg=FFMPEG)
        return result, events

    def test_dense_finds_single_late_frame_and_proves_decode_eof(self):
        result, events = self.measure(max_frames=4)
        self.assertEqual(result["processing_status"], "complete")
        self.assertEqual(result["coverage"], "full_decoded_stream")
        self.assertEqual(result["evidence_result"], "detected")
        self.assertTrue(result["decode_eof"])
        self.assertEqual(result["decoded_frames"], 4)
        self.assertEqual(result["inferred_frames"], 4)
        self.assertEqual(result["inferred_fraction_of_stream"], 1)
        self.assertEqual([item["timestamp_seconds"] for item in events], [0, 0.25, 0.5, 0.75])
        self.assertEqual([len(item["detections"]) for item in events], [0, 0, 0, 1])
        self.assertIsNone(events[-1]["detections"][0]["score"])
        self.assertIsNone(result["annotation"]["logo_present"])
        self.assertIsNone(result["annotation"]["text_present"])
        json.dumps(result, allow_nan=False)

    def test_sampled_miss_is_not_absence_or_full_coverage(self):
        result, events = self.measure(every_nth=2)
        self.assertEqual(result["coverage"], "sampled")
        self.assertEqual(result["inferred_frames"], 2)
        self.assertEqual(result["inferred_fraction_of_stream"], 0.5)
        self.assertEqual(result["evidence_result"], "no_detections_in_processed_frames")
        self.assertIsNone(result["annotation"]["text_present"])
        self.assertEqual(events[-1]["inference_status"], "skipped")
        self.assertEqual(result["max_inferred_gap_seconds"], 0.5)

    def test_budget_and_duration_stops_remain_incomplete(self):
        path = self.video()
        for limits, count, reason in (({"max_frames": 3}, 3, "frame_budget"),
                                      ({"max_seconds": 0.5}, 2, "duration_budget")):
            with self.subTest(limits=limits):
                result, events = self.measure(path, **limits)
                self.assertEqual(result["processing_status"], "incomplete")
                self.assertEqual(result["coverage"], "dense_prefix")
                self.assertEqual(result["stop_reason"], reason)
                self.assertEqual(result["decoded_frames"], count)
                self.assertFalse(result["decode_eof"])
                self.assertIsNone(result["inferred_fraction_of_stream"])
                self.assertIsNone(result["annotation"]["text_present"])

    def test_clean_negative_is_no_detections_not_adjudicated_absence(self):
        result, events = self.measure(self.video(text=False))
        self.assertEqual(result["evidence_result"], "no_detections")
        self.assertEqual(result["coverage"], "full_decoded_stream")
        self.assertIsNone(result["annotation"]["text_present"])

    def test_original_pts_and_relative_timestamps_are_both_preserved(self):
        result, events = self.measure(self.video(offset=True))
        self.assertEqual([event["timestamp_seconds"] for event in events], [2, 2.25, 2.5, 2.75])
        self.assertEqual([event["relative_seconds"] for event in events], [0, 0.25, 0.5, 0.75])
        self.assertEqual(result["observed_span_seconds"], 0.75)

    def test_variable_frame_intervals_are_measured_not_derived_from_nominal_fps(self):
        result, events = self.measure(self.video(variable_pts=True))
        self.assertEqual([event["relative_seconds"] for event in events], [0, 0.25, 0.75, 1.5])
        self.assertEqual(result["max_inferred_gap_seconds"], 0.75)
        self.assertEqual(result["coverage"], "full_decoded_stream")

    def test_earlier_positive_survives_later_inference_error_without_full_coverage(self):
        detector = FixtureDetector()

        def detect(frame):
            if frame.index > 0:
                raise RuntimeError("fixture detector failure after positive")
            return [{"polygon": [[1, 1], [30, 1], [30, 14], [1, 14]], "score": 0.7}]

        detector.detect = detect
        result, events = self.measure(detector=detector)
        self.assertEqual(result["processing_status"], "error")
        self.assertEqual(result["evidence_result"], "detected")
        self.assertEqual(result["positive_frames"], 1)
        self.assertEqual(result["coverage"], "dense_prefix")
        self.assertIsNone(result["inferred_fraction_of_stream"])

    def test_decode_only_is_not_text_measurement(self):
        events = []
        result = self.subject.measure_clip(self.video(), self.subject.Limits(),
                                           detector=None, emit=events.append, ffmpeg=FFMPEG)
        self.assertEqual(result["evidence_result"], "not_measured")
        self.assertEqual(result["coverage"], "not_measured")
        self.assertEqual(result["inferred_frames"], 0)
        self.assertTrue(result["decode_eof"])

    def test_corrupt_media_and_detector_failure_do_not_become_negatives(self):
        corrupt = self.directory / "corrupt.mov"
        corrupt.write_bytes(b"not a movie")
        result, events = self.measure(corrupt)
        self.assertEqual(result["processing_status"], "error")
        self.assertIsNone(result["annotation"]["text_present"])
        detector = FixtureDetector()
        detector.detect = lambda frame: (_ for _ in ()).throw(RuntimeError("fixture failure"))
        result, events = self.measure(detector=detector)
        self.assertEqual(result["processing_status"], "error")
        self.assertEqual(result["inferred_frames"], 0)
        self.assertEqual(result["evidence_result"], "not_measured")
        self.assertEqual(result["decoded_frames"], 1)
        self.assertEqual(events[0]["inference_status"], "failed")

    def test_invalid_limits_rejected_before_launch(self):
        for options in ({"max_frames": 0}, {"max_frames": True}, {"max_frames": 1801},
                        {"max_seconds": float("nan")}, {"max_seconds": 61},
                        {"timeout": float("inf")}, {"threads": 5}, {"every_nth": 0},
                        {"max_pixels": 9_000_000}, {"detector_side": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.subject.Limits(**options)

    def test_large_frame_rejected_without_inference(self):
        result, events = self.measure(max_pixels=64)
        self.assertEqual(result["processing_status"], "error")
        self.assertEqual(result["inferred_frames"], 0)

    def test_detection_schema_never_fabricates_or_accepts_invalid_scores(self):
        frame = self.subject.Frame(0, 0, "1/4", 0, 0, 32, 16, bytes(1536))
        valid = {"dt_polys": [[[1, 1], [30, 1], [30, 14], [1, 14]]], "dt_scores": [0.73]}
        self.assertEqual(self.subject.paddle_detections(valid, frame)[0]["score"], 0.73)
        self.assertIsNone(self.subject.paddle_detections({"dt_polys": valid["dt_polys"]}, frame)[0]["score"])
        for value in ({}, {"dt_polys": valid["dt_polys"], "dt_scores": []},
                      {**valid, "dt_scores": [float("nan")]}, {**valid, "dt_scores": [1.1]},
                      {"dt_polys": [[[100, 0]] * 4]}, {"dt_polys": [[[0, 0]] * 4]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.subject.paddle_detections(value, frame)

    def test_checkpoint_files_have_explicit_hashes_and_symlinks_are_rejected(self):
        model = self.directory / "model"
        model.mkdir()
        for name in ("inference.json", "inference.pdiparams", "inference.yml"):
            (model / name).write_bytes(b"fixture only, not a real model")
        provenance = self.subject.checkpoint_provenance(model, "fixture-source", "fixture-revision")
        self.assertEqual(len(provenance["files"]), 3)
        self.assertTrue(all(len(record["sha256"]) == 64 for record in provenance["files"]))
        (model / "inference.yml").unlink()
        (model / "inference.yml").symlink_to(model / "inference.json")
        with self.assertRaises(ValueError):
            self.subject.checkpoint_provenance(model, "fixture-source", "fixture-revision")

    def test_adapter_refuses_missing_local_checkpoint_before_optional_import(self):
        with self.assertRaises(ValueError):
            self.subject.PaddleDetector({"model_dir": None})

    def test_adapter_uses_bgr_full_frame_and_limits_opencv_threads(self):
        try:
            import numpy
        except ImportError:
            self.skipTest("adapter array test requires already-installed NumPy")
        model = self.directory / "model"
        model.mkdir()
        for name in ("inference.json", "inference.pdiparams", "inference.yml"):
            (model / name).write_text("synthetic, not a real checkpoint")
        thread_counts = []
        observed = {}

        class TextDetectionFixture:
            def __init__(self, **options):
                observed.update(options)

            def predict(self, array, batch_size):
                if (thread_counts != [1] or array.shape != (2, 2, 3)
                        or array[0, 0].tolist() != [30, 20, 10] or batch_size != 1):
                    raise ValueError("incorrect full-frame BGR input or thread controls")
                yield {"dt_polys": numpy.array([[[0, 0], [1, 0], [1, 1], [0, 1]]]),
                       "dt_scores": [0.71], "input_path": None, "page_index": None}

        config = {"media": str(model / "inference.json"), "clip_id": "fixture", "detector": "paddle",
                  "device": "cpu", "ffmpeg": FFMPEG, "limits": self.subject.asdict(self.subject.Limits()),
                  "model_dir": str(model), "model_source": "fixture", "model_revision": "fixture"}
        modules = {"paddleocr": SimpleNamespace(TextDetection=TextDetectionFixture),
                   "cv2": SimpleNamespace(setNumThreads=thread_counts.append)}
        with patch.dict(sys.modules, modules):
            detector = self.subject.PaddleDetector(config)
            frame = self.subject.Frame(0, 0, "1/4", 0, 0, 2, 2, bytes([10, 20, 30]) * 4)
            detections = detector.detect(frame)
        self.assertEqual(detections[0]["score"], 0.71)
        self.assertEqual(observed["model_dir"], str(model))
        self.assertFalse(observed["enable_hpi"])
        self.assertEqual(observed["device"], "cpu")

    def cli(self, *arguments, cwd=None):
        return subprocess.run([sys.executable, "-B", "-m", "dbmetrics.cli.ocr",
                               *map(str, arguments)], capture_output=True, text=True, timeout=15, cwd=cwd)

    def test_worker_overrides_inherited_dependency_caches_before_subprocess_imports(self):
        media = self.video()
        shared = self.directory / "shared-cache-fixture"
        shared.mkdir()
        (shared / "untouched").write_text("existing cache fixture")
        variables = ("HOME", "XDG_CACHE_HOME", "PADDLE_PDX_CACHE_HOME", "PADDLE_HOME",
                     "PADDLE_EXTENSION_DIR", "TORCH_HOME", "TORCH_EXTENSIONS_DIR", "HF_HOME",
                     "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_ASSETS_CACHE", "HUGGINGFACE_ASSETS_CACHE",
                     "HF_XET_CACHE",
                     "TRANSFORMERS_CACHE", "PYTORCH_TRANSFORMERS_CACHE", "PYTORCH_PRETRAINED_BERT_CACHE",
                     "MODELSCOPE_CACHE", "MPLCONFIGDIR", "CUDA_CACHE_PATH", "TRITON_CACHE_DIR",
                     "NUMBA_CACHE_DIR", "OPENCV_OPENCL_CACHE_DIR", "TMPDIR", "TMP", "TEMP")
        capture = self.directory / "cache-paths.json"
        decoder = self.directory / "cache-probe-ffmpeg"
        decoder.write_text(
            f"#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\n"
            "if '-version' in sys.argv:\n"
            f"    locations = {{name: os.environ[name] for name in {variables!r}}}\n"
            f"    Path({str(capture)!r}).write_text(json.dumps(locations))\n"
            "    for location in locations.values():\n"
            "        Path(location).mkdir(parents=True, exist_ok=True)\n"
            "        (Path(location) / 'cache-write-fixture').write_text('fixture')\n"
            f"os.execv({FFMPEG!r}, [{FFMPEG!r}, *sys.argv[1:]])\n")
        decoder.chmod(0o700)
        inherited = {name: str(shared / name) for name in variables}
        with patch.dict(os.environ, inherited):
            result = self.cli("--media", media, "--clip-id", "private-cache", "--detector", "decode-only",
                              "--ffmpeg", decoder, "--output-dir", self.directory / "reports")
            self.assertEqual(os.environ["PADDLE_PDX_CACHE_HOME"], inherited["PADDLE_PDX_CACHE_HOME"])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sorted(path.name for path in shared.iterdir()), ["untouched"])
        locations = json.loads(capture.read_text())
        scratch = Path(locations["HOME"])
        self.assertEqual(scratch.parent, self.directory / "reports")
        for name, location in locations.items():
            with self.subTest(variable=name):
                self.assertTrue(Path(location) == scratch or scratch in Path(location).parents)
        self.assertFalse(scratch.exists())

    def test_dns_resolvers_are_blocked_before_secondary_safety_audit_hook(self):
        source = (
            "import socket, sys\n"
            f"sys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "from dbmetrics._ocr import offline_audit\n"
            "sys.addaudithook(offline_audit)\n"
            "def safety_hook(event, arguments):\n"
            "    if event.startswith('socket.') and event != 'socket.__new__':\n"
            "        raise AssertionError('resolver escaped offline hook: ' + event)\n"
            "sys.addaudithook(safety_hook)\n"
            "calls = [lambda: socket.getaddrinfo('fixture.invalid', 80),\n"
            "         lambda: socket.gethostbyname('fixture.invalid'),\n"
            "         lambda: socket.gethostbyname_ex('fixture.invalid'),\n"
            "         lambda: socket.gethostbyaddr('192.0.2.1'),\n"
            "         lambda: socket.getnameinfo(('192.0.2.1', 80), 0)]\n"
            "for call in calls:\n"
            "    try:\n"
            "        call()\n"
            "    except RuntimeError as error:\n"
            "        assert 'offline benchmark refuses' in str(error)\n"
            "    else:\n"
            "        raise AssertionError('resolver was not blocked')\n")
        result = subprocess.run([sys.executable, "-I", "-S", "-B", "-c", source],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_cli_relative_ffmpeg_remains_runnable_after_worker_changes_directory(self):
        media = self.video()
        (self.directory / "ffmpeg").symlink_to(FFMPEG)
        arguments = ["--media", media.name, "--clip-id", "relative-cli", "--detector", "decode-only",
                     "--ffmpeg", "./ffmpeg", "--output-dir", self.directory / "reports"]
        dry_run = self.cli(*arguments, "--dry-run", cwd=self.directory)
        self.assertEqual(dry_run.returncode, 0, dry_run.stderr)
        self.assertTrue(json.loads(dry_run.stdout)["ffmpeg_available"])
        result = self.cli(*arguments, cwd=self.directory)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["decoded_frames"], 4)
        self.assertEqual(json.loads(dry_run.stdout)["config"]["ffmpeg"], str(Path(FFMPEG).resolve()))

    def test_public_runner_normalizes_relative_media_model_and_ffmpeg_without_mutating_config(self):
        media = self.video()
        decoder = self.directory / "ffmpeg"
        decoder.symlink_to(FFMPEG)
        model = self.directory / "model"
        model.mkdir()
        for name in ("inference.json", "inference.pdiparams", "inference.yml"):
            (model / name).write_text("fixture only; decode-only never loads it")
        config = {"media": os.path.relpath(media), "clip_id": "relative-public", "detector": "decode-only",
                  "device": "cpu", "ffmpeg": os.path.relpath(decoder),
                  "limits": self.subject.asdict(self.subject.Limits()), "model_dir": os.path.relpath(model),
                  "model_source": "fixture", "model_revision": "fixture"}
        original = dict(config)
        result = self.subject.run_benchmark(config, self.directory / "reports")
        self.assertEqual(result["processing_status"], "complete", result["diagnostic"])
        self.assertEqual(result["decoded_frames"], 4)
        self.assertEqual(config, original)
        start = json.loads(Path(result["report_path"]).read_text().splitlines()[0])
        self.assertEqual(start["config"]["ffmpeg"], str(Path(FFMPEG).resolve()))
        self.assertEqual(start["config"]["media"], str(media))
        self.assertEqual(start["config"]["model_dir"], str(model))

    def test_cli_dry_run_works_without_packages_or_checkpoint_and_writes_nothing(self):
        media = self.directory / "not-decoded.mov"
        media.write_bytes(b"validation only")
        output = self.directory / "reports"
        result = self.cli("--media", media, "--clip-id", "clip-1", "--dry-run", "--output-dir", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "dry_run")
        self.assertFalse(report["inference_executed"])
        self.assertFalse(report["ready_for_inference"])
        self.assertFalse(output.exists())

    def test_cli_reports_are_private_nonoverwriting_and_decode_only_is_explicit(self):
        media = self.video()
        output = self.directory / "reports"
        arguments = ["--media", media, "--clip-id", "clip-1", "--detector", "decode-only",
                     "--ffmpeg", FFMPEG, "--output-dir", output, "--max-frames", "4"]
        first = self.cli(*arguments)
        self.assertEqual(first.returncode, 0, first.stderr)
        first_path = Path(json.loads(first.stdout)["report_path"])
        original = first_path.read_bytes()
        second = self.cli(*arguments)
        self.assertEqual(second.returncode, 0, second.stderr)
        second_path = Path(json.loads(second.stdout)["report_path"])
        self.assertNotEqual(first_path, second_path)
        self.assertEqual(first_path.read_bytes(), original)
        self.assertEqual(stat.S_IMODE(first_path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        events = [json.loads(line) for line in original.splitlines()]
        self.assertEqual(events[0]["type"], "start")
        self.assertEqual(events[-1]["type"], "summary")
        self.assertEqual(events[-1]["evidence_result"], "not_measured")
        self.assertEqual(events[-1]["decoded_frames"], 4)
        self.assertEqual(len(next(event for event in events if event["type"] == "provenance")
                             ["media"]["sha256"]), 64)
        provenance = next(event for event in events if event["type"] == "provenance")
        self.assertTrue(provenance["ffmpeg_version"].startswith("ffmpeg version"))
        self.assertEqual(len(provenance["ffmpeg_sha256"]), 64)

    def test_cli_deadline_kills_blocked_decoder_and_keeps_private_failure_report(self):
        decoder = self.directory / "blocked-ffmpeg"
        decoder.write_text(f"#!{sys.executable}\nimport sys, time\n"
                           "if '-version' in sys.argv:\n    print('fixture ffmpeg version')\n"
                           "else:\n    time.sleep(30)\n")
        decoder.chmod(0o700)
        media = self.directory / "fixture.mov"
        media.write_bytes(b"not decoded by the blocking process fixture")
        started = time.monotonic()
        result = self.cli("--media", media, "--clip-id", "clip-timeout", "--detector", "decode-only",
                          "--ffmpeg", decoder, "--output-dir", self.directory / "reports", "--timeout", "0.3")
        self.assertEqual(result.returncode, 3, result.stderr)
        self.assertLess(time.monotonic() - started, 3)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["stop_reason"], "deadline")
        self.assertEqual(summary["processing_status"], "incomplete")
        self.assertFalse(summary["decode_eof"])
        self.assertEqual(summary["evidence_result"], "not_measured")
        records = [json.loads(line) for line in Path(summary["report_path"]).read_text().splitlines()]
        self.assertEqual(records[-1]["type"], "summary")

    def test_worker_start_failure_still_finishes_error_journal(self):
        media = self.directory / "fixture.mov"
        media.write_bytes(b"fixture")
        config = {"media": str(media), "clip_id": "spawn-failure", "detector": "decode-only",
                  "device": "cpu", "ffmpeg": sys.executable, "limits": self.subject.asdict(self.subject.Limits()),
                  "model_dir": None, "model_source": None, "model_revision": None}
        with patch.object(self.subject.subprocess, "Popen", side_effect=OSError("fixture spawn failure")):
            result = self.subject.run_benchmark(config, self.directory / "reports")
        self.assertEqual(result["processing_status"], "error")
        records = [json.loads(line) for line in Path(result["report_path"]).read_text().splitlines()]
        self.assertEqual(records[-1]["type"], "summary")

    def test_worker_preserves_explicit_local_font_outside_scratch(self):
        font = self.directory / "trusted-font-fixture.ttf"
        font.write_bytes(b"trusted regular file fixture, not an actual font")
        original = font.read_bytes()
        variable = "PADDLE_PDX_LOCAL_FONT_FILE_PATH"
        config = {"media": str(font), "clip_id": "font-environment-fixture", "detector": "decode-only",
                  "device": "cpu", "ffmpeg": sys.executable, "limits": self.subject.asdict(self.subject.Limits()),
                  "model_dir": None, "model_source": None, "model_revision": None}
        with patch.dict(os.environ, {variable: str(font)}):
            with patch.object(self.subject.subprocess, "Popen", side_effect=OSError("fixture prevents launch")) as launch:
                result = self.subject.run_benchmark(config, self.directory / "reports")
            self.assertEqual(os.environ[variable], str(font))
        environment = launch.call_args.kwargs["env"]
        self.assertEqual(environment.get(variable), str(font))
        self.assertNotEqual(environment[variable], environment["HOME"])
        self.assertNotIn(Path(environment["HOME"]), font.parents)
        self.assertEqual(font.read_bytes(), original)
        self.assertEqual(result["processing_status"], "error")
        self.assertEqual(result["inferred_frames"], 0)

    def test_offline_worker_blocks_network_before_optional_adapter_import(self):
        model = self.directory / "model"
        model.mkdir()
        for name in ("inference.json", "inference.pdiparams", "inference.yml"):
            (model / name).write_text("synthetic model directory, never real weights")
        (self.directory / "paddleocr.py").write_text(
            "import socket\nsocket.create_connection(('127.0.0.1', 9))\n")
        with patch.dict(os.environ, {"PYTHONPATH": str(self.directory)}):
            result = self.cli("--media", self.video(), "--clip-id", "offline-import",
                              "--model-dir", model, "--model-source", "fixture", "--model-revision", "fixture",
                              "--ffmpeg", FFMPEG, "--output-dir", self.directory / "reports")
        self.assertEqual(result.returncode, 3, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["processing_status"], "error")
        self.assertIn("offline benchmark refuses", summary["diagnostic"])
        self.assertIsNone(summary["annotation"]["text_present"])

    def test_cli_validation_refuses_limits_missing_checkpoint_and_public_output(self):
        media = self.directory / "clip.mov"
        media.write_bytes(b"fixture")
        output = self.directory / "public"
        output.mkdir(mode=0o755)
        for extra in (["--dry-run", "--max-frames", "0"],
                      ["--output-dir", self.directory / "unused"],
                      ["--detector", "decode-only", "--output-dir", output]):
            with self.subTest(extra=extra):
                result = self.cli("--media", media, "--clip-id", "clip-1", *extra)
                self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.directory / "unused").exists())

    def test_config_payload_is_bounded_before_worker_pipe_write(self):
        media = self.directory / "fixture.mov"
        media.write_bytes(b"fixture")
        config = {"media": str(media), "clip_id": "clip-1", "detector": "decode-only", "device": "cpu",
                  "ffmpeg": FFMPEG, "limits": self.subject.asdict(self.subject.Limits()),
                  "model_source": "x" * 65536}
        with self.assertRaises(ValueError):
            self.subject.validate_config(config)

    def test_missing_optional_package_is_an_error_journal_not_a_negative(self):
        model = self.directory / "model"
        model.mkdir()
        for name in ("inference.json", "inference.pdiparams", "inference.yml"):
            (model / name).write_text("synthetic fixture, not loadable weights")
        (self.directory / "paddleocr.py").write_text("raise ImportError('fixture Paddle unavailable')\n")
        with patch.dict(os.environ, {"PYTHONPATH": str(self.directory)}):
            result = self.cli("--media", self.video(), "--clip-id", "missing-package",
                              "--model-dir", model, "--model-source", "fixture", "--model-revision", "fixture",
                              "--ffmpeg", FFMPEG, "--output-dir", self.directory / "reports")
        self.assertEqual(result.returncode, 3, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["processing_status"], "error")
        self.assertEqual(summary["evidence_result"], "not_measured")
        self.assertIn("ImportError", summary["diagnostic"])

    def test_import_is_standard_library_only(self):
        command = [sys.executable, "-I", "-S", "-B", "-c",
                   f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r}); import dbmetrics._ocr; "
                   "assert not {'paddle', 'paddleocr', 'numpy'} & sys.modules.keys()"]
        subprocess.run(command, check=True, capture_output=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
