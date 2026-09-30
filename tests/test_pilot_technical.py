"""Bounded direct measurements on synthetic RGB/PTS, not scientific validation."""

import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

from test_pilot_visual import FFMPEG, showinfo


FIELDS = {"rgb_chroma_mean", "near_gray_pixel_fraction", "frame_interval_cv",
          "optical_flow_mean_px_per_pair"}
FLOW = "optical_flow_mean_px_per_pair"


def synthetic_backend(magnitude=1.0):
    import numpy

    backend = Mock(__version__="synthetic-not-real-opencv", COLOR_RGB2GRAY=7)
    backend.cvtColor.side_effect = lambda frame, code: frame[:, :, 0]
    backend.calcOpticalFlowFarneback.side_effect = lambda previous, current, *args, **kwargs: (
        numpy.full((*previous.shape, 2), magnitude / math.sqrt(2), dtype=numpy.float32))
    return numpy, backend


class TechnicalTests(unittest.TestCase):
    def setUp(self):
        self.subject = importlib.import_module("dbp_metrics._technical")

    def measure(self, raw, pts=None, diagnostics=None, backend=None, **options):
        width = options.setdefault("width", 2)
        if pts is None:
            pts = [index * 40 for index in range(len(raw) // (width * width * 3))]
        self.runner = Mock(return_value=(raw, showinfo(pts, width=width) if diagnostics is None else diagnostics))
        with patch.object(self.subject, "_load_flow_backend", return_value=backend or synthetic_backend()):
            return self.subject.measure_technical("/dev/fd/17", run_process=self.runner, **options)

    def evidence(self, records, field="rgb_chroma_mean"):
        return records[field]["evidence"][0]

    def assert_unknown(self, records, status):
        self.assertEqual(set(records), FIELDS)
        for record in records.values():
            self.assertIsNone(record["value"])
            self.assertEqual(record["processing_status"], status)

    def test_exact_chroma_and_threshold_are_pixel_weighted_not_a_color_class(self):
        raw = bytes((0, 0, 0, 128, 128, 130, 128, 128, 131, 255, 0, 0))
        records = self.measure(raw * 3)
        self.assertEqual(records["rgb_chroma_mean"]["value"], 260 / (4 * 255))
        self.assertEqual(records["near_gray_pixel_fraction"]["value"], 0.5)
        self.assertEqual(records["frame_interval_cv"]["value"], 0.0)
        self.assertAlmostEqual(records[FLOW]["value"], 1, places=6)
        evidence = self.evidence(records)
        self.assertEqual(evidence["sampling"]["near_gray_threshold_rgb_levels"], 2)
        self.assertIn("not", evidence["sampling"]["color_basis"])
        self.assertEqual(set(records), FIELDS)
        json.dumps(records, allow_nan=False)

    def test_black_gray_white_and_primary_colors(self):
        for rgb, chroma, near_gray in (((0, 0, 0), 0, 1), ((128, 128, 128), 0, 1),
                                     ((255, 255, 255), 0, 1), ((255, 0, 0), 1, 0)):
            with self.subTest(rgb=rgb):
                records = self.measure(bytes(rgb) * 4)
                self.assertEqual(records["rgb_chroma_mean"]["value"], chroma)
                self.assertEqual(records["near_gray_pixel_fraction"]["value"], near_gray)
                self.assertEqual(records[FLOW]["processing_status"], "missing")
                self.assertEqual(records["frame_interval_cv"]["processing_status"], "missing")

    def test_timing_uses_consecutive_presentation_intervals_not_frame_rate(self):
        records = self.measure(bytes(36), pts=[0, 40, 120])
        self.assertAlmostEqual(records["frame_interval_cv"]["value"], 1 / 3)
        sampling = self.evidence(records, "frame_interval_cv")["sampling"]
        self.assertEqual(sampling["source_timestamps_seconds"], [0, 0.04, 0.12])
        self.assertEqual(sampling["coverage"], "bounded_consecutive_prefix_not_full_clip")
        self.assertIsNone(sampling["source_duration_seconds"])
        self.assertEqual(sampling["frame_count"], 3)
        for field in FIELDS - {"frame_interval_cv"}:
            self.assertNotIn("source_timestamps_seconds", self.evidence(records, field)["sampling"])
        self.assertEqual(self.measure(bytes(24))["frame_interval_cv"]["processing_status"], "missing")

    def test_flow_is_fixed_farneback_mean_magnitude_with_version_and_no_decomposition(self):
        backend = synthetic_backend()
        records = self.measure(bytes(36), backend=backend)
        calls = backend[1].calcOpticalFlowFarneback.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0].kwargs, self.subject.FLOW_PARAMETERS)
        self.assertIsNone(calls[0].args[2])
        self.assertEqual(backend[1].getNumThreads.call_count, 1)
        self.assertEqual(backend[1].setNumThreads.call_args_list[0].args, (1,))
        evidence = self.evidence(records, FLOW)
        self.assertEqual(evidence["opencv_version"], "synthetic-not-real-opencv")
        self.assertEqual(evidence["sampling"]["cut_masking"], "none; cuts can inflate flow")
        self.assertIn("not camera", evidence["definition"])
        self.assertEqual(evidence["aggregates"]["flow_pair_count"], 2)
        self.assertAlmostEqual(evidence["aggregates"]["flow_pair_mean_sum"], 2, places=6)
        self.assertNotIn("flow_pair_means", evidence["aggregates"])

    def test_missing_optional_dependency_only_fails_flow(self):
        runner = Mock(return_value=(bytes(36), showinfo([0, 40, 80])))
        with patch.object(self.subject, "_load_flow_backend", side_effect=ImportError("cv2 absent")):
            records = self.subject.measure_technical("/dev/fd/17", width=2, run_process=runner)
        self.assertEqual(records[FLOW]["processing_status"], "error")
        self.assertEqual(self.evidence(records, FLOW)["reason"], "flow_backend_failed")
        for field in FIELDS - {FLOW}:
            self.assertEqual(records[field]["processing_status"], "complete")

    def test_nonfinite_out_of_bound_or_failed_flow_never_becomes_zero(self):
        for magnitude in (float("nan"), float("inf"), 100):
            records = self.measure(bytes(36), backend=synthetic_backend(magnitude))
            self.assertIsNone(records[FLOW]["value"])
            self.assertEqual(records[FLOW]["processing_status"], "error")
            self.assertEqual(records["rgb_chroma_mean"]["processing_status"], "complete")
        backend = synthetic_backend()
        backend[1].calcOpticalFlowFarneback.side_effect = RuntimeError("synthetic compute failure")
        records = self.measure(bytes(36), backend=backend)
        self.assertEqual(records[FLOW]["processing_status"], "error")

    def test_decoder_reuses_bounded_runner_fd_protocols_pts_and_no_resampling(self):
        records = self.measure(bytes(36), pass_fds=(17,), timeout=4, max_frames=3, window_seconds=2)
        command, deadline, maximum = self.runner.call_args.args
        self.assertEqual(self.runner.call_args.kwargs, {"pass_fds": (17,), "process_group": False,
                                                       "capture_stderr": True})
        for flag, expected in (("-i", "/dev/fd/17"), ("-map", "0:V:0"), ("-f", "mov"),
                               ("-protocol_whitelist", "file,pipe"), ("-vsync", "0"),
                               ("-frames:v", "3"), ("-threads", "1"), ("-filter_threads", "1")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        transform = command[command.index("-vf") + 1]
        self.assertIn("setpts=PTS-STARTPTS", transform)
        self.assertIn("trim=end_frame=3", transform)
        self.assertIn("showinfo", transform)
        self.assertNotIn("fps=", transform)
        self.assertNotIn("select=", transform)
        self.assertEqual(maximum, 36 + 32768 + 3 * 512)
        self.assertLessEqual(deadline, time.monotonic() + 4)
        self.assertTrue(self.evidence(records)["sampling"]["frame_limit_reached"])

    def test_decode_and_pts_errors_never_leave_partial_scores(self):
        for raw, diagnostics in ((bytes(11), showinfo([0])), (bytes(48), showinfo([0, 40, 80, 120])),
                                 (bytes(36), b""), (bytes(36), showinfo([0, 0, 40])),
                                 (bytes(36), showinfo([0, 80, 40])), (bytes(36), showinfo([0, 40, 10000])),
                                 (bytes(36), showinfo([40, 80, 120])), (bytes(36), b"x" * 40000),
                                 (bytes(36), showinfo([0, 40, 80]) + b"[error] invalid data")):
            with self.subTest(length=len(raw), diagnostics=diagnostics[:60]):
                self.assert_unknown(self.measure(raw, diagnostics=diagnostics, max_frames=3), "error")
        self.assert_unknown(self.measure(b""), "missing")

    def test_budget_validation_precedes_process_or_dependency_load(self):
        for options in ({"timeout": 0}, {"timeout": True}, {"timeout": float("nan")}, {"timeout": 301},
                        {"max_frames": 0}, {"max_frames": True}, {"max_frames": 601},
                        {"width": 0}, {"width": 321}, {"width": 1.5},
                        {"window_seconds": 0}, {"window_seconds": 61}, {"window_seconds": float("inf")},
                        {"max_frames": 600, "width": 320}):
            runner = Mock()
            with self.subTest(options=options), patch.object(self.subject, "_load_flow_backend") as backend:
                with self.assertRaises(ValueError):
                    self.subject.measure_technical("/dev/fd/17", run_process=runner, **options)
                runner.assert_not_called()
                backend.assert_not_called()

    def test_compute_timeout_invalidates_all_fields(self):
        backend = synthetic_backend()
        backend[1].calcOpticalFlowFarneback.side_effect = lambda *args, **kwargs: time.sleep(0.1)
        self.assert_unknown(self.measure(bytes(36), backend=backend, timeout=0.05), "error")

    def test_builtin_runner_bounds_output_and_time_and_reaps(self):
        with tempfile.TemporaryDirectory() as directory:
            for body in ("import time; time.sleep(5)", "import os\nwhile True: os.write(2, b'x'*8192)"):
                binary = Path(directory) / "fake-ffmpeg"
                binary.write_text("#!" + sys.executable + "\n" + body + "\n")
                binary.chmod(0o700)
                started = time.monotonic()
                self.assert_unknown(self.subject.measure_technical("ignored", ffmpeg=binary,
                                    width=2, max_frames=3, timeout=0.2), "error")
                self.assertLess(time.monotonic() - started, 2)

    def test_startup_imports_no_optional_compute_dependencies(self):
        code = "import dbp_metrics._technical, dbp_metrics; import sys; assert not {'cv2','numpy','torch'} & set(sys.modules)"
        subprocess.run([sys.executable, "-B", "-c", code], check=True, capture_output=True, timeout=10)


@unittest.skipUnless(importlib.util.find_spec("cv2"), "optional OpenCV unavailable; no installation requested")
class RealOpticalFlowTests(unittest.TestCase):
    def test_fixed_seed_translation_and_stationary_frames(self):
        import numpy
        from dbp_metrics import _technical as pilot_technical

        width = 64
        generator = numpy.random.default_rng(19)
        gray = generator.integers(0, 256, (width, width), dtype=numpy.uint8)
        frame = numpy.repeat(gray[:, :, None], 3, axis=2)
        shifted = numpy.roll(frame, 2, axis=1)
        for following, lower, upper in ((frame, 0, 0.05), (shifted, 1.7, 2.3)):
            records = pilot_technical.measure_technical("unused", width=width,
                run_process=Mock(return_value=(frame.tobytes() + following.tobytes(), showinfo([0, 40], width=width))))
            self.assertEqual(records[FLOW]["processing_status"], "complete")
            self.assertGreaterEqual(records[FLOW]["value"], lower)
            self.assertLess(records[FLOW]["value"], upper)


@unittest.skipUnless(os.path.isfile(FFMPEG) and os.access(FFMPEG, os.X_OK), "optional real FFmpeg unavailable")
class RealTechnicalFFmpegTests(unittest.TestCase):
    def test_cfr_and_vfr_preserve_actual_pts_without_fps_resampling(self):
        from dbp_metrics import _technical as pilot_technical

        with tempfile.TemporaryDirectory() as directory:
            for variable, expected in ((False, 0), (True, 1 / 3)):
                media = Path(directory) / f"fixture-{variable}.mp4"
                command = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-s", "16x16", "-r", "25", "-i", "pipe:0"]
                if variable:
                    command += ["-vf", "setpts=if(eq(N\\,2)\\,3\\,N)"]
                command += ["-vsync", "0", "-an", "-c:v", "libx264", "-threads", "1", "-crf", "0",
                            "-pix_fmt", "yuv444p", str(media)]
                subprocess.run(command, input=bytes([128]) * 16 * 16 * 3 * 3,
                               capture_output=True, timeout=10, check=True)
                with media.open("rb") as descriptor:
                    records = pilot_technical.measure_technical(f"/dev/fd/{descriptor.fileno()}",
                        pass_fds=(descriptor.fileno(),), ffmpeg=FFMPEG, width=16, timeout=10)
                self.assertAlmostEqual(records["frame_interval_cv"]["value"], expected)
                self.assertAlmostEqual(records["rgb_chroma_mean"]["value"], 0, delta=2 / 255)
                self.assertEqual(records["near_gray_pixel_fraction"]["value"], 1)


if __name__ == "__main__":
    unittest.main()
