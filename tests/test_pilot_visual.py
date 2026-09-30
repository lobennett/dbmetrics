"""CPU visual contracts using synthetic RGB and Python subprocess fixtures."""

import importlib
from fractions import Fraction
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


FIELDS = {"brightness_mean", "contrast_mean", "colorfulness_mean", "temporal_change_mean"}
FFMPEG = os.environ.get("DBP_TEST_FFMPEG", shutil.which("ffmpeg") or "")


def showinfo(pts, width=2, time_base="1/1000"):
    prefix = "[Parsed_showinfo_5 @ 0x1234] [info] "
    lines = [prefix + "config in time_base: " + time_base + ", frame_rate: 0/0"]
    for index, timestamp in enumerate(pts):
        seconds = float(timestamp * Fraction(time_base))
        lines.append(prefix + f"n: {index} pts: {timestamp} pts_time:{seconds:.6g} "
                     f"pos: 0 fmt:rgb24 sar:1/1 s:{width}x{width} i:P iskey:1 type:I")
    return ("\n".join(lines) + "\n").encode()


class PilotVisualTests(unittest.TestCase):
    def setUp(self):
        try:
            self.subject = importlib.import_module("dbp_metrics._visual")
        except ModuleNotFoundError:
            self.fail("pilot_visual implementation is not available")

    def measure(self, raw, pts=None, diagnostics=None, **options):
        self.calls = []
        if pts is None:
            spacing = max(500, round(1000 / options.get("sample_fps", 2)))
            pts = [index * spacing for index in range(len(raw) // 12)]

        def runner(command, deadline, max_output, **kwargs):
            self.calls.append((command, deadline, max_output, kwargs))
            return raw, showinfo(pts) if diagnostics is None else diagnostics

        return self.subject.measure_visual("/dev/fd/17", width=2, run_process=runner, **options)

    def evidence(self, result, field="brightness_mean"):
        return result[field]["evidence"][0]

    def test_interval_preserves_source_timestamps_and_rejects_outside_frames(self):
        result = self.measure(bytes([255]) * 24, pts=[1100, 1600], interval=(1.05, 2))
        self.assertEqual(result["brightness_mean"]["value"], 1)
        sampling = self.evidence(result)["sampling"]
        self.assertEqual(sampling["source_timestamps_seconds"], [1.1, 1.6])
        self.assertEqual(sampling["requested_interval_seconds"], [1.05, 2])
        self.assertEqual(self.evidence(result)["algorithm"], "dbp-visual-rgb-interval-v2")
        for timestamps in ([1000, 1600], [1100, 2000]):
            result = self.measure(bytes(24), pts=timestamps, interval=(1.05, 2))
            self.assertIsNone(result["brightness_mean"]["value"])
            self.assertEqual(result["brightness_mean"]["processing_status"], "error")

    def test_bad_or_truncated_interval_is_rejected_before_decoding(self):
        for interval in ((-1, 2), (2, 2), (2, 1), (True, 2), (0, float("nan")),
                         (0, 601), (0,), "0,1", (0, 10 ** 1000)):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.measure(bytes(12), interval=interval)
        with self.assertRaises(ValueError):
            self.measure(bytes(12), interval=(0, 3), max_frames=2)

    def test_container_origin_requires_interval_and_retains_delayed_video_pts(self):
        result = self.measure(bytes(24), pts=[100, 600], interval=(0, 1.05), timestamp_origin="container")
        self.assertEqual(result["brightness_mean"]["evidence"][0]["sampling"]["source_timestamps_seconds"], [0.1, 0.6])
        self.assertEqual(result["brightness_mean"]["evidence"][0]["algorithm"], "dbp-visual-rgb-container-interval-v1")
        for options in (dict(timestamp_origin="container"), dict(interval=(0, 1), timestamp_origin="other")):
            with self.assertRaises(ValueError):
                self.measure(bytes(12), **options)

    def assert_unknown(self, result, status, reason):
        self.assertEqual(set(result), FIELDS)
        for record in result.values():
            self.assertIsNone(record["value"])
            self.assertEqual(record["processing_status"], status)
            self.assertEqual(record["evidence"][0]["reason"], reason)

    def test_black_white_frames_have_unit_temporal_change_not_contrast(self):
        result = self.measure(bytes(12) + bytes([255]) * 12)
        self.assertEqual(set(result), FIELDS)
        for field, expected in {"brightness_mean": 0.5, "contrast_mean": 0.0,
                                "colorfulness_mean": 0.0, "temporal_change_mean": 1.0}.items():
            self.assertAlmostEqual(result[field]["value"], expected)
            self.assertEqual(result[field]["processing_status"], "complete")
        json.dumps(result, allow_nan=False)

    def test_spatial_population_contrast_and_stationary_temporal_change(self):
        frame = bytes(6) + bytes([255]) * 6
        result = self.measure(frame * 3)
        self.assertAlmostEqual(result["brightness_mean"]["value"], 0.5)
        self.assertAlmostEqual(result["contrast_mean"]["value"], 0.5)
        self.assertEqual(result["temporal_change_mean"]["value"], 0.0)

    def test_encoded_luma_uses_rec709_weights_without_linearization(self):
        for rgb, expected in [((255, 0, 0), 0.2126), ((0, 255, 0), 0.7152),
                              ((0, 0, 255), 0.0722), ((128, 128, 128), 128 / 255)]:
            with self.subTest(rgb=rgb):
                result = self.measure(bytes(rgb) * 4)
                self.assertAlmostEqual(result["brightness_mean"]["value"], expected)
                self.assertEqual(result["contrast_mean"]["value"], 0.0)

    def test_colorfulness_is_normalized_opponent_proxy_not_clipped_to_one(self):
        red = self.measure(bytes((255, 0, 0)) * 4)
        self.assertAlmostEqual(red["colorfulness_mean"]["value"], 0.3 * math.sqrt(1.25))
        result = self.measure(bytes((255, 0, 0, 0, 255, 255)) * 2)
        self.assertAlmostEqual(result["colorfulness_mean"]["value"], math.sqrt(1.25))
        self.assertIn("Hasler", self.evidence(result, "colorfulness_mean")["definition"])

    def test_metrics_average_per_frame_not_pooled_pixels(self):
        frame = bytes(6) + bytes([255]) * 6
        result = self.measure(frame + bytes(12) + bytes([255]) * 12)
        self.assertAlmostEqual(result["contrast_mean"]["value"], 1 / 6)
        self.assertAlmostEqual(result["temporal_change_mean"]["value"], 0.75)

    def test_single_frame_keeps_spatial_metrics_but_temporal_is_unknown(self):
        result = self.measure(bytes(12))
        self.assertEqual(result["brightness_mean"]["value"], 0.0)
        record = result["temporal_change_mean"]
        self.assertIsNone(record["value"])
        self.assertEqual(record["processing_status"], "missing")
        self.assertEqual(record["evidence"][0]["reason"], "insufficient_sampled_frames")

    def test_observed_variable_cadence_never_claims_requested_grid_or_full_coverage(self):
        result = self.measure(bytes(36), pts=[0, 700, 1400], max_frames=3, sample_fps=2)
        sampling = self.evidence(result)["sampling"]
        self.assertEqual(sampling["frame_count"], 3)
        self.assertEqual(sampling["source_timestamps_seconds"], [0.0, 0.7, 1.4])
        self.assertEqual(sampling["observed_sample_span_seconds"], 1.4)
        self.assertEqual(sampling["minimum_spacing_seconds"], 0.5)
        self.assertNotIn("nominal_timestamps_seconds", sampling)
        self.assertNotIn("nominal_sample_span_seconds", sampling)
        self.assertEqual(sampling["requested_window_seconds"], 1.5)
        self.assertTrue(sampling["frame_limit_reached"])
        self.assertEqual(sampling["coverage"], "not_established")
        self.assertIsNone(sampling["source_duration_seconds"])
        self.assertIn("observed", sampling["timestamp_basis"])
        self.assertIn("first decoded frame", sampling["timestamp_basis"])
        self.assertIn("variable", sampling["cadence_basis"])
        self.assertIn("square", sampling["transform"])
        self.assertIn("rgb24", sampling["representation"])
        self.assertEqual(self.evidence(result)["algorithm"], "dbp-visual-rgb-v2")
        for record in result.values():
            self.assertEqual(record["evidence"][0]["sampling"], sampling)
        uncapped = self.measure(bytes(12), max_frames=3)
        self.assertFalse(self.evidence(uncapped)["sampling"]["frame_limit_reached"])
        self.assertEqual(self.evidence(uncapped)["sampling"]["coverage"], "not_established")

    def test_command_and_callback_keep_descriptor_and_bound_cpu_bytes_and_time(self):
        started = time.monotonic()
        result = self.measure(bytes(24), pass_fds=(17,), ffmpeg="/usr/bin/ffmpeg",
                              timeout=4, max_frames=2, sample_fps=4)
        command, deadline, max_output, options = self.calls[0]
        self.assertEqual(command[0], "/usr/bin/ffmpeg")
        self.assertEqual(command[command.index("-i") + 1], "/dev/fd/17")
        self.assertEqual(command[command.index("-map") + 1], "0:V:0")
        self.assertEqual(command[command.index("-frames:v") + 1], "2")
        self.assertEqual(float(command[command.index("-t") + 1]), 1.0)
        self.assertEqual(command[command.index("-pix_fmt") + 1], "rgb24")
        output_options = command[command.index("-i") + 2:]
        self.assertEqual(output_options[output_options.index("-f") + 1], "rawvideo")
        self.assertEqual(command[command.index("-hwaccel") + 1], "none")
        self.assertEqual(command[command.index("-threads") + 1], "1")
        self.assertLess(command.index("-threads"), command.index("-i"))
        self.assertEqual(command[command.index("-threads:v") + 1], "1")
        self.assertEqual(command[command.index("-filter_threads") + 1], "1")
        self.assertEqual(command[command.index("-vsync") + 1], "0")
        for flag in ("-an", "-sn", "-dn", "-nostdin", "-noautorotate", "-nostats", "-xerror"):
            self.assertIn(flag, command)
        self.assertEqual(command[command.index("-loglevel") + 1], "level+info")
        transform = command[command.index("-vf") + 1]
        self.assertIn("setpts=PTS-STARTPTS", transform)
        self.assertNotIn("fps=", transform)
        self.assertIn("select=", transform)
        self.assertIn("prev_selected_t", transform)
        self.assertIn("trim=duration=1.0", transform)
        self.assertIn("trim=end_frame=2", transform)
        self.assertIn("showinfo", transform)
        self.assertIn("scale=2:2", transform)
        self.assertEqual(command[-1], "pipe:1")
        self.assertEqual(options, {"pass_fds": (17,), "process_group": False,
                                   "capture_stderr": True})
        self.assertGreaterEqual(deadline, started + 4)
        self.assertLess(deadline, time.monotonic() + 4)
        self.assertEqual(max_output, 24 + 32768 + 2 * 512)
        sampling = self.evidence(result)["sampling"]
        self.assertEqual(sampling["max_raw_bytes"], 24)
        self.assertEqual(sampling["minimum_spacing_seconds"], 0.5)
        self.assertEqual(sampling["timeout_seconds"], 4)
        self.assertEqual(self.evidence(result)["ffmpeg_binary"], "/usr/bin/ffmpeg")

    def test_input_is_forced_mov_with_only_local_file_and_pipe_protocols(self):
        self.measure(bytes(12), pass_fds=(17,))
        command = self.calls[0][0]
        input_options = command[:command.index("-i")]
        self.assertIn("-protocol_whitelist", input_options)
        self.assertEqual(input_options[input_options.index("-protocol_whitelist") + 1], "file,pipe")
        self.assertIn("-f", input_options)
        self.assertEqual(input_options[input_options.index("-f") + 1], "mov")

    def test_default_budget_is_small_and_explicit(self):
        def runner(command, deadline, max_output, **options):
            return bytes(160 * 160 * 3), showinfo([0], width=160)

        result = self.subject.measure_visual("/dev/fd/17", run_process=runner)
        sampling = self.evidence(result)["sampling"]
        self.assertEqual(sampling["max_frames"], 120)
        self.assertEqual(sampling["sample_fps"], 2)
        self.assertEqual(sampling["width"], 160)
        self.assertEqual(sampling["height"], 160)
        self.assertEqual(sampling["requested_window_seconds"], 60)
        self.assertEqual(sampling["max_raw_bytes"], 9216000)
        self.assertEqual(sampling["timeout_seconds"], 30)

    def test_empty_decode_is_missing_and_partial_or_excess_frames_are_errors(self):
        empty = self.measure(b"")
        self.assert_unknown(empty, "missing", "no_sampled_frames")
        self.assertEqual(self.evidence(empty)["sampling"]["source_timestamps_seconds"], [])
        for raw in (bytes(11), bytes(13), bytes(36)):
            with self.subTest(size=len(raw)):
                self.assert_unknown(self.measure(raw, max_frames=2), "error", "invalid_raw_output")

    def test_missing_duplicate_out_of_order_or_unpaired_pts_are_errors(self):
        valid = showinfo([0, 500])
        invalid = [b"", showinfo([0]), showinfo([0, 500, 1000]), showinfo([0, 0]),
                   showinfo([500, 0]), showinfo([0, 250]), showinfo([0, 1000]),
                   showinfo([-500, 0]), showinfo([500, 1000]),
                   valid.replace(b"1/1000", b"1/0"), valid.replace(b"1/1000", b"0/1000"),
                   valid.replace(b"config in time_base", b"missing time base"),
                   valid + valid.splitlines()[0] + b"\n", valid.replace(b"n: 1", b"n: 2"),
                   valid.replace(b"fmt:rgb24", b"fmt:yuv420p"), valid.replace(b"s:2x2", b"s:2x3"),
                   valid.replace(b"pts_time:0.5", b"pts_time:nan"),
                   valid.replace(b"pts_time:0.5", b"pts_time:0.9"),
                   valid + b"[decoder] [error] corrupt frame\n",
                   valid + b"[decoder] [fatal] invalid data\n", valid + b"\xff"]
        for diagnostics in invalid:
            with self.subTest(diagnostics=diagnostics):
                result = self.measure(bytes(24), diagnostics=diagnostics, max_frames=2)
                self.assert_unknown(result, "error", "decode_failed")
        self.assert_unknown(self.measure(b"", diagnostics=showinfo([0])), "error", "decode_failed")

    def test_integer_pts_and_time_base_not_rounded_pts_time_define_timestamps(self):
        result = self.measure(bytes(36), diagnostics=showinfo([0, 2, 4], time_base="1/3"))
        self.assertEqual(self.evidence(result)["sampling"]["source_timestamps_seconds"],
                         [0.0, 2 / 3, 4 / 3])

    def test_decode_errors_and_diagnostics_do_not_become_measurements(self):
        for error in (FileNotFoundError("not installed"), ValueError("output limit"),
                      subprocess.TimeoutExpired("ffmpeg", 1), OSError("bad descriptor")):
            def runner(*args, **options):
                raise error

            with self.subTest(error=type(error).__name__):
                result = self.subject.measure_visual("/dev/fd/17", run_process=runner)
                self.assert_unknown(result, "error", "decode_failed")

        def noisy_runner(*args, **options):
            return bytes(12), b"corrupt frame" * 10000

        result = self.subject.measure_visual("/dev/fd/17", width=2, run_process=noisy_runner)
        self.assert_unknown(result, "error", "decode_failed")
        self.assertLess(len(json.dumps(result)), 16000)

    def test_invalid_resource_controls_raise_before_launch(self):
        invalid = [{"timeout": 0}, {"timeout": float("nan")}, {"timeout": float("inf")},
                   {"timeout": True}, {"timeout": 301}, {"max_frames": 0},
                   {"max_frames": True}, {"max_frames": 1.5}, {"max_frames": 601},
                   {"width": 0}, {"width": 321}, {"width": True}, {"width": 2.5},
                   {"sample_fps": 0}, {"sample_fps": True}, {"sample_fps": "2"},
                   {"sample_fps": float("inf")}, {"sample_fps": 31},
                   {"sample_fps": 0.01}, {"max_frames": 600, "sample_fps": 0.1},
                   {"max_frames": 600, "width": 320}]
        for options in invalid:
            with self.subTest(options=options):
                with patch.object(self.subject.subprocess, "Popen") as process:
                    with self.assertRaises(ValueError):
                        self.subject.measure_visual("/dev/fd/17", **options)
                    process.assert_not_called()

    def executable(self, body):
        temporary = tempfile.TemporaryDirectory(prefix="dbp-visual-test-")
        self.addCleanup(temporary.cleanup)
        executable = Path(temporary.name) / "fake-ffmpeg"
        executable.write_text("#!" + sys.executable + "\n" + body, encoding="utf-8")
        executable.chmod(0o700)
        return str(executable)

    def test_real_pipe_runner_passes_held_descriptor_without_reopening_original_path(self):
        binary = self.executable(
            "import os, sys\n"
            "path = sys.argv[sys.argv.index('-i') + 1]\n"
            "with open(path, 'rb') as media:\n"
            "    os.write(1, media.read())\n"
            "os.write(2, " + repr(showinfo([0])) + ")\n")
        with tempfile.TemporaryFile() as media:
            media.write(bytes((255, 0, 0)) * 4)
            media.flush()
            media.seek(0)
            descriptor = media.fileno()
            result = self.subject.measure_visual("/dev/fd/" + str(descriptor),
                pass_fds=(descriptor,), ffmpeg=binary, width=2, timeout=3)
            self.assertAlmostEqual(result["brightness_mean"]["value"], 0.2126)
            os.fstat(descriptor)

    def test_real_pipe_runner_bounds_both_streams_and_reaps_process(self):
        for channel in (1, 2):
            binary = self.executable("import os\nwhile True:\n    os.write(" +
                                     str(channel) + ", b'x' * 8192)\n")
            original = subprocess.Popen
            processes = []

            def spawn(*args, **options):
                process = original(*args, **options)
                processes.append(process)
                self.assertFalse(options.get("start_new_session", False))
                return process

            with self.subTest(channel=channel), patch.object(self.subject.subprocess, "Popen", spawn):
                result = self.subject.measure_visual("ignored", ffmpeg=binary, width=1,
                                                     max_frames=1, timeout=3)
                self.assert_unknown(result, "error", "decode_failed")
                self.assertIn("output limit", self.evidence(result)["diagnostic"])
                self.assertIsNotNone(processes[0].poll())
                self.assertTrue(processes[0].stdout.closed)
                self.assertTrue(processes[0].stderr.closed)

    def test_real_pipe_runner_times_out_even_after_pipes_close(self):
        for body in ("import time\ntime.sleep(10)\n",
                     "import os, time\nos.close(1)\nos.close(2)\ntime.sleep(10)\n"):
            with self.subTest(body=body):
                binary = self.executable(body)
                started = time.monotonic()
                result = self.subject.measure_visual("ignored", ffmpeg=binary, timeout=0.2)
                self.assert_unknown(result, "error", "decode_failed")
                self.assertLess(time.monotonic() - started, 2)

    def test_real_pipe_runner_rejects_nonzero_exit_and_error_stderr(self):
        for suffix in ("raise SystemExit(2)\n", "os.write(2, b'decode error')\n"):
            binary = self.executable("import os\nos.write(1, bytes(12))\n" + suffix)
            result = self.subject.measure_visual("ignored", ffmpeg=binary, width=2, timeout=3)
            self.assert_unknown(result, "error", "decode_failed")


@unittest.skipUnless(os.path.isfile(FFMPEG) and os.access(FFMPEG, os.X_OK),
                     "optional real FFmpeg unavailable; set DBP_TEST_FFMPEG")
class RealFFmpegVisualTests(unittest.TestCase):
    def test_container_mode_preserves_video_delay_instead_of_rebasing_the_segment(self):
        result = self.measure_fixture([0] * 10 + [255] * 10, 10, video_delay=0.1,
                                      interval=(0, 1.05), timestamp_origin="container")
        self.assertEqual(result["brightness_mean"]["processing_status"], "complete", result)
        self.assertLess(result["brightness_mean"]["value"], 0.01)
        self.assertEqual(result["brightness_mean"]["evidence"][0]["sampling"]["source_timestamps_seconds"], [0.1, 0.6])

    def test_sub_frame_interval_boundaries_do_not_round_into_neighboring_frames(self):
        for interval, expected in (((1.0000001, 2), [1.1, 1.6]), ((0, 1.0000001), [0, 0.5, 1])):
            with self.subTest(interval=interval):
                result = self.measure_fixture([0] * 10 + [255] * 10, 10, interval=interval)
                self.assertEqual(result["brightness_mean"]["processing_status"], "complete", result)
                self.assertEqual(result["brightness_mean"]["evidence"][0]["sampling"]["source_timestamps_seconds"], expected)

    def test_exact_half_uses_only_its_frames_not_parent_brightness(self):
        levels = [0] * 10 + [255] * 10
        parent = self.measure_fixture(levels, 10, interval=(0, 2))
        first = self.measure_fixture(levels, 10, interval=(0, 1))
        second = self.measure_fixture(levels, 10, interval=(1, 2))
        self.assertAlmostEqual(parent["brightness_mean"]["value"], 0.5, delta=0.01)
        self.assertLess(first["brightness_mean"]["value"], 0.01)
        self.assertGreater(second["brightness_mean"]["value"], 0.99)
        self.assertEqual(second["brightness_mean"]["evidence"][0]["sampling"]["source_timestamps_seconds"], [1, 1.5])
        self.assertEqual(second["temporal_change_mean"]["value"], 0)

    def measure_fixture(self, levels, rate, **options):
        subject = importlib.import_module("dbp_metrics._visual")
        video_delay = options.pop("video_delay", 0)
        with tempfile.TemporaryDirectory(prefix="dbp-visual-real-") as directory:
            media_path = Path(directory) / "fixture.mp4"
            subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "16x12",
                            "-r", str(rate), "-i", "pipe:0", "-an", "-c:v", "libx264",
                            "-threads", "1", "-crf", "0", "-pix_fmt", "yuv444p",
                            "-vf", f"setpts=PTS+{video_delay}/TB", "-vsync", "0",
                            str(media_path)],
                           input=b"".join(bytes([level]) * 16 * 12 * 3 for level in levels),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=True)
            with media_path.open("rb") as media:
                result = subject.measure_visual("/dev/fd/" + str(media.fileno()),
                    pass_fds=(media.fileno(),), ffmpeg=FFMPEG, width=2, **options)
                os.fstat(media.fileno())
                return result

    def test_one_source_frame_one_second_has_unknown_temporal_change(self):
        result = self.measure_fixture([0], 1)
        temporal = result["temporal_change_mean"]
        self.assertIsNone(temporal["value"])
        self.assertEqual(temporal["processing_status"], "missing")
        sampling = temporal["evidence"][0]["sampling"]
        self.assertEqual(sampling["frame_count"], 1)
        self.assertEqual(sampling["source_timestamps_seconds"], [0.0])

    def test_two_source_frames_at_quarter_fps_are_one_undiluted_difference(self):
        result = self.measure_fixture([0, 255], 0.25)
        temporal = result["temporal_change_mean"]
        self.assertAlmostEqual(temporal["value"], 1.0)
        self.assertEqual(temporal["processing_status"], "complete")
        sampling = temporal["evidence"][0]["sampling"]
        self.assertEqual(sampling["frame_count"], 2)
        self.assertEqual(sampling["source_timestamps_seconds"], [0.0, 4.0])
        self.assertEqual(sampling["observed_sample_span_seconds"], 4.0)
        self.assertEqual(sampling["coverage"], "not_established")

    def test_observed_cadence_and_frame_cap_on_nonintegral_requested_grid(self):
        result = self.measure_fixture([0, 255] * 10, 3, max_frames=3)
        record = result["brightness_mean"]
        self.assertEqual(record["processing_status"], "complete", record)
        sampling = record["evidence"][0]["sampling"]
        self.assertEqual(sampling["source_timestamps_seconds"], [0.0, 2 / 3, 4 / 3])
        self.assertEqual(sampling["frame_count"], 3)
        self.assertTrue(sampling["frame_limit_reached"])

    def test_source_frame_beyond_prefix_is_not_measured(self):
        result = self.measure_fixture([0, 255], 0.125, max_frames=3, sample_fps=0.5)
        record = result["brightness_mean"]
        self.assertEqual(record["processing_status"], "complete", record)
        sampling = record["evidence"][0]["sampling"]
        self.assertEqual(sampling["source_timestamps_seconds"], [0.0])
        self.assertEqual(sampling["requested_window_seconds"], 6.0)
        self.assertFalse(sampling["frame_limit_reached"])
        self.assertIsNone(result["temporal_change_mean"]["value"])

    def test_default_frame_cap_fits_bounded_showinfo_diagnostics(self):
        result = self.measure_fixture([0, 255] * 150, 4)
        record = result["brightness_mean"]
        self.assertEqual(record["processing_status"], "complete", record)
        sampling = record["evidence"][0]["sampling"]
        self.assertEqual(sampling["frame_count"], 120)
        self.assertEqual(sampling["source_timestamps_seconds"], [index / 2 for index in range(120)])


if __name__ == "__main__":
    unittest.main()
