"""CPU audio contracts using synthetic PCM and bounded subprocess fixtures."""

import importlib
import json
import math
import os
import shutil
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


FIELDS = {"audio_rms_amplitude", "audio_silence_fraction"}
FFMPEG = os.environ.get("DBP_TEST_FFMPEG", shutil.which("ffmpeg") or "")


def pcm(samples):
    return struct.pack("<" + "h" * len(samples), *samples)


class PilotAudioTests(unittest.TestCase):
    def setUp(self):
        try:
            self.subject = importlib.import_module("dbp_metrics._audio")
        except ModuleNotFoundError:
            self.fail("pilot_audio implementation is not available")

    def measure(self, raw, diagnostics=b"", **options):
        self.calls = []

        def runner(command, deadline, max_output, **kwargs):
            self.calls.append((command, deadline, max_output, kwargs))
            return raw, diagnostics

        return self.subject.measure_audio("/dev/fd/17", run_process=runner, **options)

    def evidence(self, result, field="audio_rms_amplitude"):
        return result[field]["evidence"][0]

    def assert_unknown(self, result, status, reason):
        self.assertEqual(set(result), FIELDS)
        for record in result.values():
            self.assertIsNone(record["value"])
            self.assertEqual(record["processing_status"], status)
            self.assertEqual(record["evidence"][0]["reason"], reason)
            self.assertNotEqual(record["evidence"][0]["sampling"]["coverage"],
                                "full_decoded_audio")

    def test_silence_is_measured_only_when_samples_exist(self):
        result = self.measure(bytes(640), has_audio=True)
        self.assertEqual(set(result), FIELDS)
        self.assertEqual(result["audio_rms_amplitude"]["value"], 0)
        self.assertEqual(result["audio_silence_fraction"]["value"], 1)
        for record in result.values():
            self.assertEqual(record["processing_status"], "complete")
        json.dumps(result, allow_nan=False)

    def test_constant_and_full_scale_normalize_by_32768(self):
        for level in (16384, -16384, -32768, 32767):
            with self.subTest(level=level):
                result = self.measure(pcm([level] * 320))
                self.assertAlmostEqual(result["audio_rms_amplitude"]["value"], abs(level) / 32768)
                self.assertEqual(result["audio_silence_fraction"]["value"], 0)

    def test_rms_is_pooled_energy_not_mean_window_rms_or_mean_sample(self):
        result = self.measure(pcm([0] * 320 + [16384, -16384] * 160))
        self.assertAlmostEqual(result["audio_rms_amplitude"]["value"], math.sqrt(0.125))
        self.assertEqual(result["audio_silence_fraction"]["value"], 0.5)

    def test_squared_sample_sum_is_exact_and_reconstructs_rms(self):
        for samples in ([0], [-32768, 32767, -33, 32, 1, -1],
                        [16384] * 320 + [0] * 80):
            with self.subTest(sample_count=len(samples)):
                result = self.measure(pcm(samples))
                expected = sum(sample ** 2 for sample in samples)
                for record in result.values():
                    sampling = record["evidence"][0]["sampling"]
                    self.assertIs(type(sampling["squared_sample_sum"]), int)
                    self.assertEqual(sampling["squared_sample_sum"], expected)
                rms = math.sqrt(sampling["squared_sample_sum"] / sampling["sample_count"]) / 32768
                self.assertEqual(result["audio_rms_amplitude"]["value"], rms)

    def test_maximum_squared_sample_sum_is_json_safe_integer(self):
        result = self.measure(pcm([-32768] * 960000))
        serialized = json.loads(json.dumps(result, allow_nan=False))
        for record in serialized.values():
            squared_sum = record["evidence"][0]["sampling"]["squared_sample_sum"]
            self.assertIs(type(squared_sum), int)
            self.assertEqual(squared_sum, 960000 * 32768 ** 2)
            self.assertLess(squared_sum, 2 ** 53)
        self.assertEqual(result["audio_rms_amplitude"]["value"], 1)

    def test_partial_final_window_is_sample_weighted(self):
        result = self.measure(pcm([16384] * 320 + [0] * 80))
        self.assertAlmostEqual(result["audio_rms_amplitude"]["value"], math.sqrt(0.2))
        self.assertEqual(result["audio_silence_fraction"]["value"], 0.2)
        sampling = self.evidence(result)["sampling"]
        self.assertEqual(sampling["silent_samples"], 80)
        self.assertEqual(sampling["window_count"], 2)
        self.assertEqual(sampling["partial_window_samples"], 80)
        self.assertEqual(sampling["decoded_duration_seconds"], 0.025)

    def test_one_sample_audio_is_not_missing(self):
        result = self.measure(pcm([0]))
        self.assertEqual(result["audio_silence_fraction"]["value"], 1)
        self.assertEqual(self.evidence(result)["sampling"]["sample_count"], 1)
        self.assertEqual(self.evidence(result)["sampling"]["partial_window_samples"], 1)

    def test_silence_threshold_uses_block_rms_not_peak_or_pointwise_count(self):
        for level, expected in ((32, 1), (33, 0), (-32, 1), (-33, 0)):
            with self.subTest(level=level):
                result = self.measure(pcm([level] * 320))
                self.assertEqual(result["audio_silence_fraction"]["value"], expected)
        below = self.measure(pcm([100] + [0] * 319))
        above = self.measure(pcm([1000] + [0] * 319))
        self.assertEqual(below["audio_silence_fraction"]["value"], 1)
        self.assertEqual(above["audio_silence_fraction"]["value"], 0)

    def test_no_track_empty_audio_and_unconfirmed_empty_are_distinct(self):
        result = self.measure(b"", has_audio=False)
        self.assert_unknown(result, "missing", "no_audio_track")
        self.assertEqual(self.calls, [])
        self.assert_unknown(self.measure(b"", has_audio=True), "missing", "empty_audio")
        self.assert_unknown(self.measure(b""), "missing", "no_decoded_audio")

    def test_no_stream_runner_error_is_missing_only_when_track_presence_unknown(self):
        def runner(*args, **options):
            raise ValueError("subprocess failed or reported errors: Output file #0 does not contain any stream\n")

        result = self.subject.measure_audio("/dev/fd/17", run_process=runner)
        self.assert_unknown(result, "missing", "no_audio_track")
        result = self.subject.measure_audio("/dev/fd/17", has_audio=True, run_process=runner)
        self.assert_unknown(result, "error", "decode_failed")

    def test_odd_and_nonbyte_output_never_yield_partial_values(self):
        for raw in (bytes(641), "not PCM", None):
            with self.subTest(raw=raw):
                self.assert_unknown(self.measure(raw), "error", "invalid_pcm_output")

    def test_exact_sample_budget_completes_but_one_sentinel_is_unknown(self):
        complete = self.measure(bytes(640), max_duration_seconds=0.02)
        self.assertEqual(complete["audio_rms_amplitude"]["processing_status"], "complete")
        for count in (321, 322, 1000):
            with self.subTest(count=count):
                result = self.measure(bytes(count * 2), max_duration_seconds=0.02)
                self.assert_unknown(result, "missing", "sample_limit_exceeded")
                self.assertTrue(self.evidence(result)["sampling"]["sample_limit_reached"])

    def test_default_budget_has_sixty_seconds_and_exactly_one_sentinel(self):
        result = self.measure(bytes(640))
        sampling = self.evidence(result)["sampling"]
        expected = {
            "sample_rate_hz": 16000, "channels": 1, "sample_format": "s16le",
            "normalization_divisor": 32768, "silence_threshold_rms": 0.001,
            "silence_comparison": "<=", "window_samples": 320, "window_seconds": 0.02,
            "partial_window_weighting": "sample_count", "sample_count": 320,
            "silent_samples": 320, "squared_sample_sum": 0,
            "window_count": 1, "partial_window_samples": 0,
            "decoded_duration_seconds": 0.02, "coverage": "full_decoded_audio",
            "sample_limit_reached": False, "max_duration_seconds": 60,
            "max_samples": 960000, "sentinel_samples": 1, "max_raw_bytes": 1920002,
            "max_combined_output_bytes": 1952770, "timeout_seconds": 30,
            "stream_selection": "0:a:0 (first audio stream)",
        }
        for key, value in expected.items():
            self.assertEqual(sampling[key], value, key)
        self.assertIn("phase cancellation", sampling["downmix"])
        self.assertIn("not loudness", sampling["downmix"])
        for record in result.values():
            self.assertEqual(record["evidence"][0]["sampling"], sampling)
            self.assertEqual(record["evidence"][0]["algorithm"], "dbp-audio-pcm-v1")
            self.assertTrue(record["evidence"][0]["definition"])
        self.assertIsNot(result["audio_rms_amplitude"]["evidence"][0]["sampling"],
                         result["audio_silence_fraction"]["evidence"][0]["sampling"])

    def test_command_and_callback_restrict_cpu_protocol_demuxer_and_output(self):
        started = time.monotonic()
        self.measure(bytes(640), ffmpeg="/trusted/ffmpeg", pass_fds=(17,),
                     has_audio=True, process_group=True, timeout=4, max_duration_seconds=1)
        command, deadline, maximum, options = self.calls[0]
        self.assertEqual(command[0], "/trusted/ffmpeg")
        self.assertEqual(command[command.index("-i") + 1], "/dev/fd/17")
        before = command[:command.index("-i")]
        after = command[command.index("-i") + 2:]
        self.assertEqual(before[before.index("-protocol_whitelist") + 1], "file,pipe")
        self.assertEqual(before[before.index("-f") + 1], "mov")
        self.assertEqual(before[before.index("-threads") + 1], "1")
        self.assertEqual(before[before.index("-err_detect") + 1], "explode")
        self.assertEqual(command[command.index("-map") + 1], "0:a:0")
        self.assertEqual(command[command.index("-ac") + 1], "1")
        self.assertEqual(command[command.index("-ar") + 1], "16000")
        self.assertEqual(command[command.index("-threads:a") + 1], "1")
        self.assertEqual(command[command.index("-filter_threads") + 1], "1")
        self.assertEqual(after[after.index("-f") + 1], "s16le")
        self.assertEqual(command[command.index("-c:a") + 1], "pcm_s16le")
        self.assertEqual(command[command.index("-loglevel") + 1], "error")
        for flag in ("-vn", "-sn", "-dn", "-nostdin", "-nostats", "-xerror"):
            self.assertIn(flag, command)
        transform = command[command.index("-af") + 1]
        self.assertIn("aresample=16000:async=0:dither_method=none", transform)
        self.assertIn("sample_fmts=s16:channel_layouts=mono", transform)
        self.assertIn("atrim=end_sample=16001", transform)
        self.assertIn("asetpts=N/SR/TB", transform)
        self.assertAlmostEqual(float(command[command.index("-t") + 1]), 16001 / 16000)
        self.assertEqual(command[-1], "pipe:1")
        self.assertEqual(maximum, 32002 + 32768)
        self.assertEqual(options, {"pass_fds": (17,), "process_group": True,
                                   "capture_stderr": True})
        self.assertGreaterEqual(deadline, started + 4)
        self.assertLessEqual(deadline, time.monotonic() + 4)

    def test_parent_deadline_never_extends_and_default_group_stays_with_worker(self):
        deadline = time.monotonic() + 2
        self.measure(bytes(640), deadline=deadline, timeout=20)
        self.assertEqual(self.calls[0][1], deadline)
        self.assertFalse(self.calls[0][3]["process_group"])
        self.measure(bytes(640), deadline=time.monotonic() + 1000, timeout=1)
        self.assertLessEqual(self.calls[0][1], time.monotonic() + 1)

    def test_expired_deadline_does_not_launch_and_late_callback_is_rejected(self):
        self.assert_unknown(self.measure(bytes(640), deadline=time.monotonic() - 1),
                            "error", "decode_failed")
        self.assertEqual(self.calls, [])

        def runner(*args, **options):
            time.sleep(0.03)
            return bytes(640), b""

        result = self.subject.measure_audio("ignored", timeout=0.01, run_process=runner)
        self.assert_unknown(result, "error", "decode_failed")

    def test_decode_failure_timeout_and_stderr_never_become_measurements(self):
        for error in (FileNotFoundError("not installed"), OSError("bad descriptor"),
                      ValueError("subprocess output limit exceeded"),
                      subprocess.TimeoutExpired("ffmpeg", 1)):
            def runner(*args, **options):
                raise error

            with self.subTest(error=type(error).__name__):
                result = self.subject.measure_audio("ignored", run_process=runner)
                self.assert_unknown(result, "error", "decode_failed")
        for diagnostics in (b"corrupt packet", b"[error] truncated input", b"x" * 32769,
                            "not bytes", None):
            with self.subTest(diagnostics=type(diagnostics).__name__):
                result = self.measure(bytes(640), diagnostics=diagnostics)
                self.assert_unknown(result, "error", "decode_failed")
                self.assertLess(len(json.dumps(result)), 12000)

    def test_invalid_controls_raise_before_launch(self):
        invalid = [{"timeout": value} for value in (0, -1, True, 301, "1", math.inf, math.nan)]
        invalid += [{"max_duration_seconds": value} for value in
                    (0, -1, True, 61, "1", math.inf, math.nan, 1 / 32000)]
        invalid += [{"deadline": value} for value in (True, "1", math.inf, math.nan)]
        invalid += [{"has_audio": value} for value in (0, 1, "true")]
        invalid += [{"process_group": value} for value in (0, 1, "false")]
        for options in invalid:
            with self.subTest(options=options), patch.object(self.subject.subprocess, "Popen") as process:
                with self.assertRaises(ValueError):
                    self.subject.measure_audio("ignored", **options)
                process.assert_not_called()

    def executable(self, body):
        temporary = tempfile.TemporaryDirectory(prefix="dbp-audio-test-")
        self.addCleanup(temporary.cleanup)
        executable = Path(temporary.name) / "fake-ffmpeg"
        executable.write_text("#!" + sys.executable + "\n" + body, encoding="utf-8")
        executable.chmod(0o700)
        return str(executable)

    def test_builtin_runner_passes_held_descriptor_and_preserves_ownership(self):
        binary = self.executable(
            "import os, sys\n"
            "path = sys.argv[sys.argv.index('-i') + 1]\n"
            "with open(path, 'rb') as media:\n"
            "    os.write(1, media.read())\n")
        with tempfile.TemporaryFile() as media:
            media.write(pcm([16384] * 320))
            media.flush()
            media.seek(0)
            descriptor = media.fileno()
            result = self.subject.measure_audio("/dev/fd/" + str(descriptor),
                pass_fds=(descriptor,), ffmpeg=binary, timeout=3)
            self.assertEqual(result["audio_rms_amplitude"]["value"], 0.5)
            os.fstat(descriptor)

    def test_builtin_runner_bounds_both_streams_and_reaps(self):
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
                result = self.subject.measure_audio("ignored", ffmpeg=binary,
                    max_duration_seconds=0.02, timeout=3)
                self.assert_unknown(result, "error", "decode_failed")
                self.assertIn("output limit", self.evidence(result)["diagnostic"])
                self.assertIsNotNone(processes[0].poll())
                self.assertTrue(processes[0].stdout.closed)
                self.assertTrue(processes[0].stderr.closed)

    def test_builtin_runner_times_out_even_after_pipes_close(self):
        for body in ("import time\ntime.sleep(10)\n",
                     "import os, time\nos.close(1)\nos.close(2)\ntime.sleep(10)\n"):
            with self.subTest(body=body):
                binary = self.executable(body)
                started = time.monotonic()
                result = self.subject.measure_audio("ignored", ffmpeg=binary, timeout=0.2)
                self.assert_unknown(result, "error", "decode_failed")
                self.assertLess(time.monotonic() - started, 2)

    def test_builtin_runner_rejects_nonzero_and_error_stderr(self):
        for suffix in ("raise SystemExit(2)\n", "os.write(2, b'truncated decode')\n"):
            binary = self.executable("import os\nos.write(1, bytes(640))\n" + suffix)
            result = self.subject.measure_audio("ignored", ffmpeg=binary, timeout=3)
            self.assert_unknown(result, "error", "decode_failed")


@unittest.skipUnless(os.path.isfile(FFMPEG) and os.access(FFMPEG, os.X_OK),
                     "optional real FFmpeg unavailable; set DBP_TEST_FFMPEG")
class RealFFmpegAudioTests(unittest.TestCase):
    def setUp(self):
        try:
            self.subject = importlib.import_module("dbp_metrics._audio")
        except ModuleNotFoundError:
            self.fail("pilot_audio implementation is not available")
        temporary = tempfile.TemporaryDirectory(prefix="dbp-audio-real-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def fixture(self, samples, channels=1, rate=16000):
        media_path = self.directory / "fixture.mov"
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                        "-f", "s16le", "-ac", str(channels), "-ar", str(rate), "-i", "pipe:0",
                        "-c:a", "pcm_s16le", "-threads", "1", "-movflags", "+faststart",
                        str(media_path)], input=pcm(samples), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, timeout=10, check=True)
        return media_path

    def measure_path(self, media_path, **options):
        with media_path.open("rb") as media:
            result = self.subject.measure_audio("/dev/fd/" + str(media.fileno()),
                pass_fds=(media.fileno(),), ffmpeg=FFMPEG, timeout=5, **options)
            os.fstat(media.fileno())
            return result

    def test_lossless_silence_constant_and_mixed_partial_windows(self):
        for samples, rms, fraction in (([0] * 400, 0, 1), ([16384] * 400, 0.5, 0),
                                       ([16384] * 320 + [0] * 80, math.sqrt(0.2), 0.2)):
            with self.subTest(rms=rms, fraction=fraction):
                result = self.measure_path(self.fixture(samples), has_audio=True)
                record = result["audio_rms_amplitude"]
                self.assertEqual(record["processing_status"], "complete", record)
                self.assertAlmostEqual(record["value"], rms)
                self.assertAlmostEqual(result["audio_silence_fraction"]["value"], fraction)
                self.assertEqual(record["evidence"][0]["sampling"]["sample_count"], 400)
                self.assertEqual(record["evidence"][0]["sampling"]["squared_sample_sum"],
                                 sum(sample ** 2 for sample in samples))

    def test_stereo_phase_cancellation_is_mono_energy_not_channel_energy(self):
        result = self.measure_path(self.fixture([16384, -16384] * 1600, channels=2))
        self.assertEqual(result["audio_rms_amplitude"]["value"], 0, result)
        self.assertEqual(result["audio_silence_fraction"]["value"], 1)

    def test_resamples_to_sixteen_khz(self):
        result = self.measure_path(self.fixture([0] * 4800, rate=48000))
        record = result["audio_rms_amplitude"]
        self.assertEqual(record["processing_status"], "complete", record)
        self.assertEqual(record["value"], 0)
        self.assertEqual(record["evidence"][0]["sampling"]["sample_count"], 1600)

    def test_full_sixty_second_boundary_and_one_sample_over(self):
        for count, status in ((960000, "complete"), (960001, "missing"), (976000, "missing")):
            with self.subTest(count=count):
                result = self.measure_path(self.fixture([0] * count))
                record = result["audio_rms_amplitude"]
                self.assertEqual(record["processing_status"], status, record)
                if status == "missing":
                    self.assertIsNone(record["value"])
                    self.assertEqual(record["evidence"][0]["reason"], "sample_limit_exceeded")
                else:
                    self.assertEqual(record["evidence"][0]["sampling"]["sample_count"], count)

    def test_video_only_is_missing_without_parent_probe(self):
        media_path = self.directory / "video.mp4"
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin",
                        "-f", "lavfi", "-i", "color=black:size=16x16:rate=1", "-frames:v", "1",
                        "-an", "-c:v", "mpeg4", "-threads", "1", str(media_path)],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=True)
        result = self.measure_path(media_path)
        for record in result.values():
            self.assertIsNone(record["value"])
            self.assertEqual(record["processing_status"], "missing", record)
            self.assertEqual(record["evidence"][0]["reason"], "no_audio_track")

    def test_truncated_and_malformed_inputs_never_return_partial_metrics(self):
        media_path = self.fixture([16384] * 16000)
        raw = media_path.read_bytes()
        for payload in (raw[:-4000], b"not an MP4 file"):
            with self.subTest(size=len(payload)):
                damaged = self.directory / "damaged.mov"
                damaged.write_bytes(payload)
                result = self.measure_path(damaged, has_audio=True)
                for record in result.values():
                    self.assertIsNone(record["value"])
                    self.assertEqual(record["processing_status"], "error", record)


if __name__ == "__main__":
    unittest.main()
