"""Full-clip FFmpeg cut-candidate contracts; no dataset or ML dependencies."""

import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


FFMPEG = os.environ.get("DBP_TEST_FFMPEG", shutil.which("ffmpeg") or "")


def probe(frames="3", duration="0.3", **video_fields):
    video = {"index": 0, "codec_type": "video", "width": 16, "height": 12,
             "avg_frame_rate": "10/1", "nb_frames": frames,
             "disposition": {"attached_pic": 0}}
    video.update(video_fields)
    return json.dumps({"streams": [video], "format": {"duration": duration}}).encode()


def metadata(scores, pts=None, times=None):
    prefix = "[Parsed_metadata_3 @ 0x1234] [info] "
    lines = ["[info] ffmpeg version test-1 Copyright (c) FFmpeg developers"]
    for index, score in enumerate(scores):
        timestamp = pts[index] if pts is not None else index * 1024
        seconds = times[index] if times is not None else index / 10
        lines.extend([prefix + f"frame:{index} pts:{timestamp} pts_time:{seconds}",
                      prefix + f"lavfi.scene_score={score}"])
    return ("\n".join(lines) + "\n").encode()


class PilotCutsTests(unittest.TestCase):
    def setUp(self):
        try:
            self.subject = importlib.import_module("dbp_metrics._cuts")
        except ModuleNotFoundError:
            self.fail("pilot_cuts implementation is not available")

    def detect(self, scores=(0, 0, 0), probe_raw=None, diagnostics=None, **options):
        self.calls = []

        def runner(command, deadline, max_output, **kwargs):
            self.calls.append((command, deadline, max_output, kwargs))
            return b"", metadata(scores) if diagnostics is None else diagnostics

        return self.subject.detect_ffmpeg_cuts("/dev/fd/17",
            probe() if probe_raw is None else probe_raw, run_process=runner, **options)

    def assert_unknown(self, result, status="error"):
        self.assertIsNone(result["value"], result)
        self.assertEqual(result["processing_status"], status, result)
        self.assertIn("reason", result["evidence"][0])
        self.assertNotIn("cut_frame_indices", result["evidence"][0])
        self.assertLess(len(json.dumps(result, allow_nan=False).encode()), 262144)

    def test_full_single_shot_is_complete_zero_with_provenance(self):
        result = self.detect()
        self.assertEqual(result["value"], 0)
        self.assertEqual(result["processing_status"], "complete")
        evidence = result["evidence"][0]
        self.assertEqual(evidence["detector"], "FFmpeg.scene_score")
        self.assertEqual(evidence["algorithm"], "dbp-ffmpeg-cuts-v1")
        self.assertEqual(evidence["software_version"], "test-1")
        self.assertEqual(evidence["threshold"], 0.3)
        self.assertEqual(evidence["decoded_frames"], 3)
        self.assertEqual(evidence["cut_frame_indices"], [])
        self.assertEqual(evidence["frame_count_source"], "ffprobe.stream.nb_frames")
        self.assertTrue(evidence["candidate_only"])
        self.assertIn("not", evidence["warning"])
        self.assertNotIn("diagnostic", evidence)
        self.assertEqual(evidence["transform"], self.subject.TRANSFORM)
        self.assertNotIn("interval_seconds", evidence)
        self.assertNotIn("interval_time_basis", evidence)
        self.assertNotIn("interval_boundary_rule", evidence)

    def test_threshold_inclusive_excludes_first_frame_and_keeps_observed_pts(self):
        result = self.detect(diagnostics=metadata([1, 0.3, 0.299999],
                            pts=[4096, 6144, 7168], times=[0.4, 0.6, 0.7]))
        self.assertEqual(result["value"], 1)
        evidence = result["evidence"][0]
        self.assertEqual(evidence["cut_frame_indices"], [1])
        self.assertEqual(evidence["cut_pts"], [6144])
        self.assertEqual(evidence["cut_pts_time_seconds"], [0.6])
        self.assertIn("decoded_frame_indices", evidence["timing_basis"])
        self.assertIn("rounded", evidence["timestamp_basis"])
        self.assertEqual(self.detect([0, 0.2, 0.1], threshold=0.1)["value"], 2)
        self.assertEqual(self.detect([0, 0, 0], threshold=0)["value"], 2)
        self.assertEqual(self.detect([0], probe_raw=probe("1"))["value"], 0)

    def test_command_preserves_fd_and_decodes_every_frame_with_sentinel_budget(self):
        started = time.monotonic()
        self.detect(pass_fds=(17,), timeout=4, ffmpeg="/opt/ffmpeg")
        command, deadline, max_output, options = self.calls[0]
        self.assertEqual(command[0], "/opt/ffmpeg")
        for flag, value in {"-i": "/dev/fd/17", "-map": "0:V:0", "-hwaccel": "none",
                            "-threads": "1", "-threads:v": "1", "-filter_threads": "1",
                            "-frames:v": "4", "-vsync": "0",
                            "-protocol_whitelist": "file,pipe"}.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertEqual(command[:command.index("-i")][-2:], ["-f", "mov"])
        for flag in ("-nostdin", "-nostats", "-noautorotate", "-an", "-sn", "-dn", "-xerror"):
            self.assertIn(flag, command)
        transform = command[command.index("-vf") + 1]
        self.assertEqual(transform, self.subject.TRANSFORM)
        self.assertIn("scale=160:-2", transform)
        self.assertIn("format=yuv420p", transform)
        self.assertIn("select='gte(scene,0)'", transform)
        self.assertIn("metadata=mode=print:key=lavfi.scene_score", transform)
        self.assertNotIn("fps=", transform)
        self.assertNotIn("-t", command)
        self.assertEqual(command[-3:], ["-f", "null", "pipe:1"])
        self.assertEqual(options, {"pass_fds": (17,), "process_group": False,
                                   "capture_stderr": True})
        self.assertGreaterEqual(deadline, started + 4)
        self.assertLessEqual(deadline, time.monotonic() + 4)
        self.assertLessEqual(max_output, 32768 + 4 * 512)

    def test_interval_configuration_is_strict_and_validated_before_launch(self):
        invalid = [(), (0,), (0, 1, 2), [0, 1], "0,1", True,
                   (True, 1), (0, False), ("0", 1), (0, "1"),
                   (float("nan"), 1), (0, float("inf")), (-0.1, 1),
                   (0, 0), (1, 1), (2, 1), (0, 600.1)]
        for interval in invalid:
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                self.detect(interval=interval)
            self.assertEqual(self.calls, [])

        accepted = self.detect(interval=(0, 0.3005))
        self.assertEqual(accepted["processing_status"], "complete", accepted)
        self.assertEqual(len(self.calls), 1)
        with self.assertRaises(ValueError):
            self.detect(interval=(0, 0.302))
        self.assertEqual(self.calls, [])

    def test_timestamp_origin_configuration_is_validated_before_launch(self):
        for options in ({"timestamp_origin": "container"},
                        {"interval": (0, 0.2), "timestamp_origin": "other"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.detect(**options)
            self.assertEqual(self.calls, [])

    def test_explicit_first_frame_origin_preserves_default_interval_command(self):
        self.detect(interval=(0, 0.2))
        default_command = self.calls[0][0]
        self.detect(interval=(0, 0.2), timestamp_origin="first_frame")
        self.assertEqual(self.calls[0][0], default_command)
        self.assertNotIn("-copyts", default_command)

    def test_interval_counts_only_transitions_whose_adjacent_frames_are_inside(self):
        result = self.detect(diagnostics=metadata([0, 1, 1, 1],
            pts=[90000, 91024, 92048, 93072], times=[0, 1, 2, 3]),
            probe_raw=probe("4", duration="4"), interval=(1, 3))
        self.assertEqual(result["processing_status"], "complete", result)
        self.assertEqual(result["value"], 1)
        evidence = result["evidence"][0]
        self.assertEqual(evidence["algorithm"], "dbp-ffmpeg-cuts-interval-v1")
        self.assertEqual(evidence["decoded_frames"], 4)
        self.assertEqual(evidence["interval_observed_frames"], 2)
        self.assertEqual(evidence["cut_frame_indices"], [2])
        self.assertEqual(evidence["cut_pts"], [92048])
        self.assertEqual(evidence["cut_pts_time_seconds"], [2.0])
        self.assertEqual(evidence["interval_seconds"], {"start": 1, "end": 3})
        self.assertIn("setpts=PTS-STARTPTS", evidence["interval_time_basis"])
        self.assertIn("setpts=PTS-STARTPTS", evidence["timestamp_basis"])
        self.assertNotIn("default timestamp origin", evidence["timestamp_basis"])
        self.assertIn("both", evidence["interval_boundary_rule"])
        self.assertIn("[start,end)", evidence["interval_boundary_rule"])
        self.assertEqual(evidence["coverage"], "complete_indexed_clip")
        transform = self.calls[0][0][self.calls[0][0].index("-vf") + 1]
        self.assertEqual(transform, "setpts=PTS-STARTPTS," + self.subject.TRANSFORM)
        self.assertNotIn("trim", transform)

    def test_container_interval_preserves_source_pts_and_both_neighbor_rule(self):
        result = self.detect(diagnostics=metadata([0, 1, 1],
            pts=[100, 600, 1100], times=[0.1, 0.6, 1.1]),
            probe_raw=probe(duration="1.2"), interval=(0, 1.05),
            timestamp_origin="container")
        self.assertEqual(result["processing_status"], "complete", result)
        self.assertEqual(result["value"], 1)
        evidence = result["evidence"][0]
        self.assertEqual(evidence["algorithm"], "dbp-ffmpeg-cuts-container-interval-v1")
        self.assertEqual(evidence["interval_observed_frames"], 2)
        self.assertEqual(evidence["cut_frame_indices"], [1])
        self.assertEqual(evidence["cut_pts"], [600])
        self.assertEqual(evidence["cut_pts_time_seconds"], [0.6])
        self.assertIn("zero-origin container", evidence["interval_time_basis"])
        self.assertIn("first video frame", evidence["interval_time_basis"])
        self.assertNotIn("setpts=PTS-STARTPTS", evidence["timestamp_basis"])
        command = self.calls[0][0]
        self.assertLess(command.index("-copyts"), command.index("-i"))
        transform = command[command.index("-vf") + 1]
        self.assertEqual(transform, self.subject.TRANSFORM)
        self.assertNotIn("setpts", transform)

    def test_interval_with_no_observed_frames_is_missing_but_one_frame_is_zero(self):
        absent = self.detect(diagnostics=metadata([0, 1, 1], times=[0, 0.1, 0.2]),
                             interval=(0.4, 0.5), probe_raw=probe(duration="0.5"))
        self.assertIsNone(absent["value"], absent)
        self.assertEqual(absent["processing_status"], "missing", absent)
        absent_evidence = absent["evidence"][0]
        self.assertEqual(absent_evidence["reason"], "no_observed_frames_in_interval")
        self.assertEqual(absent_evidence["decoded_frames"], 3)
        self.assertEqual(absent_evidence["interval_observed_frames"], 0)
        self.assertEqual(absent_evidence["coverage"], "complete_indexed_clip")

        one_frame = self.detect(diagnostics=metadata([0, 1, 1], times=[0, 0.1, 0.2]),
                                interval=(0.1, 0.2))
        self.assertEqual(one_frame["processing_status"], "complete", one_frame)
        self.assertEqual(one_frame["value"], 0)
        one_frame_evidence = one_frame["evidence"][0]
        self.assertEqual(one_frame_evidence["interval_observed_frames"], 1)
        self.assertEqual(one_frame_evidence["cut_frame_indices"], [])

    def test_interval_still_requires_full_indexed_decode_coverage(self):
        result = self.detect(diagnostics=metadata([0, 1], times=[0, 0.1]),
                             interval=(0, 0.15))
        self.assert_unknown(result)

    def test_unknown_invalid_or_unbounded_indexed_count_never_launches(self):
        for frames in (None, "N/A", "", "0", "18001", "3.0", "-1", True, 3.0):
            with self.subTest(frames=frames):
                self.assert_unknown(self.detect(probe_raw=probe(frames)))
                self.assertEqual(self.calls, [])
        self.assertEqual(self.detect(probe_raw=probe(3))["value"], 0)

    def test_probe_safety_and_resource_bounds_reject_before_decode(self):
        invalid = [probe(duration="601"), probe(duration="NaN"), probe(duration=None),
                   probe(width=4096, height=4096), probe(width=0), probe(width=True),
                   probe(height=None), probe(width=1, height=8388608),
                   b'{}', b'{"streams":[],"streams":[]}', b'{"duration":NaN}',
                   b'\xff', b' ' * 131073]
        for raw in invalid:
            with self.subTest(raw=raw[:200]):
                self.assert_unknown(self.detect(probe_raw=raw))
                self.assertEqual(self.calls, [])

    def test_nonattached_video_count_used_not_cover_art(self):
        payload = json.loads(probe())
        payload["streams"].insert(0, {"index": 1, "codec_type": "video", "nb_frames": "1",
                                      "disposition": {"attached_pic": 1}})
        self.assertEqual(self.detect(probe_raw=json.dumps(payload))["value"], 0)
        payload["streams"][0]["disposition"]["attached_pic"] = 0
        self.assert_unknown(self.detect(probe_raw=json.dumps(payload)))
        self.assertEqual(self.calls, [])

    def test_missing_partial_extra_frames_or_sparse_scores_never_become_zero(self):
        for scores in ([], [0], [0, 0], [0, 0, 0, 0]):
            with self.subTest(scores=scores):
                self.assert_unknown(self.detect(scores))

    def test_malformed_unpaired_nonfinite_or_nonmonotonic_metadata_is_error(self):
        valid = metadata([0, 0, 0])
        invalid = [valid.replace(b"frame:1", b"frame:2"),
                   valid.replace(b"pts:1024", b"pts:0"),
                   valid.replace(b"pts_time:0.1", b"pts_time:0.0"),
                   valid.replace(b"pts_time:0.1", b"pts_time:nan"),
                   valid.replace(b"pts_time:0.2", b"pts_time:601"),
                   valid.replace(b"lavfi.scene_score=0", b"lavfi.scene_score=NaN", 1),
                   valid.replace(b"lavfi.scene_score=0", b"lavfi.scene_score=inf", 1),
                   valid.replace(b"lavfi.scene_score=0", b"lavfi.scene_score=-0.1", 1),
                   valid.replace(b"lavfi.scene_score=0", b"lavfi.scene_score=1.1", 1),
                   valid.replace(b"lavfi.scene_score=0", b"lavfi.scene_score=", 1),
                   valid.replace(b"lavfi.scene_score", b"missing_score", 1),
                   valid + valid.splitlines()[-1] + b"\n",
                   valid.replace(b"frame:1 pts:1024", b"frame:1 pts:NOPTS"),
                   valid + b"[decoder] [error] corrupt frame\n", valid + b"\xff",
                   valid.replace(b"ffmpeg version test-1", b"missing version")]
        for diagnostics in invalid:
            with self.subTest(diagnostics=diagnostics):
                self.assert_unknown(self.detect(diagnostics=diagnostics))

    def test_failures_have_private_bounded_evidence_and_no_values(self):
        errors = [FileNotFoundError("/private/secret.mp4"), OSError("secret"),
                  ValueError("secret" * 100000), subprocess.TimeoutExpired("secret", 1),
                  subprocess.CalledProcessError(1, "secret")]
        for error in errors:
            def runner(*args, **kwargs):
                raise error

            with self.subTest(error=type(error).__name__):
                result = self.subject.detect_ffmpeg_cuts("/dev/fd/17", probe(), run_process=runner)
                self.assert_unknown(result, "missing" if isinstance(error, FileNotFoundError) else "error")
                self.assertNotIn("secret", json.dumps(result))
        self.assert_unknown(self.detect(diagnostics=b"x" * 100000))

    def test_invalid_configuration_raises_without_launch(self):
        for options in ({"timeout": 0}, {"timeout": 301}, {"timeout": True},
                        {"timeout": float("nan")}, {"timeout": "30"},
                        {"threshold": -0.1}, {"threshold": 1.01}, {"threshold": True},
                        {"threshold": float("inf")}, {"threshold": "0.3"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.detect(**options)

    def test_maximum_clip_and_evidence_cut_cap(self):
        result = self.detect(probe_raw=probe("18000", duration="600"),
                             diagnostics=metadata([0] * 18000, times=[index / 30 for index in range(18000)]))
        self.assertEqual(result["value"], 0)
        self.assertLess(len(json.dumps(result).encode()), 262144)
        result = self.detect([0] + [1] * 2048, probe_raw=probe("2049", duration="205"))
        self.assertEqual(result["value"], 2048)
        self.assertLess(len(json.dumps(result).encode()), 131072)
        self.assert_unknown(self.detect([0] + [1] * 2049,
                            probe_raw=probe("2050", duration="205")))

    def test_default_runner_enforces_deadline_and_output_limit(self):
        from dbp_metrics._runtime import run_bounded

        for script, timeout in (("import time; time.sleep(5)", 0.1),
                                ("import sys; sys.stderr.write('x'*100000)", 2)):
            def runner(command, deadline, max_output, **kwargs):
                return run_bounded([sys.executable, "-c", script], deadline, max_output, **kwargs)

            started = time.monotonic()
            result = self.subject.detect_ffmpeg_cuts("/dev/fd/17", probe(),
                                                    timeout=timeout, run_process=runner)
            self.assert_unknown(result)
            self.assertLess(time.monotonic() - started, 3)

    def test_import_does_not_load_application_or_ml(self):
        completed = subprocess.run([sys.executable, "-c",
            "import sys; import dbp_metrics._cuts; "
            "assert not {'app','torch','transformers','scenedetect','cv2','pilot_extract'} & set(sys.modules)"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        self.assertEqual(completed.returncode, 0, completed.stderr)


@unittest.skipUnless(os.path.isfile(FFMPEG) and os.access(FFMPEG, os.X_OK),
                     "optional real FFmpeg unavailable; set DBP_TEST_FFMPEG")
class RealFFmpegCutsTests(unittest.TestCase):
    def detect_fixture(self, levels, indexed_frames=None, video_delay=0, **options):
        subject = importlib.import_module("dbp_metrics._cuts")
        with tempfile.TemporaryDirectory(prefix="dbp-cuts-real-") as directory:
            media_path = Path(directory) / "fixture.mp4"
            subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "16x12",
                            "-r", "10", "-i", "pipe:0", "-an", "-c:v", "libx264",
                            "-threads", "1", "-crf", "0", "-pix_fmt", "yuv420p",
                            "-vf", f"setpts=PTS+{video_delay}/TB", "-vsync", "0", str(media_path)],
                           input=b"".join(bytes([level]) * 16 * 12 * 3 for level in levels),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10, check=True)
            with media_path.open("rb") as media:
                result = subject.detect_ffmpeg_cuts("/dev/fd/" + str(media.fileno()),
                    probe(str(len(levels) if indexed_frames is None else indexed_frames),
                          duration=str(len(levels) / 10)),
                    pass_fds=(media.fileno(),), ffmpeg=FFMPEG, **options)
                os.fstat(media.fileno())
                return result

    def test_continuous_single_color_has_no_cut(self):
        result = self.detect_fixture([64] * 20)
        self.assertEqual(result["processing_status"], "complete", result)
        self.assertEqual(result["value"], 0)
        self.assertEqual(result["evidence"][0]["decoded_frames"], 20)

    def test_concatenated_black_white_cut_is_exact_first_white_frame(self):
        result = self.detect_fixture([0] * 10 + [255] * 10)
        self.assertEqual(result["processing_status"], "complete", result)
        self.assertEqual(result["value"], 1)
        self.assertEqual(result["evidence"][0]["cut_frame_indices"], [10])
        self.assertEqual(result["evidence"][0]["cut_pts_time_seconds"], [1.0])

    def test_interval_uses_half_open_adjacent_frame_boundaries(self):
        levels = [0] * 10 + [255] * 10
        result = self.detect_fixture(levels, interval=(0.9, 1.1))
        self.assertEqual(result["processing_status"], "complete", result)
        self.assertEqual(result["value"], 1)
        self.assertEqual(result["evidence"][0]["cut_frame_indices"], [10])
        self.assertEqual(result["evidence"][0]["cut_pts_time_seconds"], [1.0])
        self.assertEqual(self.detect_fixture(levels, interval=(1.0, 1.1))["value"], 0)
        self.assertEqual(self.detect_fixture(levels, interval=(0.9, 1.0))["value"], 0)

    def test_container_interval_preserves_delayed_video_pts(self):
        result = self.detect_fixture([0] * 5 + [255] * 5 + [0] * 5, video_delay=0.1,
                                     interval=(0, 1.05), timestamp_origin="container")
        self.assertEqual(result["processing_status"], "complete", result)
        self.assertEqual(result["value"], 1)
        self.assertEqual(result["evidence"][0]["cut_frame_indices"], [5])
        self.assertEqual(result["evidence"][0]["cut_pts_time_seconds"], [0.6])

    def test_underestimated_and_overestimated_frame_metadata_are_unknown(self):
        for indexed in (19, 21):
            with self.subTest(indexed=indexed):
                result = self.detect_fixture([0] * 20, indexed_frames=indexed)
                self.assertIsNone(result["value"], result)
                self.assertEqual(result["processing_status"], "error", result)


if __name__ == "__main__":
    unittest.main()
