"""Offline extractor regressions using only temporary software fixtures."""

import hashlib
import importlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch


CLIP_ID = "fixture-source_part_00000001"
PROBE = {
    "streams": [
        {"index": 0, "codec_type": "video", "width": 640, "height": 360,
         "avg_frame_rate": "30000/1001", "nb_read_frames": "90", "nb_frames": "90"},
        {"index": 1, "codec_type": "audio"},
    ],
    "format": {"duration": "3.003"},
}


class PilotExtractTests(unittest.TestCase):
    def setUp(self):
        try:
            self.subject = importlib.import_module("dbp_metrics.legacy")
        except ModuleNotFoundError:
            self.fail("dbp_metrics.legacy implementation is not available")
        temporary = tempfile.TemporaryDirectory(prefix="dbp-extract-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.media_root = self.root / "media"
        self.media_root.mkdir()
        self.media = self.media_root / (CLIP_ID + ".mp4")
        self.media.write_bytes(b"software fixture, not real video")
        self.manifest = self.root / "manifest.json"
        self.write_manifest([self.clip()])
        self.output = self.root / "measurements.json"
        self.probe = self.executable("import json\nprint(" + repr(json.dumps(PROBE)) + ")\n")

    def clip(self, **changes):
        return {"clip_id": CLIP_ID, "source_id": "fixture-source", "part": 1} | changes

    def write_manifest(self, clips):
        self.manifest.write_text(json.dumps({
            "schema_version": "dbt1-evidence-manifest-v1", "clips": clips,
        }), encoding="utf-8")

    def executable(self, body):
        executable = self.root / "fixture-ffprobe"
        executable.write_text("#!" + sys.executable + "\n" + body, encoding="utf-8")
        executable.chmod(0o700)
        return str(executable)

    def run_extract(self, **changes):
        options = {"manifest_path": self.manifest, "media_root": self.media_root,
                   "output_path": self.output, "media_kind": "original",
                   "ffprobe": self.probe, "timeout": 5, "run_id": "fixture-run"}
        return self.subject.extract_pilot(**(options | changes))

    def records(self):
        return {record["field_id"]: record for record in json.loads(
            self.output.read_text(encoding="utf-8"))["records"]}

    def test_probe_measurements_have_exact_schema_and_private_provenance(self):
        self.assertEqual(self.run_extract(), 0)
        payload = json.loads(self.output.read_text())
        self.assertEqual(set(payload), {"schema_version", "manifest_sha256", "run_id", "records"})
        self.assertEqual(payload["schema_version"], "dbt1-pilot-measurements-v1")
        self.assertEqual(payload["manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        self.assertEqual(payload["run_id"], "fixture-run")
        expected = {"duration_seconds": 3.003, "width_px": 640, "height_px": 360,
                    "frame_rate": 30000 / 1001, "audio_track_present": True}
        for field, value in expected.items():
            record = self.records()[field]
            self.assertEqual(set(record), {"clip_id", "field_id", "value", "method",
                "processing_status", "review_status", "evidence", "media_sha256",
                "media_path", "media_kind", "model_id", "model_revision"})
            self.assertEqual(record["value"], value)
            self.assertEqual(record["processing_status"], "complete")
            self.assertEqual(record["method"], "ffprobe")
            self.assertEqual(record["media_sha256"], hashlib.sha256(self.media.read_bytes()).hexdigest())
            self.assertEqual(record["media_kind"], "original")
            self.assertTrue(record["evidence"])
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o600)
        self.assertEqual(set(self.records()), set(expected))

    def test_isolated_worker_imports_shared_semantics_before_running_probe(self):
        options = {"clip": self.clip(), "media_root": str(self.media_root), "media_kind": "original",
                   "ffprobe": self.probe, "timeout": 5, "max_media_bytes": 1024,
                   "cuts": False, "cut_threshold": 27.0, "stage": "probe"}
        worker = subprocess.run([sys.executable, "-I", "-B", self.subject.__file__, "--worker", json.dumps(options)],
                                cwd=self.root, capture_output=True, timeout=10)
        self.assertEqual(worker.returncode, 0, worker.stderr.decode())
        payload = json.loads(worker.stdout)
        self.assertFalse(payload["failed"])
        records = {record["field_id"]: record for record in payload["records"]}
        self.assertEqual(records["duration_seconds"]["value"], 3.003)
        self.assertEqual(records["duration_seconds"]["processing_status"], "complete")
        self.assertEqual(records["faces_visible"]["processing_status"], "missing")
        self.assertEqual(records["faces_visible"]["evidence"], [{"reason": "not_requested"}])

    def test_manifest_rejects_identity_mismatch_duplicates_traversal_and_oversize(self):
        bad_clips = [[self.clip(source_id="other")], [self.clip(part=True)],
                     [self.clip(part=2)], [self.clip(clip_id="../" + CLIP_ID)],
                     [self.clip(clip_id=CLIP_ID + "\n")], [self.clip(), self.clip()],
                     [self.clip(clip_id=f"fixture-source_part_{part:08d}", part=part)
                      for part in range(10001)], []]
        for clips in bad_clips:
            with self.subTest(clips=clips):
                self.write_manifest(clips)
                with self.assertRaises(ValueError):
                    self.subject.load_manifest(self.manifest)
        self.manifest.write_bytes(b" " * (self.subject.MAX_MANIFEST_BYTES + 1))
        with self.assertRaises(ValueError):
            self.subject.load_manifest(self.manifest)

    def test_manifest_hash_is_over_exact_bytes_and_32_clips_are_allowed(self):
        self.write_manifest([self.clip(clip_id=f"fixture-source_part_{part:08d}", part=part)
                             for part in range(32)])
        clips, digest = self.subject.load_manifest(self.manifest)
        self.assertEqual(len(clips), 32)
        self.assertEqual(digest, hashlib.sha256(self.manifest.read_bytes()).hexdigest())

    def test_large_manifest_retains_exact_identity_and_hash(self):
        self.write_manifest([self.clip(clip_id=f"fixture-source_part_{part:08d}", part=part)
                             for part in range(5000)])
        try:
            clips, digest = self.subject.load_manifest(self.manifest)
        except ValueError as error:
            self.fail(f"The approved 5000-clip manifest was refused: {error}")
        self.assertEqual(len(clips), 5000)
        self.assertEqual(clips[-1]["part"], 4999)
        self.assertEqual(digest, hashlib.sha256(self.manifest.read_bytes()).hexdigest())

    def test_batches_select_exact_members_without_changing_manifest_hash(self):
        additional = self.clip(clip_id="second-source_part_00000002", source_id="second-source", part=2)
        self.write_manifest([self.clip(), additional])
        (self.media_root / (additional["clip_id"] + ".mp4")).write_bytes(b"second software fixture")
        try:
            status = self.run_extract(offset=1, limit=1)
        except TypeError as error:
            self.fail(f"Bounded extraction batches are unavailable: {error}")
        self.assertEqual(status, 0)
        payload = json.loads(self.output.read_text())
        self.assertEqual({record["clip_id"] for record in payload["records"]}, {additional["clip_id"]})
        self.assertEqual(payload["manifest_sha256"], hashlib.sha256(self.manifest.read_bytes()).hexdigest())

    def test_batch_bounds_refuse_invalid_or_empty_selection_before_output(self):
        for options in ({"offset": -1}, {"offset": True}, {"offset": 1}, {"limit": 0}, {"limit": True}, {"limit": 10001}):
            with self.subTest(options=options):
                try:
                    with self.assertRaises(ValueError):
                        self.run_extract(**options)
                except TypeError as error:
                    self.fail(f"Batch validation is unavailable: {error}")
                self.assertFalse(self.output.exists())

    def visual_executable(self, body):
        executable = self.root / "fixture-ffmpeg"
        executable.write_text("#!" + sys.executable + "\n" + body, encoding="utf-8")
        executable.chmod(0o700)
        return str(executable)

    def test_visual_stage_emits_measured_proxies_with_media_provenance(self):
        diagnostics = ("[Parsed_showinfo_5] config in time_base: 1/1000\n"
                       "[Parsed_showinfo_5] n: 0 pts: 0 pts_time:0 fmt:rgb24 s:160x160\n"
                       "[Parsed_showinfo_5] n: 1 pts: 500 pts_time:0.5 fmt:rgb24 s:160x160\n")
        ffmpeg = self.visual_executable("import sys\nsys.stderr.write(" + repr(diagnostics) + ")\n"
                                       "sys.stdout.buffer.write(bytes(160*160*3) + bytes([255])*(160*160*3))\n")
        try:
            status = self.run_extract(stage="visual", ffmpeg=ffmpeg)
        except ValueError as error:
            self.fail(f"CPU visual measurements are unavailable: {error}")
        self.assertEqual(status, 0)
        records = self.records()
        self.assertEqual(set(records), {"brightness_mean", "contrast_mean", "colorfulness_mean", "temporal_change_mean"})
        for field, value in {"brightness_mean": 0.5, "contrast_mean": 0, "colorfulness_mean": 0, "temporal_change_mean": 1}.items():
            self.assertAlmostEqual(records[field]["value"], value)
            self.assertEqual(records[field]["method"], "ffmpeg_visual")
            self.assertEqual(records[field]["media_sha256"], hashlib.sha256(self.media.read_bytes()).hexdigest())
            self.assertEqual(records[field]["evidence"][0]["sampling"]["coverage"], "not_established")

    def test_audio_stage_decodes_without_ffprobe(self):
        self.probe = str(self.root / "unavailable-ffprobe")
        ffmpeg = self.visual_executable("import sys\nsys.stdout.buffer.write(bytes(1280))\n")
        self.assertEqual(self.run_extract(stage="audio", ffmpeg=ffmpeg), 0)
        self.assertEqual(self.records()["audio_silence_fraction"]["value"], 1)
        self.assertEqual(self.records()["audio_rms_amplitude"]["value"], 0)

    def test_audio_stage_does_not_require_unambiguous_video_stream_layout(self):
        payload = json.loads(json.dumps(PROBE))
        payload["streams"].append({**payload["streams"][0], "index": 2})
        self.probe = self.executable("import json\nprint(" + repr(json.dumps(payload)) + ")\n")
        ffmpeg = self.visual_executable("import sys\nsys.stdout.buffer.write(bytes(1280))\n")
        self.assertEqual(self.run_extract(stage="audio", ffmpeg=ffmpeg), 0)
        self.assertEqual(self.records()["audio_silence_fraction"]["value"], 1)
        with self.assertRaises(ValueError):
            self.subject.parse_probe(json.dumps(payload).encode())

    def test_visual_decode_failure_does_not_become_zero_motion_or_darkness(self):
        ffmpeg = self.visual_executable("import sys\nsys.stderr.write('fixture decode failed')\nsys.exit(1)\n")
        try:
            status = self.run_extract(stage="visual", ffmpeg=ffmpeg)
        except ValueError as error:
            self.fail(f"CPU visual measurements are unavailable: {error}")
        self.assertEqual(status, 1)
        self.assertEqual(len(self.records()), 4)
        for record in self.records().values():
            self.assertIsNone(record["value"])
            self.assertEqual(record["processing_status"], "error")

    def test_explicit_ffmpeg_cut_stage_retains_complete_frame_evidence(self):
        lines = ["ffmpeg version 4.4 Copyright FFmpeg developers"]
        for index in range(90):
            lines.append(f"[Parsed_metadata_3 @ fixture] [info] frame:{index} pts:{index * 40} pts_time:{index * 0.04}")
            lines.append(f"[Parsed_metadata_3 @ fixture] [info] lavfi.scene_score={0.8 if index == 45 else 0.0}")
        ffmpeg = self.visual_executable("import sys\nsys.stderr.write(" + repr("\n".join(lines) + "\n") + ")\n")
        try:
            status = self.run_extract(stage="ffmpeg_cuts", ffmpeg=ffmpeg)
        except ValueError as error:
            self.fail(f"The explicit FFmpeg cut backend is unavailable: {error}")
        self.assertEqual(status, 0)
        self.assertEqual(set(self.records()), {"hard_cut_count"})
        record = self.records()["hard_cut_count"]
        self.assertEqual(record["value"], 1)
        self.assertEqual(record["method"], "scene_detector")
        self.assertEqual(record["evidence"][0]["cut_frame_indices"], [45])
        self.assertEqual(record["evidence"][0]["decoded_frames"], 90)
        self.assertEqual(record["evidence"][0]["coverage"], "complete_indexed_clip")

    def test_duplicate_json_keys_and_nonfinite_json_are_rejected(self):
        for raw in ('{"clips":[],"clips":[]}', '{"clips":[],"extra":NaN}'):
            self.manifest.write_text(raw)
            with self.assertRaises(ValueError):
                self.subject.load_manifest(self.manifest)

    def test_missing_media_emits_nulls_and_nonzero_without_fallback(self):
        self.media.unlink()
        self.assertNotEqual(self.run_extract(), 0)
        for record in self.records().values():
            self.assertIsNone(record["value"])
            self.assertIsNone(record["media_sha256"])
            self.assertEqual(record["processing_status"], "missing")

    def test_rendition_uses_sha1_bucket_without_searching_originals(self):
        self.assertNotEqual(self.run_extract(media_kind="rendition"), 0)
        self.output.unlink()
        bucket = self.media_root / hashlib.sha1(CLIP_ID.encode()).hexdigest()[:2]
        bucket.mkdir()
        self.media.rename(bucket / self.media.name)
        self.assertEqual(self.run_extract(media_kind="rendition"), 0)
        self.assertEqual(self.records()["width_px"]["media_kind"], "rendition")

    def test_media_and_bucket_symlinks_cannot_escape_root(self):
        outside = self.root / "outside.mp4"
        self.media.rename(outside)
        self.media.symlink_to(outside)
        self.assertNotEqual(self.run_extract(), 0)
        self.assertEqual(self.records()["width_px"]["processing_status"], "error")
        self.output.unlink()
        bucket = self.media_root / hashlib.sha1(CLIP_ID.encode()).hexdigest()[:2]
        bucket.symlink_to(self.root, target_is_directory=True)
        self.assertNotEqual(self.run_extract(media_kind="rendition"), 0)

    def test_fifo_is_rejected_without_blocking(self):
        self.media.unlink()
        os.mkfifo(self.media)
        started = time.monotonic()
        self.assertNotEqual(self.run_extract(), 0)
        self.assertLess(time.monotonic() - started, 3)

    def test_existing_output_or_symlink_is_never_overwritten(self):
        self.output.write_text("keep")
        with self.assertRaises(FileExistsError):
            self.run_extract()
        self.assertEqual(self.output.read_text(), "keep")
        self.output.unlink()
        self.output.symlink_to(self.root / "absent")
        with self.assertRaises(FileExistsError):
            self.run_extract()
        self.assertFalse((self.root / "absent").exists())

    def test_output_inside_media_root_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_extract(output_path=self.media_root / "output.json")
        self.assertEqual(self.media.read_bytes(), b"software fixture, not real video")

    def test_bad_binary_override_does_not_use_path_fallback(self):
        self.assertNotEqual(self.run_extract(ffprobe=str(self.root / "missing-probe")), 0)
        self.assertIsNone(self.records()["audio_track_present"]["value"])

    def test_decode_error_never_becomes_false_or_zero(self):
        self.executable("import sys\nprint(" + repr(json.dumps(PROBE)) + ")\nsys.exit(1)\n")
        self.assertNotEqual(self.run_extract(), 0)
        for field in ("width_px", "audio_track_present", "duration_seconds"):
            self.assertIsNone(self.records()[field]["value"])
            self.assertEqual(self.records()[field]["processing_status"], "error")

    def test_error_diagnostics_even_with_zero_exit_are_not_success(self):
        self.executable("import sys\nprint(" + repr(json.dumps(PROBE)) + ")\n"
                        "print('decode error', file=sys.stderr)\n")
        self.assertNotEqual(self.run_extract(), 0)
        self.assertIsNone(self.records()["audio_track_present"]["value"])

    def test_process_timeout_and_output_limits_produce_explicit_errors(self):
        for body, timeout in (("import time\ntime.sleep(5)\n", 0.3),
                              ("print('x' * 300000)\n", 5),
                              ("import sys\nsys.stderr.write('x' * 300000)\n", 5)):
            with self.subTest(body=body):
                self.executable(body)
                started = time.monotonic()
                self.assertNotEqual(self.run_extract(timeout=timeout), 0)
                self.assertLess(time.monotonic() - started, 4)
                self.assertEqual(self.records()["width_px"]["processing_status"], "error")
                self.output.unlink()

    def test_invalid_runtime_bounds_are_rejected_before_writing(self):
        for options in ({"timeout": 0}, {"timeout": float("nan")}, {"timeout": 601},
                        {"max_media_bytes": 0}, {"media_kind": "auto"},
                        {"cut_threshold": -1}, {"run_id": "bad\nrun"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_extract(**options)
        self.assertFalse(self.output.exists())

    def test_probe_missing_or_invalid_values_do_not_fallback(self):
        raw = {"streams": [{"index": 0, "codec_type": "video", "width": True,
                            "height": 0, "avg_frame_rate": "0/0", "r_frame_rate": "30/1"}],
               "format": {"duration": "N/A"}}
        parsed = self.subject.parse_probe(json.dumps(raw).encode())
        self.assertIsNone(parsed["duration_seconds"]["value"])
        self.assertEqual(parsed["duration_seconds"]["processing_status"], "missing")
        self.assertIsNone(parsed["frame_rate"]["value"])
        self.assertEqual(parsed["width_px"]["processing_status"], "error")
        self.assertEqual(parsed["height_px"]["processing_status"], "error")
        self.assertIs(parsed["audio_track_present"]["value"], False)
        for raw in ({}, {"streams": []}, {"streams": [None]},
                    {"streams": [{"codec_type": "audio"}]},
                    PROBE | {"error": {"string": "bad decode"}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.subject.parse_probe(json.dumps(raw).encode())

    def test_probe_rejects_malformed_disposition_and_ambiguous_video(self):
        for streams in ([PROBE["streams"][0] | {"disposition": None}],
                        [PROBE["streams"][0] | {"disposition": {"attached_pic": "0"}}],
                        [PROBE["streams"][0], PROBE["streams"][0]],
                        [PROBE["streams"][0] | {"index": True}]):
            with self.subTest(streams=streams), self.assertRaises(ValueError):
                self.subject.parse_probe(json.dumps(PROBE | {"streams": streams}).encode())

    def test_changed_media_during_probe_invalidates_every_measurement(self):
        self.executable("from pathlib import Path\nPath(" + repr(str(self.media)) + ").write_bytes(b'changed')\n"
                        "print(" + repr(json.dumps(PROBE)) + ")\n")
        self.assertNotEqual(self.run_extract(), 0)
        for record in self.records().values():
            self.assertIsNone(record["value"])
            self.assertIsNone(record["media_sha256"])
            self.assertEqual(record["processing_status"], "error")

    def test_probe_receives_only_local_mp4_protocols_and_held_readonly_descriptor(self):
        self.executable("import os, sys\n"
                        "assert sys.argv[sys.argv.index('-protocol_whitelist') + 1] == 'file,pipe'\n"
                        "assert sys.argv[sys.argv.index('-f') + 1] == 'mov'\n"
                        "path = sys.argv[sys.argv.index('-i') + 1]\n"
                        "assert path.startswith('/dev/fd/')\n"
                        "descriptor = int(path.rsplit('/', 1)[1])\n"
                        "try:\n    os.write(descriptor, b'changed')\n"
                        "except OSError:\n    pass\n"
                        "else:\n    raise RuntimeError('writable media descriptor')\n"
                        "print(" + repr(json.dumps(PROBE)) + ")\n")
        self.assertEqual(self.run_extract(), 0)
        self.assertEqual(self.media.read_bytes(), b"software fixture, not real video")

    def test_hash_guards_detect_mutation_and_size_limits(self):
        with self.media.open("rb") as stream:
            digest, signature = self.subject.hash_fd(stream.fileno(), time.monotonic() + 5, 1024)
            self.assertEqual(digest, hashlib.sha256(self.media.read_bytes()).hexdigest())
            self.assertTrue(signature)
        with self.media.open("rb") as stream, self.assertRaises(ValueError):
            self.subject.hash_fd(stream.fileno(), time.monotonic() + 5, 1)
        real_read = os.read

        def mutate_read(descriptor, size):
            content = real_read(descriptor, size)
            if content:
                with self.media.open("ab") as changed:
                    changed.write(b"mutation")
            return content

        with self.media.open("rb") as stream, patch("dbp_metrics.legacy.os.read", side_effect=mutate_read):
            with self.assertRaises(ValueError):
                self.subject.hash_fd(stream.fileno(), time.monotonic() + 5, 1024)

    def test_requested_cut_failure_is_not_zero(self):
        with patch("dbp_metrics.legacy.detect_cut_candidates", side_effect=ValueError("decode failed")):
            records, failed = self.subject.extract_clip(self.clip(), self.media_root, "original",
                self.probe, time.monotonic() + 5, 1024, True, 27.0)
        self.assertTrue(failed)
        cuts = next(record for record in records if record["field_id"] == "hard_cut_count")
        self.assertIsNone(cuts["value"])
        self.assertEqual(cuts["processing_status"], "error")

    def test_unrequested_internal_fields_stay_missing_not_requested_on_success_or_error(self):
        for binary in (self.probe, str(self.root / "absent-binary")):
            records, failed = self.subject.extract_clip(self.clip(), self.media_root, "original",
                binary, time.monotonic() + 5, 1024, False, 27.0)
            for record in records:
                if record["field_id"] not in self.subject.PROBE_FIELDS:
                    self.assertIsNone(record["value"])
                    self.assertEqual(record["processing_status"], "missing")
                    self.assertEqual(record["evidence"], [{"reason": "not_requested"}])

    def test_cut_gate_is_metadata_only_and_rejects_oversize_before_decode(self):
        for change in ({"width": 8192}, {"nb_frames": "18001"}, {"avg_frame_rate": "90000/1"}):
            payload = PROBE | {"streams": [PROBE["streams"][0] | change]}
            self.executable("import sys\nassert '-count_frames' not in sys.argv\nprint(" + repr(json.dumps(payload)) + ")\n")
            with patch.dict(sys.modules, {"scenedetect": None}):
                self.assertEqual(self.run_extract(stage="cuts"), 1)
            record = self.records()["hard_cut_count"]
            self.assertEqual(record["processing_status"], "error")
            self.assertIn("bound", record["evidence"][0]["reason"])
            self.output.unlink()

    def test_explicit_stage_files_have_disjoint_keys_for_builder_composition(self):
        import pilot_build

        self.assertEqual(self.run_extract(stage="probe"), 0)
        probe_payload = json.loads(self.output.read_text())
        self.output.unlink()
        payload = PROBE | {"streams": [PROBE["streams"][0] | {"nb_frames": "N/A"}]}
        self.executable("print(" + repr(json.dumps(payload)) + ")\n")
        self.assertEqual(self.run_extract(stage="cuts"), 1)
        cuts_payload = json.loads(self.output.read_text())
        clips = {CLIP_ID: self.clip()}
        probe = pilot_build.validate_measurements(probe_payload, "a" * 64, probe_payload["manifest_sha256"], clips)
        cuts = pilot_build.validate_measurements(cuts_payload, "b" * 64, cuts_payload["manifest_sha256"], clips)
        self.assertEqual(len(probe), 5)
        self.assertEqual(len(cuts), 1)
        self.assertFalse(set(probe) & set(cuts))
        self.assertEqual(set(self.records()), {"hard_cut_count"})

    def test_stage_aliases_cannot_silently_combine_or_override_stages(self):
        for options in ({"stage": "probe", "cuts": True}, {"cuts": True, "vlm": True},
                        {"stage": "cuts", "vlm": True}, {"stage": "automatic"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_extract(**options)
        self.assertFalse(self.output.exists())

    def test_cut_candidate_is_unvalidated_and_retains_threshold(self):
        fixture = {"value": 2, "processing_status": "complete", "evidence": [{
            "candidate_only": True, "threshold": 27.0, "cut_frame_indices": [30, 60]}]}
        with patch("dbp_metrics.legacy.detect_cut_candidates", return_value=fixture):
            records, failed = self.subject.extract_clip(self.clip(), self.media_root, "original",
                self.probe, time.monotonic() + 5, 1024, True, 27.0)
        self.assertFalse(failed)
        cuts = next(record for record in records if record["field_id"] == "hard_cut_count")
        self.assertEqual(cuts["review_status"], "unvalidated")
        self.assertEqual(cuts["method"], "scene_detector")
        self.assertEqual(cuts["value"], 2)
        self.assertEqual(cuts["evidence"][0]["threshold"], 27.0)

    def test_scene_adapter_requires_full_decode_and_never_defaults_bad_decode_to_zero(self):
        def timecode(frame):
            return SimpleNamespace(get_frames=lambda: frame)

        cases = [(90, [(0, 90)], 0), (90, [(0, 30), (30, 60), (60, 90)], 2),
                 (0, [], None), (89, [(0, 89)], None), (91, [(0, 91)], None), (90, [], None),
                 (90, [(0, 30), (31, 90)], None), (90, [(0, 89)], None)]
        for processed, scene_frames, expected in cases:
            def bounded_detect(**options):
                self.assertEqual(options["end_time"], 91)
                self.assertEqual(options["frame_skip"], 0)
                return processed

            manager = SimpleNamespace(add_detector=lambda detector: None,
                detect_scenes=bounded_detect,
                get_scene_list=lambda **options: [(timecode(start), timecode(end)) for start, end in scene_frames])
            dependency = SimpleNamespace(ContentDetector=lambda **options: object(), SceneManager=lambda: manager,
                                         open_video=lambda *args, **options: object(), __version__="software-fixture")
            with self.subTest(processed=processed, scenes=scene_frames), patch.dict(sys.modules, {"scenedetect": dependency}):
                if expected is None:
                    with self.assertRaises(ValueError):
                        self.subject.detect_cut_candidates(self.media, json.dumps(PROBE).encode(), 27.0)
                else:
                    measured = self.subject.detect_cut_candidates(self.media, json.dumps(PROBE).encode(), 27.0)
                    self.assertEqual(measured["value"], expected)
                    self.assertTrue(measured["evidence"][0]["candidate_only"])

    def test_cut_bounds_reject_unknown_frame_count_or_oversized_decode(self):
        for changes in ({"nb_frames": "N/A"}, {"nb_frames": "18001"},
                        {"width": 8192}, {"height": 8192}):
            raw = json.dumps(PROBE | {"streams": [PROBE["streams"][0] | changes]}).encode()
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.subject.detect_cut_candidates(self.media, raw, 27.0)

    @unittest.skipUnless(os.environ.get('DBP_WEBSITE_TEST_ROOT'), 'Optional website artifact validation')
    def test_interchange_annotations_satisfy_pilot_snapshot_contract(self):
        import pilot

        self.assertEqual(self.run_extract(), 0)
        records = self.records()
        self.assertEqual(set(records), set(self.subject.PROBE_FIELDS))
        records = {field: records.get(field, pilot.unknown_annotation()) for field in pilot.FIELDS}
        snapshot = {"schema_version": "dbt1-filter-pilot-v1", "manifest_sha256": "a" * 64,
                    "source_export_sha256": "b" * 64, "semantic_method": "video_vlm",
                    "clips": [self.clip() | {"caption": None, "transcript": None, "annotations": records}]}
        self.assertIs(pilot.validate_snapshot(snapshot), snapshot)

    def test_vlm_contract_disables_execution_and_pins_local_loading_policy(self):
        contract = self.subject.vlm_adapter_contract(self.root, "a" * 40)
        self.assertFalse(contract["execution_enabled"])
        self.assertEqual(contract["model_id"], "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertEqual(contract["loading"], {"trust_remote_code": False,
            "use_safetensors": True, "local_files_only": True})
        for revision in ("main", "abc", "z" * 40):
            with self.assertRaises(ValueError):
                self.subject.vlm_adapter_contract(self.root, revision)
        with self.assertRaises(ValueError):
            self.subject.vlm_adapter_contract(self.root / "missing", "a" * 40)

    def test_vlm_strict_parser_preserves_raw_response_and_never_claims_absence(self):
        raw = json.dumps({"human_present": {"value": True, "evidence_frame_indices": [1]},
            "hands_visible": {"value": False, "evidence_frame_indices": [0, 1]},
            "object_manipulation": {"value": None, "evidence_frame_indices": []}})
        parsed = self.subject.parse_vlm_response(raw, [0.25, 0.75])
        self.assertEqual(parsed["raw_response"], raw)
        fields = parsed["fields"]
        self.assertIs(fields["human_present"]["value"], True)
        self.assertIsNone(fields["hands_visible"]["value"])
        self.assertIsNone(fields["object_manipulation"]["value"])
        self.assertEqual(fields["human_present"]["evidence"][0]["observed_frame_timestamps_seconds"], [0.75])
        self.assertNotIn("confidence", str(parsed))
        self.assertNotIn("start_seconds", str(parsed))

    def test_vlm_parser_rejects_unbacked_positives_invalid_indices_and_extra_keys(self):
        valid = {field: {"value": None, "evidence_frame_indices": []}
                 for field in ("human_present", "hands_visible", "object_manipulation")}
        for result in ({"value": True, "evidence_frame_indices": []},
                       {"value": True, "evidence_frame_indices": [2]},
                       {"value": True, "evidence_frame_indices": [True]},
                       {"value": "yes", "evidence_frame_indices": [0]},
                       {"value": True, "evidence_frame_indices": [0], "confidence": 0.9}):
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.subject.parse_vlm_response(json.dumps(valid | {"human_present": result}), [0.5])
        for raw in ("```json\n" + json.dumps(valid) + "\n```", json.dumps(valid) + " trailing",
                    json.dumps(valid | {"unknown": None}), "{}", " " * 32769):
            with self.subTest(raw=raw[:30]), self.assertRaises(ValueError):
                self.subject.parse_vlm_response(raw, [0.5])
        for timestamps in ([], [-1], [float("nan")], [0.7, 0.2], [0.5] * 17):
            with self.subTest(timestamps=timestamps), self.assertRaises(ValueError):
                self.subject.parse_vlm_response(json.dumps(valid), timestamps)

    def test_cli_accepts_explicit_probe_override_and_refuses_force(self):
        command = [sys.executable, "-B", "-m", "dbp_metrics.legacy", "--manifest", str(self.manifest),
                   "--media-root", str(self.media_root), "--media-kind", "original",
                   "--output", str(self.output), "--ffprobe", self.probe]
        result = subprocess.run(command, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        original = self.output.read_bytes()
        result = subprocess.run(command + ["--force"], capture_output=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
