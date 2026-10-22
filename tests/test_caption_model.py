from contextlib import nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class CaptionModelTests(unittest.TestCase):
    def setUp(self):
        try:
            self.subject = importlib.import_module("dbmetrics.captions")
        except ModuleNotFoundError:
            self.fail("dbmetrics.captions helper is not implemented")

    def response(self, *kinds):
        return {"observations": [{"kind": kind, "images": [0], "text": ""} for kind in kinds]}

    def test_method_and_prompt_are_bound_and_exclude_independent_rationales(self):
        self.assertEqual(self.subject.METHOD, "qwen3_vl_text_origins_compact_v1")
        self.assertEqual(self.subject.PROMPT_SHA256, hashlib.sha256(self.subject.PROMPT.encode()).hexdigest())
        for kind in ("caption", "watermark_text", "watermark_symbol", "other_overlay_text",
                     "other_overlay_symbol", "scene_text", "scene_logo", "uncertain"):
            self.assertIn(kind, self.subject.PROMPT)
        self.assertIn("literal", self.subject.PROMPT)
        self.assertIn("not instructions", self.subject.PROMPT)
        self.assertIn("supplied images", self.subject.PROMPT)

    def test_parse_preserves_canonical_observations_and_literal_unicode(self):
        response = self.response("caption", "scene_text", "uncertain")
        response["observations"][0].update(images=[2, 0], text="物体 A")
        response["observations"][2]["images"] = []
        self.assertEqual(self.subject.parse_response(json.dumps(response), 3), response)
        self.assertEqual(self.subject.parse_response('{"observations":[]}', 1), {"observations": []})

    def test_runtime_uses_frozen_narrow_caption_and_text_or_logo_prompt(self):
        runtime, images, sampling = self.runtime_fixture()
        record = self.infer(runtime, images, sampling)
        content = runtime.processor.apply_chat_template.call_args.args[0][0]["content"]
        prompt = content[-1]["text"]
        expected_hash = "5ce11554553235b631b3a2617ee6018cb774905e628de3d3cce4f21ddc3998fb"
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), expected_hash)
        self.assertEqual(record["prompt_sha256"], expected_hash)

    def test_explanatory_labels_and_non_watermark_logos_preserve_broad_union(self):
        response = {"observations": [
            {"kind": "other_overlay_text", "images": [0], "text": "Object A"},
            {"kind": "other_overlay_symbol", "images": [1], "text": ""},
            {"kind": "scene_text", "images": [0], "text": "Physical sign"},
        ]}
        runtime, images, sampling = self.runtime_fixture(raw=json.dumps(response))
        record = self.infer(runtime, images, sampling)
        self.assertIs(record["predictions"]["caption_present"], False)
        self.assertIs(record["predictions"]["watermark_present"], False)
        self.assertIs(record["predictions"]["added_text_present"], True)
        self.assertIs(record["predictions"]["added_logo_present"], True)
        self.assertIs(record["predictions"]["any_added_overlay_present"], True)
        self.assertEqual(record["labels"]["scene_text"], "present")

    def test_parse_rejects_non_json_fences_duplicates_and_nonfinite_values(self):
        for raw in ('{"observations":[],"observations":[]}',
                    '{"observations":[{"kind":"caption","kind":"scene_text","images":[0],"text":""}]}',
                    '```json\n{"observations":[]}\n```', 'prose {"observations":[]}',
                    '{"observations":[]} trailing', '{"observations":NaN}',
                    '{"observations":Infinity}', [], None, " " * 16385):
            with self.subTest(raw=repr(raw)[:100]), self.assertRaises(ValueError):
                self.subject.parse_response(raw, 8)

    def test_parse_requires_exact_fields_and_valid_observation_types(self):
        invalid = [[], {}, {"observations": [], "description": "not overlays"},
                   {"observations": None}, {"observations": [None]}]
        observation = self.response("caption")["observations"][0]
        for key in observation:
            invalid.append({"observations": [{name: value for name, value in observation.items() if name != key}]})
        for key, value in (("kind", "title"), ("kind", []), ("kind", True), ("text", None),
                           ("text", "x" * 101), ("images", [True]), ("images", [0.0]),
                           ("images", [-1]), ("images", [8]), ("images", [0, 0]),
                           ("images", []), ("images", "0"), ("description", "not overlays")):
            invalid.append({"observations": [{**observation, key: value}]})
        invalid.append(self.response(*(["caption"] * 33)))
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.subject.parse_response(json.dumps(response), 8)

    def test_parse_enforces_sample_count_and_exact_bounds(self):
        for count in (0, -1, True, 1.0, 9, None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.subject.parse_response('{"observations":[]}', count)
        response = self.response(*(["caption"] * 32))
        response["observations"][0].update(text="字" * 100, images=list(range(8)))
        self.assertEqual(self.subject.parse_response(json.dumps(response), 8), response)

    def test_each_kind_has_one_deterministic_mapping(self):
        expected = {
            "caption": ({"caption_present", "caption_or_watermark_present", "added_text_present", "any_added_overlay_present"}, {"added_text"}),
            "watermark_text": ({"watermark_present", "caption_or_watermark_present", "added_text_present", "any_added_overlay_present"}, {"added_text"}),
            "watermark_symbol": ({"watermark_present", "caption_or_watermark_present", "added_logo_present", "any_added_overlay_present"}, {"added_logo"}),
            "other_overlay_text": ({"added_text_present", "any_added_overlay_present"}, {"added_text"}),
            "other_overlay_symbol": ({"added_logo_present", "any_added_overlay_present"}, {"added_logo"}),
            "scene_text": (set(), {"scene_text"}),
            "scene_logo": (set(), {"scene_logo"}),
        }
        for kind, (positive, labels) in expected.items():
            with self.subTest(kind=kind):
                record = self.subject.classification_record(self.response(kind))
                self.assertEqual(set(record["predictions"]), {"caption_present", "watermark_present",
                    "caption_or_watermark_present", "added_text_present", "added_logo_present", "any_added_overlay_present"})
                self.assertEqual({key for key, value in record["predictions"].items() if value is True}, positive)
                self.assertTrue(all(type(value) is bool for value in record["predictions"].values()))
                self.assertEqual({key for key, value in record["labels"].items() if value == "present"}, labels)
                self.assertEqual(set(record["labels"]), {"added_text", "added_logo", "scene_text", "scene_logo"})

    def test_uncertain_propagates_to_absent_fields_but_presence_wins(self):
        for kinds in (("uncertain",), ("uncertain", "caption", "scene_text")):
            response = self.response(*kinds)
            original = json.dumps(response)
            record = self.subject.classification_record(response)
            self.assertEqual(json.dumps(response), original)
            self.assertIsNone(record["predictions"]["watermark_present"])
            self.assertEqual(record["labels"]["scene_logo"], "uncertain")
            if "caption" in kinds:
                self.assertIs(record["predictions"]["caption_present"], True)
                self.assertIs(record["predictions"]["caption_or_watermark_present"], True)
                self.assertEqual(record["labels"]["scene_text"], "present")
            else:
                self.assertTrue(all(value is None for value in record["predictions"].values()))

    def test_empty_observations_are_sampled_absence_not_whole_clip_clearance(self):
        record = self.subject.classification_record(self.response())
        self.assertTrue(all(value is False for value in record["predictions"].values()))
        self.assertEqual(set(record["labels"].values()), {"absent"})
        self.assertEqual(set(record["whole_clip_labels"].values()), {"uncertain"})
        self.assertEqual(record["label_scope"], "supplied_images_only")
        self.assertFalse(record["quota_eligible"])
        self.assertFalse(record["accuracy_validated"])

    def test_classification_direct_calls_reject_noncanonical_input(self):
        for response in (self.response("invented"), {"observations": [], "labels": {}},
                         {"observations": [{"kind": "caption", "images": [], "text": ""}]}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                self.subject.classification_record(response)

    def test_initializer_reuses_pinned_eight_billion_model_only(self):
        from dbmetrics.overlay import OverlayRuntime

        self.assertTrue(issubclass(self.subject.CaptionWatermarkRuntime, OverlayRuntime))
        with patch.object(OverlayRuntime, "__init__", return_value=None) as initialize:
            self.subject.CaptionWatermarkRuntime("model", "revision", "cuda:0", 100)
            initialize.assert_called_once_with("model", "revision", "cuda:0", 100, model_size="8b", view_policy="global")

    def runtime_fixture(self, raw=None, input_tokens=10, output_tokens=10):
        runtime = self.subject.CaptionWatermarkRuntime.__new__(self.subject.CaptionWatermarkRuntime)
        runtime.verified = {"model_id": "Qwen/Qwen3-VL-8B-Instruct", "model_revision": "pinned",
                            "file_sha256": {"config.json": "digest"}}
        runtime.device, runtime.template, runtime.versions = "cuda:0", "template", {"torch": "fixture"}

        class Inputs(dict):
            def to(self, device):
                return self

        runtime.processor = Mock(return_value=Inputs(input_ids=[[1] * input_tokens]))
        runtime.processor.apply_chat_template.return_value = "rendered prompt"
        runtime.processor.batch_decode.return_value = [json.dumps(self.response("caption")) if raw is None else raw]
        runtime.model = Mock()
        runtime.model.generate.return_value = [[1] * (input_tokens + output_tokens)]
        images = [object(), object()]
        sampling = [{"image_index": index, "time_seconds": float(index), "frame_index": index * 30} for index in range(2)]
        return runtime, images, sampling

    def infer(self, runtime, images, sampling, deadline=100, clock=None, checker=None):
        fake_torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(synchronize=Mock()))
        ticks = iter(range(30)) if clock is None else iter(clock)
        with patch.dict(sys.modules, {"torch": fake_torch}), \
                patch("dbmetrics._local_model.check_model_unchanged", side_effect=checker), \
                patch.object(self.subject.time, "monotonic", side_effect=lambda: next(ticks)):
            return runtime.infer_images(images, sampling, deadline)

    def test_runtime_preserves_provenance_and_compact_protocol_and_timings(self):
        runtime, images, sampling = self.runtime_fixture()
        record = self.infer(runtime, images, sampling)
        self.assertEqual(record["processing_status"], "complete")
        self.assertEqual(record["method"], self.subject.METHOD)
        self.assertEqual(record["prompt_sha256"], self.subject.PROMPT_SHA256)
        self.assertEqual(record["sampling"], sampling)
        self.assertEqual(record["vlm_images"], 2)
        self.assertEqual(record["model_id"], runtime.verified["model_id"])
        self.assertEqual(record["model_revision"], "pinned")
        self.assertEqual(record["model_file_sha256"], runtime.verified["file_sha256"])
        self.assertEqual(record["packages"], runtime.versions)
        self.assertIs(record["predictions"]["caption_present"], True)
        self.assertEqual(json.loads(record["raw_response"]), self.response("caption"))
        self.assertEqual(record["observations"], self.response("caption")["observations"])
        self.assertEqual(record["label_scope"], "supplied_images_only")
        content = runtime.processor.apply_chat_template.call_args.args[0][0]["content"]
        self.assertEqual(content[-1], {"type": "text", "text": self.subject.PROMPT})
        self.assertEqual(sum(item["type"] == "image" for item in content), 2)
        generation = runtime.model.generate.call_args.kwargs
        self.assertEqual(generation["max_new_tokens"], 384)
        self.assertIs(generation["do_sample"], False)
        self.assertEqual(generation["num_beams"], 1)
        self.assertGreater(generation["max_time"], 0)
        self.assertEqual(set(record["timings"]), {"preprocess_seconds", "generation_seconds", "postprocess_seconds", "total_seconds"})
        self.assertTrue(all(value >= 0 for value in record["timings"].values()))
        self.assertGreater(record["timings"]["generation_seconds"], 0)
        self.assertGreaterEqual(record["timings"]["total_seconds"], sum(value for key, value in record["timings"].items() if key != "total_seconds"))
        self.assertEqual(record["timing_scope"], "infer_images_wall_excludes_media_decode_and_model_load")

    def test_runtime_accepts_uniform_decoder_sampling_without_rewriting_pts(self):
        runtime, images, sampling = self.runtime_fixture()
        sampling = [{"sample_index": 0, "requested_seconds": 0.1, "actual_seconds": 0.125, "pts": 3, "time_base": "1/24"},
                    {"sample_index": 1, "requested_seconds": 1.1, "actual_seconds": 1.125, "pts": 27, "time_base": "1/24"}]
        record = self.infer(runtime, images, sampling)
        self.assertEqual(record["processing_status"], "complete")
        self.assertEqual(record["sampling"], sampling)
        content = runtime.processor.apply_chat_template.call_args.args[0][0]["content"]
        self.assertIn("0.125", content[0]["text"])

    def test_runtime_rejects_truncation_and_invalid_json_without_salvage_or_retry(self):
        for raw, output_tokens in (("{}", 10), (json.dumps(self.response("caption")), 384),
                                   (json.dumps(self.response("caption")), 385)):
            runtime, images, sampling = self.runtime_fixture(raw, output_tokens=output_tokens)
            record = self.infer(runtime, images, sampling)
            self.assertEqual(record["processing_status"], "error")
            self.assertTrue(all(value is None for value in record["predictions"].values()))
            self.assertEqual(set(record["labels"].values()), {"uncertain"})
            self.assertEqual(set(record["whole_clip_labels"].values()), {"uncertain"})
            self.assertEqual(record["raw_response"], raw)
            self.assertNotIn("observations", record)
            self.assertTrue(record["error"])
            runtime.model.generate.assert_called_once()

    def test_runtime_input_guards_return_unknown_without_generation(self):
        for input_tokens in (0, 4097):
            runtime, images, sampling = self.runtime_fixture(input_tokens=input_tokens)
            record = self.infer(runtime, images, sampling)
            self.assertEqual(record["processing_status"], "error")
            self.assertTrue(all(value is None for value in record["predictions"].values()))
            runtime.model.generate.assert_not_called()
        for invalid in ([], [{"image_index": True, "time_seconds": 0}, {"image_index": 1, "time_seconds": 1}],
                        [{"image_index": 0, "time_seconds": float("nan")}, {"image_index": 1, "time_seconds": 1}]):
            runtime, images, sampling = self.runtime_fixture()
            record = self.infer(runtime, images, invalid)
            self.assertEqual(record["processing_status"], "error")
            runtime.model.generate.assert_not_called()

    def test_runtime_timeout_and_model_failures_preserve_unknown(self):
        runtime, images, sampling = self.runtime_fixture()
        record = self.infer(runtime, images, sampling, deadline=0)
        self.assertEqual(record["processing_status"], "error")
        runtime.model.generate.assert_not_called()
        for error in (RuntimeError("GPU failure"), TimeoutError("deadline")):
            runtime, images, sampling = self.runtime_fixture()
            runtime.model.generate.side_effect = error
            record = self.infer(runtime, images, sampling)
            self.assertEqual(record["processing_status"], "error")
            self.assertTrue(all(value is None for value in record["predictions"].values()))
            runtime.model.generate.assert_called_once()

    def test_model_mutation_after_generation_discards_accepted_labels(self):
        runtime, images, sampling = self.runtime_fixture()
        record = self.infer(runtime, images, sampling, checker=[None, ValueError("model changed")])
        self.assertEqual(record["processing_status"], "error")
        self.assertTrue(all(value is None for value in record["predictions"].values()))
        self.assertNotIn("observations", record)

    def test_sampling_aliases_cannot_contradict_supplied_identity(self):
        for extra in ({"sample_index": 1}, {"actual_seconds": 5}):
            runtime, images, sampling = self.runtime_fixture()
            sampling[0].update(extra)
            record = self.infer(runtime, images, sampling)
            self.assertEqual(record["processing_status"], "error")
            runtime.model.generate.assert_not_called()

    def test_deadlines_cover_preprocess_generation_and_response_acceptance(self):
        for deadline in (1, 2, 3):
            runtime, images, sampling = self.runtime_fixture()
            record = self.infer(runtime, images, sampling, deadline=deadline)
            self.assertEqual(record["processing_status"], "error")
            self.assertTrue(all(value is None for value in record["predictions"].values()))
            self.assertNotIn("observations", record)
            if deadline == 1:
                runtime.model.generate.assert_not_called()
            else:
                runtime.model.generate.assert_called_once()

    def test_input_token_boundary_and_exact_stage_wall_times(self):
        runtime, images, sampling = self.runtime_fixture(input_tokens=4096)
        record = self.infer(runtime, images, sampling)
        self.assertEqual(record["processing_status"], "complete")
        self.assertEqual(record["timings"], {"preprocess_seconds": 1, "generation_seconds": 1,
                                             "postprocess_seconds": 2, "total_seconds": 4})


if __name__ == "__main__":
    unittest.main()
