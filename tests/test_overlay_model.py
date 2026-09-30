from contextlib import nullcontext
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    from PIL import Image
except ImportError:
    Image = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class OverlayModelTests(unittest.TestCase):
    def setUp(self):
        try:
            self.subject = importlib.import_module("dbp_metrics.overlay")
        except ModuleNotFoundError:
            self.fail("dbp_metrics.overlay is not implemented")

    def evidence(self, count=100):
        return {"frames": [{"index": index, "time_seconds": index / 10,
                            "detections": []} for index in range(count)],
                "width": 100, "height": 50}

    def response(self):
        return {field: {"value": "absent", "evidence_frame_indices": [],
                        "description": "Not observed in supplied images."}
                for field in self.subject.FIELDS}

    def test_sampling_is_bounded_deterministic_and_keeps_endpoints(self):
        evidence = self.evidence()
        first = self.subject.select_frames(evidence)
        self.assertEqual(first, self.subject.select_frames(evidence))
        self.assertEqual(len(first), 8)
        self.assertEqual(len({row["index"] for row in first}), 8)
        self.assertEqual(first[0]["index"], 0)
        self.assertEqual(first[-1]["index"], 99)
        self.assertEqual(first, sorted(first, key=lambda row: row["index"]))

    def test_brief_candidate_away_from_uniform_samples_is_included(self):
        evidence = self.evidence()
        evidence["frames"][91]["detections"] = [{
            "polygon": [[1, 1], [20, 1], [20, 8], [1, 8]], "score": 0.7}]
        selected = self.subject.select_frames(evidence)
        self.assertIn(91, [row["index"] for row in selected])

    def test_short_clip_does_not_duplicate_frames(self):
        evidence = self.evidence(3)
        self.assertEqual([row["index"] for row in self.subject.select_frames(evidence)], [0, 1, 2])

    def test_present_requires_valid_sample_evidence(self):
        response = self.response()
        response["added_text"]["value"] = "present"
        with self.assertRaises(ValueError):
            self.subject.parse_response(json.dumps(response), 8)
        response["added_text"]["evidence_frame_indices"] = [8]
        with self.assertRaises(ValueError):
            self.subject.parse_response(json.dumps(response), 8)
        response["added_text"]["evidence_frame_indices"] = [2]
        self.assertEqual(self.subject.parse_response(json.dumps(response), 8)["added_text"]["value"], "present")

    def test_uncertainty_and_independent_coexisting_fields_are_preserved(self):
        response = self.response()
        for field in ("added_text", "scene_text"):
            response[field].update(value="present", evidence_frame_indices=[0])
        response["added_logo"]["value"] = "uncertain"
        parsed = self.subject.parse_response(json.dumps(response), 2)
        self.assertEqual(parsed["added_logo"]["value"], "uncertain")
        self.assertEqual(parsed["scene_text"]["value"], "present")

    def test_rejects_extra_confidence_duplicate_keys_and_boolean_labels(self):
        response = self.response()
        response["confidence"] = 0.95
        with self.assertRaises(ValueError):
            self.subject.parse_response(json.dumps(response), 8)
        response.pop("confidence")
        response["added_text"]["value"] = False
        with self.assertRaises(ValueError):
            self.subject.parse_response(json.dumps(response), 8)
        with self.assertRaises(ValueError):
            self.subject.parse_response('{"added_text":{},"added_text":{}}', 8)

    def test_absence_never_becomes_whole_clip_quota_clearance(self):
        response = self.response()
        response["added_text"].update(value="present", evidence_frame_indices=[0])
        record = self.subject.classification_record(response)
        self.assertEqual(record["labels"]["scene_text"], "absent")
        self.assertEqual(record["whole_clip_labels"]["scene_text"], "uncertain")
        self.assertEqual(record["whole_clip_labels"]["added_text"], "present")
        self.assertFalse(record["quota_eligible"])
        self.assertFalse(record["accuracy_validated"])
        self.assertIsNone(record["calibrated_confidence"])

    def test_accepts_one_json_fence_without_changing_labels(self):
        response = self.response()
        response["added_text"].update(value="uncertain", evidence_frame_indices=[0])
        encoded = json.dumps(response)
        self.assertEqual(self.subject.parse_response("```json\n" + encoded + "\n```", 8), response)
        for raw in ("Explanation\n```json\n" + encoded + "\n```",
                    "```json\n" + encoded + "\n```\nExplanation",
                    "```python\n" + encoded + "\n```"):
            with self.assertRaises(ValueError):
                self.subject.parse_response(raw, 8)

    def test_all_supplied_images_can_be_evidence_but_not_duplicates(self):
        response = self.response()
        response["added_text"].update(value="present", evidence_frame_indices=list(range(8)))
        self.assertEqual(self.subject.parse_response(json.dumps(response), 8), response)
        response["added_text"]["evidence_frame_indices"] = [0, 0]
        with self.assertRaises(ValueError):
            self.subject.parse_response(json.dumps(response), 8)

    def test_prompt_does_not_supply_a_filled_uncertain_answer(self):
        self.assertNotIn('"value":"uncertain"', self.subject.PROMPT)
        self.assertEqual(self.subject.METHOD, "ocr_guided_qwen3_overlay_development_v3")

    def test_output_contract_follows_images_without_filled_labels(self):
        self.assertTrue(hasattr(self.subject, "build_content"))
        content = self.subject.build_content([{"image_index": 0, "time_seconds": 0.0}])
        self.assertEqual(content[-1], {"type": "text", "text": self.subject.PROMPT})
        self.assertEqual(sum(item["type"] == "image" for item in content), 1)
        self.assertIn('"value":"<decision>"', self.subject.PROMPT)

    def test_model_identity_is_explicitly_pinned_not_guessed(self):
        self.assertTrue(hasattr(self.subject, "model_identity"))
        self.assertEqual(self.subject.model_identity("2b")[0], "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(self.subject.model_identity("8b"),
                         ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"))
        with self.assertRaises(ValueError):
            self.subject.model_identity("32b")

    def test_view_policy_configuration_keeps_v3_default_and_explicitly_versions_details(self):
        self.assertTrue(hasattr(self.subject, "method_config"), "Missing opt-in view policy configuration")
        original = self.subject.method_config()
        self.assertEqual(original["method"], "ocr_guided_qwen3_overlay_development_v3")
        self.assertEqual(original["prompt_sha256"], "012f080da28451410c581dfe074e780ae16eeb5f8a213e8323c2c853d49a9c37")
        self.assertEqual(original["prompt"], self.subject.PROMPT)
        self.assertEqual(original["view_policy"], "global")
        details = self.subject.method_config("global-and-corners")
        self.assertEqual(details["method"], "ocr_guided_qwen3_overlay_development_v4")
        self.assertEqual(details["prompt_version"], "v4")
        self.assertEqual(details["prompt_sha256"], hashlib.sha256(details["prompt"].encode()).hexdigest())
        self.assertNotEqual(details["prompt_sha256"], original["prompt_sha256"])
        self.assertIn(self.subject.PROMPT, details["prompt"])
        for policy in ("corners", "", None):
            with self.assertRaises(ValueError):
                self.subject.method_config(policy)

    def decoded_fixture(self, count=8, width=10, height=6):
        decoded, sampling = [], []
        for image_index in range(count):
            frame_index = image_index * 5
            pixels = bytes(channel for vertical in range(height) for horizontal in range(width)
                           for channel in (horizontal, vertical, image_index))
            decoded.append(SimpleNamespace(index=frame_index, relative_seconds=frame_index / 10,
                                            width=width, height=height, rgb=pixels))
            sampling.append({"image_index": image_index, "frame_index": frame_index,
                             "time_seconds": frame_index / 10, "ocr_candidates": image_index % 3})
        return decoded, sampling

    @unittest.skipUnless(Image, "Native crop tests require the existing Pillow runtime")
    def test_details_are_four_native_quadrants_of_one_fixed_central_frame(self):
        self.assertTrue(hasattr(self.subject, "prepare_views"), "Missing deterministic detail views")
        decoded, sampling = self.decoded_fixture()
        original = json.dumps(sampling)
        images, rows = self.subject.prepare_views(decoded, sampling, "global-and-corners")
        self.assertEqual(len(images), 12)
        self.assertEqual([row["image_index"] for row in rows], list(range(12)))
        self.assertEqual(json.dumps(sampling), original)
        for image_index in range(8):
            self.assertEqual(images[image_index].tobytes(), decoded[image_index].rgb)
            self.assertEqual(rows[image_index]["crop_bbox"], [0, 0, 10, 6])
            self.assertEqual(rows[image_index]["view"], "global")
        expected = [([0, 0, 5, 3], "corner-top-left", (0, 0, 4), (4, 2, 4)),
                    ([5, 0, 10, 3], "corner-top-right", (5, 0, 4), (9, 2, 4)),
                    ([0, 3, 5, 6], "corner-bottom-left", (0, 3, 4), (4, 5, 4)),
                    ([5, 3, 10, 6], "corner-bottom-right", (5, 3, 4), (9, 5, 4))]
        for image_index, (bbox, view, first_pixel, last_pixel) in enumerate(expected, 8):
            row, image = rows[image_index], images[image_index]
            self.assertEqual(row["crop_bbox"], bbox)
            self.assertEqual(row["view"], view)
            self.assertEqual((row["frame_index"], row["time_seconds"], row["parent_image_index"]), (20, 2.0, 4))
            self.assertEqual(image.size, (5, 3))
            self.assertEqual(image.getpixel((0, 0)), first_pixel)
            self.assertEqual(image.getpixel((4, 2)), last_pixel)
        repeated_images, repeated_rows = self.subject.prepare_views(decoded, sampling, "global-and-corners")
        self.assertEqual(rows, repeated_rows)
        self.assertEqual([image.tobytes() for image in images], [image.tobytes() for image in repeated_images])

    @unittest.skipUnless(Image, "Native crop tests require the existing Pillow runtime")
    def test_odd_nonsquare_quadrants_preserve_all_pixels_without_padding_or_upsampling(self):
        self.assertTrue(hasattr(self.subject, "prepare_views"), "Missing deterministic detail views")
        for width, height, expected_boxes in (
                (9, 5, [[0, 0, 4, 2], [4, 0, 9, 2], [0, 2, 4, 5], [4, 2, 9, 5]]),
                (5, 9, [[0, 0, 2, 4], [2, 0, 5, 4], [0, 4, 2, 9], [2, 4, 5, 9]])):
            decoded, sampling = self.decoded_fixture(width=width, height=height)
            images, rows = self.subject.prepare_views(decoded, sampling, "global-and-corners")
            self.assertEqual([row["crop_bbox"] for row in rows[8:]], expected_boxes)
            reconstruction = Image.new("RGB", (width, height))
            for image, bbox in zip(images[8:], expected_boxes):
                self.assertEqual(image.size, (bbox[2] - bbox[0], bbox[3] - bbox[1]))
                reconstruction.paste(image, tuple(bbox[:2]))
            self.assertEqual(reconstruction.tobytes(), decoded[4].rgb)

    @unittest.skipUnless(Image, "Native crop tests require the existing Pillow runtime")
    def test_global_default_and_short_clip_metadata_remain_bounded(self):
        self.assertTrue(hasattr(self.subject, "prepare_views"), "Missing deterministic detail views")
        for count in (1, 3, 8):
            decoded, sampling = self.decoded_fixture(count=count)
            images, rows = self.subject.prepare_views(decoded, sampling)
            self.assertEqual(rows, sampling)
            self.assertEqual(len(images), count)
            self.assertEqual([image.tobytes() for image in images], [frame.rgb for frame in decoded])
            images, rows = self.subject.prepare_views(decoded, sampling, "global-and-corners")
            self.assertEqual(len(images), count + 4)
            self.assertEqual([row["image_index"] for row in rows], list(range(count + 4)))
            self.assertEqual({row["parent_image_index"] for row in rows[count:]}, {count // 2})

    @unittest.skipUnless(Image, "Native crop tests require the existing Pillow runtime")
    def test_detail_views_reject_invalid_bounds_identity_and_policy(self):
        self.assertTrue(hasattr(self.subject, "prepare_views"), "Missing deterministic detail views")
        for count, width, height in ((0, 10, 6), (9, 10, 6), (8, 1, 6), (8, 10, 1)):
            decoded, sampling = self.decoded_fixture(count=count, width=width, height=height)
            with self.assertRaises(ValueError):
                self.subject.prepare_views(decoded, sampling, "global-and-corners")
        decoded, sampling = self.decoded_fixture()
        for key, value in (("image_index", 99), ("frame_index", 99), ("time_seconds", 99)):
            changed = [dict(row) for row in sampling]
            changed[4][key] = value
            with self.assertRaises(ValueError):
                self.subject.prepare_views(decoded, changed, "global-and-corners")
        with self.assertRaises(ValueError):
            self.subject.prepare_views(decoded, sampling[:-1], "global-and-corners")
        with self.assertRaises(ValueError):
            self.subject.prepare_views(decoded, sampling, "other")

    @unittest.skipUnless(Image, "Native crop tests require the existing Pillow runtime")
    def test_prompt_pairs_every_detail_with_original_frame_and_global_context(self):
        self.assertTrue(hasattr(self.subject, "prepare_views"), "Missing deterministic detail views")
        decoded, sampling = self.decoded_fixture()
        unused_images, rows = self.subject.prepare_views(decoded, sampling, "global-and-corners")
        content = self.subject.build_content(rows, view_policy="global-and-corners")
        self.assertEqual(sum(item["type"] == "image" for item in content), 12)
        for image_index, row in enumerate(rows):
            label = content[image_index * 2]["text"]
            self.assertIn(f"Image {image_index} ", label)
            self.assertIn(row["view"], label)
            self.assertIn(str(row["crop_bbox"]), label)
            self.assertIn(f"frame {row['frame_index']}", label)
            self.assertEqual(content[image_index * 2 + 1], {"type": "image"})
            if image_index >= 8:
                self.assertIn("global Image 4", label)
        self.assertEqual(content[-1]["text"], self.subject.method_config("global-and-corners")["prompt"])

    def test_detail_evidence_indices_preserve_schema_and_whole_clip_uncertainty(self):
        response = self.response()
        response["added_logo"].update(value="present", evidence_frame_indices=[0, 8, 11])
        parsed = self.subject.parse_response(json.dumps(response), 12)
        self.assertEqual(parsed, response)
        record = self.subject.classification_record(parsed)
        self.assertEqual(record["whole_clip_labels"]["scene_logo"], "uncertain")
        self.assertFalse(record["quota_eligible"])
        for indices in ([12], [-1], [11, 11], [True]):
            response["added_logo"]["evidence_frame_indices"] = indices
            with self.assertRaises(ValueError):
                self.subject.parse_response(json.dumps(response), 12)
        response["added_logo"].update(value="absent", evidence_frame_indices=[11])
        with self.assertRaises(ValueError):
            self.subject.parse_response(json.dumps(response), 12)

    def test_invalid_runtime_view_policy_is_rejected_before_model_loading(self):
        with self.assertRaises(ValueError):
            self.subject.OverlayRuntime("unused", "unused", "cuda:0", 0, view_policy="other")

    def inference_fixture(self, policy, input_tokens=100, output_tokens=2, raw=None):
        runtime = self.subject.OverlayRuntime.__new__(self.subject.OverlayRuntime)
        runtime.view_policy = policy
        runtime.verified = {"model_id": "pinned-test-model", "model_revision": "frozen",
                            "file_sha256": {"weights": "test-digest"}}
        runtime.device, runtime.template, runtime.versions = "cuda:0", "template", {"test": "fixture"}
        inputs = {"input_ids": [[1] * input_tokens]}
        class Inputs(dict):
            def to(self, device):
                self.device = device
                return self
        runtime.processor = mock.Mock(return_value=Inputs(inputs))
        runtime.processor.apply_chat_template.return_value = "rendered prompt"
        runtime.processor.batch_decode.return_value = [json.dumps(self.response()) if raw is None else raw]
        runtime.model = mock.Mock()
        runtime.model.generate.return_value = [[1] * (input_tokens + output_tokens)]
        decoded, sampling = self.decoded_fixture()
        pilot = SimpleNamespace(get=lambda clip_id: {"evidence": {"inferred_frames": 36}})
        return runtime, pilot, decoded, sampling

    @unittest.skipUnless(Image, "Native crop tests require the existing Pillow runtime")
    def test_runtime_routes_twelve_views_and_versions_evidence_without_changing_v3(self):
        for policy, count, method in (("global", 8, "ocr_guided_qwen3_overlay_development_v3"),
                                      ("global-and-corners", 12, "ocr_guided_qwen3_overlay_development_v4")):
            response = self.response()
            response["added_logo"].update(value="present", evidence_frame_indices=[count - 1])
            runtime, pilot, decoded, sampling = self.inference_fixture(policy, raw=json.dumps(response))
            with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}), \
                    mock.patch("dbp_metrics._local_model.check_model_unchanged"), \
                    mock.patch.object(self.subject, "extract_evidence_frames", return_value=(decoded, sampling)), \
                    mock.patch.object(self.subject.time, "monotonic", return_value=1):
                record = runtime.infer(pilot, "synthetic-development", "unused", 10)
            self.assertEqual(record["processing_status"], "complete")
            self.assertEqual(record["method"], method)
            self.assertEqual(record["vlm_images"], count)
            self.assertEqual(record["full_decoded_frames"], 36)
            self.assertEqual(record["evidence"], response)
            self.assertEqual(record["whole_clip_labels"]["scene_logo"], "uncertain")
            self.assertFalse(record["quota_eligible"])
            images = runtime.processor.call_args.kwargs["images"]
            content = runtime.processor.apply_chat_template.call_args.args[0][0]["content"]
            self.assertEqual(len(images), count)
            self.assertEqual(sum(item["type"] == "image" for item in content), count)
            self.assertEqual(record["prompt_sha256"], hashlib.sha256(content[-1]["text"].encode()).hexdigest())
            self.assertEqual(runtime.model.generate.call_args.kwargs["max_new_tokens"], 650)
            self.assertFalse(runtime.model.generate.call_args.kwargs["do_sample"])
            if policy == "global":
                self.assertEqual(record["sampling"], sampling)
                self.assertEqual(content[-1]["text"], self.subject.PROMPT)
            else:
                self.assertEqual(record["view_policy"], policy)
                self.assertEqual(record["prompt_version"], "v4")
                self.assertEqual(record["sampling"][11]["parent_image_index"], 4)

    @unittest.skipUnless(Image, "Native crop tests require the existing Pillow runtime")
    def test_detail_runtime_keeps_input_and_output_token_guards(self):
        for input_tokens, output_tokens in ((4096, 2), (4097, 2), (100, 650)):
            runtime, pilot, decoded, sampling = self.inference_fixture("global-and-corners", input_tokens, output_tokens)
            with mock.patch.dict(sys.modules, {"torch": SimpleNamespace(inference_mode=nullcontext)}), \
                    mock.patch("dbp_metrics._local_model.check_model_unchanged"), \
                    mock.patch.object(self.subject, "extract_evidence_frames", return_value=(decoded, sampling)), \
                    mock.patch.object(self.subject.time, "monotonic", return_value=1):
                if input_tokens > 4096:
                    with self.assertRaisesRegex(ValueError, "Input token budget exceeded"):
                        runtime.infer(pilot, "synthetic-development", "unused", 10)
                    runtime.model.generate.assert_not_called()
                    continue
                record = runtime.infer(pilot, "synthetic-development", "unused", 10)
            self.assertEqual(record["processing_status"], "error" if output_tokens == 650 else "complete")
            if output_tokens == 650:
                self.assertEqual(set(record["labels"].values()), {"uncertain"})
                self.assertFalse(record["quota_eligible"])


if __name__ == "__main__":
    unittest.main()
