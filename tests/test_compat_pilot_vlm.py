"""Local VLM control tests; all weights, frames and model responses are fixtures."""

from contextlib import nullcontext
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import dbmetrics.legacy


REVISION = "a" * 40
CONFIG = {"model_type": "qwen2_5_vl", "architectures": ["Qwen2_5_VLForConditionalGeneration"],
          "hidden_size": 2048, "intermediate_size": 11008, "num_hidden_layers": 36,
          "num_attention_heads": 16, "num_key_value_heads": 2, "vocab_size": 151936,
          "vision_config": {"depth": 32, "hidden_size": 1280, "intermediate_size": 3420,
                            "num_heads": 16, "out_hidden_size": 2048, "patch_size": 14,
                            "spatial_merge_size": 2, "temporal_patch_size": 2}}
PREPROCESSOR = {"processor_class": "Qwen2_5_VLProcessor", "image_processor_type": "Qwen2VLImageProcessor",
                "patch_size": 14, "temporal_patch_size": 2, "merge_size": 2}
QWEN3_ID = "Qwen/Qwen3-VL-2B-Instruct"
QWEN3_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
QWEN3_CONFIG = {"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"],
               "text_config": {"hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 28,
                               "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 151936,
                               "head_dim": 128},
               "vision_config": {"depth": 24, "hidden_size": 1024, "intermediate_size": 4096,
                                 "num_heads": 16, "out_hidden_size": 2048, "patch_size": 16,
                                 "spatial_merge_size": 2, "temporal_patch_size": 2,
                                 "in_channels": 3, "num_position_embeddings": 2304,
                                 "deepstack_visual_indexes": [5, 11, 17]}}
QWEN3_PREPROCESSOR = {"processor_class": "Qwen3VLProcessor", "image_processor_type": "Qwen2VLImageProcessorFast",
                     "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                     "size": {"shortest_edge": 65536, "longest_edge": 16777216},
                     "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
RESPONSE = json.dumps({"human_present": {"value": True, "evidence_frame_indices": [1]},
                       "hands_visible": {"value": False, "evidence_frame_indices": [0]},
                       "object_manipulation": {"value": None, "evidence_frame_indices": []}})


class FakeBatch(dict):
    def to(self, device):
        self.selected_device = device
        return self


class PilotVLMTests(unittest.TestCase):
    def setUp(self):
        try:
            self.subject = importlib.import_module("dbmetrics._vlm")
        except ModuleNotFoundError:
            self.fail("bounded dbmetrics._vlm adapter is not implemented")
        temporary = tempfile.TemporaryDirectory(prefix="dbp-vlm-software-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        environment = patch.dict(os.environ, {}, clear=False)
        environment.start()
        self.addCleanup(environment.stop)
        self.model_dir = self.root / "models--Qwen--Qwen2.5-VL-3B-Instruct" / "snapshots" / REVISION
        self.model_dir.mkdir(parents=True)
        self.json_file("config.json", CONFIG)
        self.json_file("preprocessor_config.json", PREPROCESSOR)
        self.json_file("tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
        self.json_file("tokenizer.json", {"fixture": True})
        (self.model_dir / "model.safetensors").write_bytes(b"not actual weights; never loaded")
        self.media = self.root / "fixture.mp4"
        self.media.write_bytes(b"not actual media")
        self.deadline = time.monotonic() + 10

    def json_file(self, name, payload):
        (self.model_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    def validate(self, **changes):
        options = {"local_model_dir": self.model_dir, "revision": REVISION, "deadline": self.deadline}
        return self.subject.validate_local_model(**(options | changes))

    def qwen3_fixture(self):
        self.model_dir = self.root / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / QWEN3_REVISION
        self.model_dir.mkdir(parents=True)
        self.json_file("config.json", QWEN3_CONFIG)
        self.json_file("preprocessor_config.json", QWEN3_PREPROCESSOR)
        self.json_file("tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})
        self.json_file("tokenizer.json", {"fixture": True})
        (self.model_dir / "model.safetensors").write_bytes(b"not actual weights; never loaded")
        return {"model_id": QWEN3_ID, "revision": QWEN3_REVISION}

    def showinfo(self, timestamps=(0.5, 1.5)):
        lines = ["[Parsed_showinfo_5 @ fixture] [info] config in time_base: 1/1000, frame_rate: 30/1"]
        for index, timestamp in enumerate(timestamps):
            lines.append(f"[Parsed_showinfo_5 @ fixture] [info] n: {index} pts: {int(timestamp * 1000)} "
                         f"pts_time:{timestamp} fmt:rgb24 sar:1/1 s:448x448 i:P iskey:1 type:I")
        return "\n".join(lines).encode()

    def decoder_output(self, timestamps=(0.5, 1.5)):
        return bytes(448 * 448 * 3 * len(timestamps)), self.showinfo(timestamps)

    def runtime(self, response=RESPONSE, prompt_length=3, generated_length=2):
        batch = FakeBatch(input_ids=[[7] * prompt_length], attention_mask=[[1] * prompt_length])
        processor = Mock()
        processor.apply_chat_template.return_value = "fixture formatted prompt"
        processor.return_value = batch
        processor.batch_decode.return_value = [response]
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.generate.return_value = [[7] * prompt_length + [9] * generated_length]
        torch = SimpleNamespace(__version__="2.9.1", float16="fixture-float16", bfloat16="fixture-bfloat16",
            inference_mode=nullcontext, cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 2,
                set_device=Mock(), is_bf16_supported=Mock(return_value=True)))
        transformers = SimpleNamespace(__version__="4.57.5",
            Qwen2_5_VLProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
            Qwen2_5_VLForConditionalGeneration=SimpleNamespace(from_pretrained=Mock(return_value=model)),
            Qwen3VLProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
            Qwen3VLForConditionalGeneration=SimpleNamespace(from_pretrained=Mock(return_value=model)))
        pillow = SimpleNamespace(Image=SimpleNamespace(frombytes=lambda mode, size, raw: (mode, size, raw)))
        modules = {"torch": torch, "transformers": transformers, "PIL": pillow,
                   "torchvision": SimpleNamespace(__version__="0.24.1"),
                   "torchvision.transforms.v2": SimpleNamespace(functional=SimpleNamespace(resize=lambda *args: None))}
        return modules, model, processor, batch

    def infer(self, **changes):
        options = {"descriptor": None, "probe": {"duration_seconds": {"value": 2.0},
                    "width_px": {"value": 640}, "height_px": {"value": 360}},
                   "local_model_dir": self.model_dir, "revision": REVISION, "device": "cuda:1",
                   "ffmpeg": "fixture-ffmpeg", "deadline": self.deadline, "frame_count": 2}
        with self.media.open("rb") as media:
            options["descriptor"] = media.fileno()
            return self.subject.infer_clip(**(options | changes))

    def test_snapshot_revision_and_config_are_verified_before_loading(self):
        metadata = self.validate()
        self.assertEqual(metadata["model_id"], "Qwen/Qwen2.5-VL-3B-Instruct")
        self.assertEqual(metadata["model_revision"], REVISION)

        self.assertEqual(metadata["revision_source"], "local_hf_snapshot_path")
        self.assertEqual(metadata["file_sha256"]["config.json"],
                         hashlib.sha256((self.model_dir / "config.json").read_bytes()).hexdigest())
        with self.assertRaises(ValueError):
            self.validate(revision="b" * 40)
        with self.assertRaises(ValueError):
            self.validate(local_model_dir=self.root)
        self.json_file("config.json", CONFIG | {"_commit_hash": "b" * 40})
        with self.assertRaises(ValueError):
            self.validate()

    def test_runtime_reuses_verified_weights_but_checks_mutation_and_identity(self):
        modules, model, processor, batch = self.runtime()
        runtime = self.subject.VLMRuntime()
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()), \
                patch("dbmetrics._vlm.validate_local_model", wraps=self.subject.validate_local_model) as validate:
            self.infer(runtime=runtime)
            self.infer(runtime=runtime)
            self.assertEqual(validate.call_count, 1)
            modules["transformers"].Qwen2_5_VLForConditionalGeneration.from_pretrained.assert_called_once()
            self.assertEqual(model.generate.call_count, 2)
            with self.assertRaisesRegex(ValueError, "identity"):
                self.infer(runtime=runtime, device="cuda:0")
            (self.model_dir / "model.safetensors").write_bytes(b"changed weights")
            with self.assertRaisesRegex(ValueError, "changed"):
                self.infer(runtime=runtime)
            self.assertEqual(model.generate.call_count, 2)

    def test_insufficient_distinct_frames_abstains_without_generation(self):
        modules, model, processor, batch = self.runtime()
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.decode_frames", side_effect=self.subject.InsufficientFrames("too few distinct frames")):
            measured = self.infer()
        self.assertTrue(all(record["value"] is None for record in measured.values()))
        self.assertTrue(all(record["processing_status"] == "missing" for record in measured.values()))
        self.assertTrue(all(record["evidence"][0]["reason"] == "insufficient_distinct_frames" for record in measured.values()))
        model.generate.assert_not_called()

    def test_prompt_response_example_obeys_evidence_parser(self):
        self.assertIn("{", self.subject.PROMPT, "The model needs an explicit nested JSON response example")
        example, _ = json.JSONDecoder().raw_decode(self.subject.PROMPT[self.subject.PROMPT.index("{"):])
        parsed = dbmetrics.legacy.parse_vlm_response(json.dumps(example), [0.5, 1.5])["fields"]
        self.assertEqual(set(parsed), set(dbmetrics.legacy.SEMANTIC_FIELDS))
        self.assertEqual(set(example), set(dbmetrics.legacy.SEMANTIC_FIELDS))
        self.assertTrue(all(field["value"] is None for field in parsed.values()))
        example["hands_visible"] = {"value": True, "evidence_frame_indices": [1]}
        hands = dbmetrics.legacy.parse_vlm_response(json.dumps(example), [0.5, 1.5])["fields"]["hands_visible"]
        self.assertTrue(hands["value"])
        self.assertEqual(hands["evidence"][0]["observed_frame_timestamps_seconds"], [1.5])

    def test_qwen3_requires_explicit_identity_and_the_approved_revision(self):
        options = self.qwen3_fixture()
        metadata = self.validate(**options)
        self.assertEqual(metadata["model_id"], QWEN3_ID)
        self.assertEqual(metadata["model_revision"], QWEN3_REVISION)
        self.assertEqual(self.subject.snapshot_path(self.model_dir, QWEN3_REVISION, model_id=QWEN3_ID),
                         self.model_dir)
        for changes in ({"model_id": "Qwen/Qwen2.5-VL-3B-Instruct"}, {"model_id": "auto"},
                        {"model_id": "Qwen/Qwen3-VL-4B-Instruct"}, {"model_id": None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate(**(options | changes))
        with self.assertRaises(ValueError):
            self.validate(revision=QWEN3_REVISION)
        self.model_dir = self.model_dir.rename(self.model_dir.with_name(REVISION))
        with self.assertRaises(ValueError):
            self.validate(model_id=QWEN3_ID)

    def test_qwen3_8b_has_separate_pinned_shape_and_byte_budget(self):
        self.assertTrue(hasattr(self.subject, "QWEN3_8B_MODEL_ID"))
        self.qwen3_fixture()
        model_id = self.subject.QWEN3_8B_MODEL_ID
        revision = self.subject.QWEN3_8B_REVISION
        destination = self.root / "models--Qwen--Qwen3-VL-8B-Instruct" / "snapshots" / revision
        destination.parent.mkdir(parents=True)
        self.model_dir = self.model_dir.rename(destination)
        config = json.loads(json.dumps(QWEN3_CONFIG))
        config["text_config"].update(hidden_size=4096, intermediate_size=12288,
                                    num_hidden_layers=36, num_attention_heads=32)
        config["vision_config"].update(depth=27, hidden_size=1152, intermediate_size=4304,
                                      out_hidden_size=4096, deepstack_visual_indexes=[8, 16, 24])
        self.json_file("config.json", config)
        self.assertEqual(self.validate(model_id=model_id, revision=revision)["model_id"], model_id)
        with self.assertRaises(ValueError):
            self.validate(model_id=model_id, revision=QWEN3_REVISION)
        config["vision_config"]["deepstack_visual_indexes"] = [5, 11, 17]
        self.json_file("config.json", config)
        with self.assertRaises(ValueError):
            self.validate(model_id=model_id, revision=revision)
        self.assertEqual(self.subject.MODEL_VARIANTS[model_id]["max_model_bytes"], 24 * 1024 ** 3)
        self.assertEqual(self.subject.MAX_MODEL_BYTES, 12 * 1024 ** 3)

    def test_qwen3_rejects_changed_shapes_deepstack_and_cross_family_configs(self):
        options = self.qwen3_fixture()
        self.validate(**options)
        changes = [{"architectures": CONFIG["architectures"]}, {"model_type": "qwen2_5_vl"},
                   {"_commit_hash": REVISION}, {"text_config": None}, {"vision_config": None}]
        for section in ("text_config", "vision_config"):
            for key, value in QWEN3_CONFIG[section].items():
                invalid = (None, [], [5, 11, 18], [5.0, 11, 17]) if isinstance(value, list) else (value + 1, float(value), True)
                changes.extend({section: QWEN3_CONFIG[section] | {key: candidate}} for candidate in invalid)
        for change in changes:
            self.json_file("config.json", QWEN3_CONFIG | change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(**options)

    def test_qwen3_rejects_processor_swaps_and_nested_loading_directives(self):
        options = self.qwen3_fixture()
        self.validate(**options)
        for change in ({"processor_class": "Qwen2_5_VLProcessor"}, {"image_processor_type": "Qwen2VLImageProcessor"},
                       {"patch_size": 14}, {"merge_size": 4}, {"temporal_patch_size": 2.0}):
            self.json_file("preprocessor_config.json", QWEN3_PREPROCESSOR | change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(**options)
        self.json_file("preprocessor_config.json", QWEN3_PREPROCESSOR)
        for name, baseline in (("config.json", QWEN3_CONFIG), ("preprocessor_config.json", QWEN3_PREPROCESSOR),
                               ("tokenizer_config.json", {"tokenizer_class": "Qwen2Tokenizer"})):
            for forbidden in ("auto_map", "quantization_config", "custom_pipelines"):
                self.json_file(name, baseline | {"nested": [{forbidden: {}}]})
                with self.subTest(name=name, forbidden=forbidden), self.assertRaises(ValueError):
                    self.validate(**options)
            self.json_file(name, baseline)
        self.json_file("tokenizer_config.json", {"tokenizer_class": "OtherTokenizer"})
        with self.assertRaises(ValueError):
            self.validate(**options)

    def test_qwen3_preserves_snapshot_file_and_hash_safeguards(self):
        options = self.qwen3_fixture()
        for name in ("pytorch_model.bin", "custom.py", "adapter_config.json"):
            forbidden = self.model_dir / name
            forbidden.write_bytes(b"fixture")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate(**options)
            forbidden.unlink()
        weights = self.model_dir / "model.safetensors"
        payload = weights.read_bytes()
        weights.unlink()
        with self.assertRaises(ValueError):
            self.validate(**options)
        weights.symlink_to(self.media)
        with self.assertRaises(ValueError):
            self.validate(**options)
        weights.unlink()
        blobs = self.model_dir.parent.parent / "blobs"
        blobs.mkdir()
        blob = blobs / hashlib.sha256(payload).hexdigest()
        blob.write_bytes(payload)
        weights.symlink_to(blob)
        verified = self.validate(**options)
        self.assertEqual(verified["file_sha256"]["model.safetensors"], blob.name)
        self.json_file("model.safetensors.index.json", {"weight_map": {"fixture": "../escape.safetensors"}})
        with self.assertRaises(ValueError):
            self.validate(**options)
        (self.model_dir / "model.safetensors.index.json").unlink()
        blob.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.validate(**options)

    def test_qwen3_optional_video_processor_cannot_route_to_another_architecture(self):
        options = self.qwen3_fixture()
        video = {"processor_class": "Qwen3VLProcessor", "video_processor_type": "Qwen3VLVideoProcessor",
                 "patch_size": 16, "temporal_patch_size": 2, "merge_size": 2,
                 "size": {"shortest_edge": 4096, "longest_edge": 25165824},
                 "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]}
        self.json_file("video_preprocessor_config.json", video)
        self.validate(**options)
        for change in ({"video_processor_type": "OtherVideoProcessor"}, {"processor_class": "OtherProcessor"},
                       {"patch_size": 14}, {"merge_size": 2.0}, {"temporal_patch_size": 4}):
            self.json_file("video_preprocessor_config.json", video | change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(**options)

    def test_qwen3_missing_runtime_classes_fail_before_decode_without_fallback(self):
        options = self.qwen3_fixture()
        for missing in ("Qwen3VLProcessor", "Qwen3VLForConditionalGeneration"):
            modules, model, processor, batch = self.runtime()
            delattr(modules["transformers"], missing)
            with self.subTest(missing=missing), patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded") as decoder:
                with self.assertRaisesRegex(ValueError, "Qwen3"):
                    self.infer(**options)
            decoder.assert_not_called()
            model.to.assert_not_called()
            modules["transformers"].Qwen2_5_VLProcessor.from_pretrained.assert_not_called()
            modules["transformers"].Qwen2_5_VLForConditionalGeneration.from_pretrained.assert_not_called()

    def test_qwen3_dispatch_is_local_bounded_and_reports_observed_evidence(self):
        options = self.qwen3_fixture()
        modules, model, processor, batch = self.runtime()
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output((0.53, 1.57))):
            measured = self.infer(**options)
        transformers = modules["transformers"]
        transformers.Qwen2_5_VLProcessor.from_pretrained.assert_not_called()
        transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained.assert_not_called()
        for loader in (transformers.Qwen3VLProcessor.from_pretrained,
                       transformers.Qwen3VLForConditionalGeneration.from_pretrained):
            self.assertEqual(loader.call_args.args, (str(self.model_dir),))
            self.assertEqual(loader.call_args.kwargs["revision"], QWEN3_REVISION)
            self.assertIs(loader.call_args.kwargs["local_files_only"], True)
            self.assertIs(loader.call_args.kwargs["trust_remote_code"], False)
        processor_options = transformers.Qwen3VLProcessor.from_pretrained.call_args.kwargs
        self.assertIs(processor_options["use_fast"], True)
        self.assertEqual(processor_options["min_pixels"], 4096)
        self.assertEqual(processor_options["max_pixels"], 200704)
        model_options = transformers.Qwen3VLForConditionalGeneration.from_pretrained.call_args.kwargs
        self.assertIs(model_options["use_safetensors"], True)
        self.assertIsNone(model_options["device_map"])
        self.assertEqual(model_options["attn_implementation"], "eager")
        model.to.assert_called_once_with("cuda:1")
        self.assertEqual(batch.selected_device, "cuda:1")
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"], 1024)
        self.assertFalse(model.generate.call_args.kwargs["do_sample"])
        self.assertEqual(len(processor.call_args.kwargs["images"]), 2)
        self.assertNotIn("videos", processor.call_args.kwargs)
        self.assertEqual(processor.apply_chat_template.call_args.kwargs["chat_template"], self.subject.CHAT_TEMPLATE)
        self.assertIs(measured["human_present"]["value"], True)
        self.assertIsNone(measured["hands_visible"]["value"])
        self.assertEqual(measured["visible_text_present"]["processing_status"], "missing")
        evidence = measured["human_present"]["evidence"][0]
        self.assertEqual(evidence["observed_frame_timestamps_seconds"], [1.57])
        self.assertEqual(evidence["run_provenance"]["model_snapshot"]["model_id"], QWEN3_ID)
        for measurement in measured.values():
            self.assertEqual(measurement["model_id"], QWEN3_ID)
            self.assertEqual(measurement["model_revision"], QWEN3_REVISION)

    def test_qwen3_invalid_responses_are_unknown_not_negative(self):
        options = self.qwen3_fixture()
        flat_response = json.dumps({"human_present": True, "hands_visible": True, "object_manipulation": True,
                                    "visible_text_present": True, "added_text_present": None})
        for response in ("not JSON", flat_response,
                         json.dumps({"human_present": {"value": True, "evidence_frame_indices": [2]}})):
            modules, model, processor, batch = self.runtime(response=response)
            with self.subTest(response=response), patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()):
                measured = self.infer(**options)
            for measurement in measured.values():
                self.assertIsNone(measurement["value"])
                self.assertEqual(measurement["processing_status"], "error")
                self.assertEqual(measurement["model_id"], QWEN3_ID)

    def test_qwen3_uses_bfloat16_after_checking_the_selected_device_without_emulation(self):
        options = self.qwen3_fixture()
        modules, model, processor, batch = self.runtime()
        cuda = modules["torch"].cuda

        def supported(*, including_emulation):
            cuda.set_device.assert_called_once_with("cuda:1")
            self.assertIs(including_emulation, False)
            return True

        cuda.is_bf16_supported.side_effect = supported
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()):
            measured = self.infer(**options)
        cuda.is_bf16_supported.assert_called_once_with(including_emulation=False)
        loader = modules["transformers"].Qwen3VLForConditionalGeneration.from_pretrained
        self.assertEqual(loader.call_args.kwargs["dtype"], "fixture-bfloat16")
        self.assertEqual(measured["human_present"]["evidence"][0]["run_provenance"]["model_dtype"], "bfloat16")

    def test_qwen3_unsupported_or_unverifiable_bfloat16_fails_before_decode_and_loading(self):
        options = self.qwen3_fixture()
        for support in (False, None, 1, "missing", RuntimeError("device error"), TypeError("old API")):
            modules, model, processor, batch = self.runtime()
            cuda = modules["torch"].cuda
            if support == "missing":
                delattr(cuda, "is_bf16_supported")
            elif isinstance(support, Exception):
                cuda.is_bf16_supported.side_effect = support
            else:
                cuda.is_bf16_supported.return_value = support
            with self.subTest(support=support), patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()) as decoder:
                with self.assertRaisesRegex(ValueError, "bfloat16.*cuda:1"):
                    self.infer(**options)
                decoder.assert_not_called()
                modules["transformers"].Qwen3VLProcessor.from_pretrained.assert_not_called()
                modules["transformers"].Qwen3VLForConditionalGeneration.from_pretrained.assert_not_called()
                model.generate.assert_not_called()

    def test_qwen25_retains_float16_without_requiring_bfloat16_support(self):
        modules, model, processor, batch = self.runtime()
        modules["torch"].cuda.is_bf16_supported.side_effect = AssertionError("Qwen2.5 must not require BF16")
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()):
            measured = self.infer()
        loader = modules["transformers"].Qwen2_5_VLForConditionalGeneration.from_pretrained
        self.assertEqual(loader.call_args.kwargs["dtype"], "fixture-float16")
        self.assertEqual(measured["human_present"]["evidence"][0]["run_provenance"]["model_dtype"], "float16")

    def test_qwen3_preserves_device_token_environment_and_mutation_gates(self):
        options = self.qwen3_fixture()
        for device in ("auto", "cpu", "cuda", "cuda:2"):
            modules, model, processor, batch = self.runtime()
            with self.subTest(device=device), patch.dict(sys.modules, modules), self.assertRaises(ValueError):
                self.infer(**options, device=device)
            model.generate.assert_not_called()
        for prompt_length, generated_length in ((4097, 2), (3, 1025)):
            modules, model, processor, batch = self.runtime(prompt_length=prompt_length, generated_length=generated_length)
            with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()), self.assertRaises(ValueError):
                self.infer(**options)
        modules, model, processor, batch = self.runtime()
        modules["torchvision"] = SimpleNamespace(__version__="0.23.0")
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded") as decoder, self.assertRaisesRegex(ValueError, "torchvision"):
            self.infer(**options)
        decoder.assert_not_called()
        transformers = modules["transformers"]
        transformers.Qwen3VLForConditionalGeneration.from_pretrained.assert_not_called()
        modules, model, processor, batch = self.runtime()
        model.generate.side_effect = lambda **kwargs: self.json_file("config.json", QWEN3_CONFIG | {"changed": True}) or [[7, 7, 7, 9, 9]]
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()), self.assertRaises(ValueError):
            self.infer(**options)

    def test_config_architecture_remote_code_quantization_and_large_models_are_rejected(self):
        for changes in ({"architectures": ["OtherModel"]}, {"model_type": "other"},
                        {"auto_map": {}}, {"quantization_config": {}}, {"hidden_size": 8192},
                        {"vision_config": CONFIG["vision_config"] | {"depth": 1024}}):
            self.json_file("config.json", CONFIG | changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.validate()

    def test_snapshot_rejects_pickle_python_missing_weights_and_escaping_symlinks(self):
        for name in ("pytorch_model.bin", "custom.py", "adapter_config.json"):
            forbidden = self.model_dir / name
            forbidden.write_bytes(b"fixture")
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate()
            forbidden.unlink()
        weights = self.model_dir / "model.safetensors"
        weights.unlink()
        with self.assertRaises(ValueError):
            self.validate()
        weights.symlink_to(self.media)
        with self.assertRaises(ValueError):
            self.validate()

    def test_hf_blob_symlink_hash_is_verified(self):
        weights = self.model_dir / "model.safetensors"
        data = weights.read_bytes()
        weights.unlink()
        blobs = self.model_dir.parent.parent / "blobs"
        blobs.mkdir()
        blob = blobs / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        weights.symlink_to(blob)
        self.assertEqual(self.validate()["file_sha256"]["model.safetensors"], blob.name)
        blob.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.validate()

    def test_safetensors_index_cannot_name_external_or_pickle_files(self):
        for target in ("../other.safetensors", "pytorch_model.bin", "missing.safetensors"):
            self.json_file("model.safetensors.index.json", {"weight_map": {"fixture": target}})
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.validate()

    def test_uniform_requested_times_are_not_substituted_for_observed_pts(self):
        self.assertEqual(self.subject.sample_times(4, 2), [1.0, 3.0])
        for duration, count in ((0, 2), (601, 2), (float("nan"), 2), (2, 9), (2, True)):
            with self.subTest(duration=duration, count=count), self.assertRaises(ValueError):
                self.subject.sample_times(duration, count)
        frames = self.subject.parse_frames(*self.decoder_output((0.53, 1.57)), [0.5, 1.5], 2.0)
        self.assertEqual([frame["actual_seconds"] for frame in frames], [0.53, 1.57])
        self.assertEqual([frame["requested_seconds"] for frame in frames], [0.5, 1.5])
        self.assertEqual(frames[0]["pts"], 530)
        self.assertEqual(frames[0]["time_base"], "1/1000")

    def test_frame_parser_rejects_missing_duplicate_nonfinite_mismatched_or_error_pts(self):
        raw, diagnostics = self.decoder_output()
        cases = [(raw[:-1], diagnostics), (raw, diagnostics.replace(b"n: 1", b"n: 0")),
                 (raw, diagnostics.replace(b"pts_time:1.5", b"pts_time:nan")),
                 (raw, diagnostics.replace(b"pts_time:1.5", b"pts_time:1.7")),
                 (raw, diagnostics + b"\n[error] corrupt frame"),
                 (raw, diagnostics.replace(b"s:448x448", b"s:640x480")),
                 (raw, b"no actual timestamp information"),
                 (raw, diagnostics.replace(b"1/1000", b"1/0")),
                 self.decoder_output((0.4, 1.5)), self.decoder_output((0.5, 2.0))]
        for pixels, stderr in cases:
            with self.subTest(stderr=stderr[:60]), self.assertRaises(ValueError):
                self.subject.parse_frames(pixels, stderr, [0.5, 1.5], 2.0)

    def test_decoder_is_single_local_invocation_with_bounded_output_and_actual_pts(self):
        binary = self.root / "fixture-ffmpeg"
        binary.write_text("#!" + sys.executable + "\nimport sys\n"
            "assert sys.argv[sys.argv.index('-protocol_whitelist') + 1] == 'file,pipe'\n"
            "assert sys.argv[sys.argv.index('-f') + 1] == 'mov'\n"
            "assert sys.argv[sys.argv.index('-frames:v') + 1] == '2'\n"
            "assert '-xerror' in sys.argv\n"
            "assert 'setpts=PTS-STARTPTS' in sys.argv[sys.argv.index('-vf') + 1]\n"
            "assert sys.argv[-1] == 'pipe:1'\n"
            "sys.stdout.buffer.write(bytes(448 * 448 * 3 * 2))\n"
            "sys.stderr.buffer.write(" + repr(self.showinfo((0.53, 1.57))) + ")\n")
        binary.chmod(0o700)
        with self.media.open("rb") as media:
            frames, provenance = self.subject.decode_frames(media.fileno(), 2.0, 2, str(binary), self.deadline)
        self.assertEqual([frame["actual_seconds"] for frame in frames], [0.53, 1.57])
        self.assertEqual(provenance["timestamp_origin"], "first_decoded_frame_pts")

    def test_decoder_timeout_oversized_output_and_nonzero_exit_fail_closed(self):
        binary = self.root / "fixture-ffmpeg"
        for body, seconds in (("import time\ntime.sleep(10)", 0.15),
                              ("import sys\nsys.stdout.buffer.write(bytes(2000000))", 5),
                              ("import sys\nsys.exit(1)", 5)):
            binary.write_text("#!" + sys.executable + "\n" + body + "\n")
            binary.chmod(0o700)
            with self.subTest(body=body), self.media.open("rb") as media, self.assertRaises(ValueError):
                self.subject.decode_frames(media.fileno(), 2.0, 2, str(binary), time.monotonic() + seconds)

    def test_legacy_contract_documents_the_executable_adapter_bounds(self):
        contract = dbmetrics.legacy.vlm_adapter_contract(self.model_dir, REVISION)
        self.assertEqual(contract["sampling"]["max_frames"], 8)
        self.assertEqual(contract["sampling"]["max_frame_side_px"], 448)

    def test_inference_is_local_pinned_single_device_and_token_bounded(self):
        modules, model, processor, batch = self.runtime()
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()), patch.dict(os.environ, {}, clear=False):
            results = self.infer()
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")
        loader = modules["transformers"].Qwen2_5_VLForConditionalGeneration.from_pretrained
        kwargs = loader.call_args.kwargs
        self.assertEqual(loader.call_args.args[0], str(self.model_dir))
        self.assertTrue(kwargs["local_files_only"])
        self.assertFalse(kwargs["trust_remote_code"])
        self.assertTrue(kwargs["use_safetensors"])
        self.assertEqual(kwargs["revision"], REVISION)
        self.assertIsNone(kwargs["device_map"])
        self.assertEqual(batch.selected_device, "cuda:1")
        model.to.assert_called_once_with("cuda:1")
        modules["torch"].cuda.set_device.assert_called_once_with("cuda:1")
        generation = model.generate.call_args.kwargs
        self.assertEqual(generation["max_new_tokens"], 1024)
        self.assertEqual(generation["num_beams"], 1)
        self.assertFalse(generation["do_sample"])
        self.assertNotIn("videos", processor.call_args.kwargs)
        self.assertEqual(len(processor.call_args.kwargs["images"]), 2)
        conversation = processor.apply_chat_template.call_args.args[0]
        self.assertIn("sample 0", str(conversation))
        self.assertIn("sample 1", str(conversation))
        self.assertIs(results["human_present"]["value"], True)
        self.assertIsNone(results["hands_visible"]["value"])
        self.assertEqual(results["human_present"]["evidence"][0]["run_provenance"]["raw_response"], RESPONSE)
        self.assertNotIn("confidence", str(results))
        processor.batch_decode.assert_called_once_with([[9, 9]], skip_special_tokens=True,
                                                        clean_up_tokenization_spaces=False)

    def test_bad_response_is_error_with_private_raw_response_not_a_negative(self):
        modules, model, processor, batch = self.runtime(response="not JSON")
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()):
            results = self.infer()
        self.assertEqual(results["human_present"]["evidence"][0]["run_provenance"]["raw_response"], "not JSON")
        for value in results.values():
            self.assertIsNone(value["value"])
            self.assertEqual(value["processing_status"], "error")

    @unittest.skipUnless(os.environ.get('DBP_WEBSITE_TEST_ROOT'), 'Optional website artifact validation')
    def test_all_null_finished_run_builds_with_complete_coverage_and_omitted_fields_missing(self):
        import test_pilot_build

        response = json.dumps({field: {"value": False if field == "hands_visible" else None,
                                      "evidence_frame_indices": [0] if field == "hands_visible" else []}
                               for field in dbmetrics.legacy.SEMANTIC_FIELDS[:3]})
        modules, model, processor, batch = self.runtime(response=response)
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()):
            measured = self.infer()
        for field in dbmetrics.legacy.SEMANTIC_FIELDS[:3]:
            self.assertEqual(measured[field]["processing_status"], "complete")
            self.assertIsNone(measured[field]["value"])
        self.assertEqual(measured["visible_text_present"]["processing_status"], "missing")
        fixture = test_pilot_build.PilotBuildTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        records = []
        for clip in fixture.manifest["clips"]:
            for record in dbmetrics.legacy.empty_records(clip["clip_id"], "/fixture/clip.mp4", "original"):
                if record["field_id"] in measured:
                    record.update(measured[record["field_id"]])
                    record["media_sha256"] = hashlib.sha256(self.media.read_bytes()).hexdigest()
                    records.append(record)
        snapshot = fixture.build(measurement_paths=[fixture.measurements(records)], semantic_method="video_vlm")
        for field in dbmetrics.legacy.SEMANTIC_FIELDS[:3]:
            self.assertEqual(sum(clip["annotations"][field]["processing_status"] == "complete" for clip in snapshot["clips"]), 23)
            self.assertTrue(all(clip["annotations"][field]["value"] is None for clip in snapshot["clips"]))

    def test_missing_or_incompatible_torchvision_fails_preflight_before_decode_or_model_load(self):
        for vision in (None, SimpleNamespace(__version__="0.23.0")):
            modules, model, processor, batch = self.runtime()
            modules["torchvision"] = vision
            with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded") as decoder:
                with self.assertRaisesRegex(ValueError, "torchvision"):
                    self.infer()
            decoder.assert_not_called()
            model.generate.assert_not_called()

    @unittest.skipUnless(os.environ.get('DBP_WEBSITE_TEST_ROOT'), 'Optional website artifact validation')
    def test_inference_error_and_unknown_results_preserve_exact_interchange(self):
        import pilot
        import pilot_build

        self.media = self.media.rename(self.root / "fixture_part_00000001.mp4")
        probe = self.root / "fixture-ffprobe"
        probe.write_text("#!" + sys.executable + "\nprint(" + repr(json.dumps({"format": {"duration": "2"},
            "streams": [{"index": 0, "codec_type": "video", "width": 640, "height": 360,
                         "avg_frame_rate": "30/1"}]})) + ")\n")
        probe.chmod(0o700)
        options = {"local_model_dir": str(self.model_dir), "revision": REVISION, "device": "cuda:1",
                   "ffmpeg": "fixture-ffmpeg", "frame_count": 2}
        modules, model, processor, batch = self.runtime()
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()):
            records, failed = dbmetrics.legacy.extract_clip({"clip_id": "fixture_part_00000001"}, self.root, "original",
                str(probe), time.monotonic() + 10, 1024, False, 27.0, vlm_options=options)
        fields = {record["field_id"]: record for record in records}
        self.assertFalse(failed)
        self.assertEqual(fields["human_present"]["method"], "video_vlm")
        self.assertEqual(fields["human_present"]["processing_status"], "complete")
        self.assertEqual(fields["human_present"]["media_sha256"], hashlib.sha256(self.media.read_bytes()).hexdigest())
        self.assertEqual(fields["human_present"]["model_revision"], REVISION)
        self.assertIsNone(fields["hands_visible"]["value"])
        self.assertIs(fields["audio_track_present"]["value"], False)
        self.assertIsInstance(fields["human_present"]["evidence"], list)
        clip = {"clip_id": "fixture_part_00000001", "source_id": "fixture", "part": 1,
                "caption": None, "transcript": None, "annotations": fields}
        snapshot = {"schema_version": "dbt1-filter-pilot-v1", "manifest_sha256": "a" * 64,
                    "source_export_sha256": "b" * 64, "semantic_method": "video_vlm", "clips": [clip]}
        pilot.validate_snapshot(snapshot)
        payload = {"schema_version": "dbt1-pilot-measurements-v1", "manifest_sha256": "a" * 64,
                   "run_id": "software-fixture", "records": records}
        measured = pilot_build.validate_measurements(payload, "c" * 64, "a" * 64, {clip["clip_id"]: clip})
        self.assertEqual({field for clip_id, field, method in measured}, set(dbmetrics.legacy.ALL_FIELDS))
        for field in pilot.TECHNICAL_FIELDS:
            self.assertEqual(snapshot["clips"][0]["annotations"][field], pilot.unknown_annotation())

    def test_evidence_indices_are_chronological_and_paired_with_actual_timestamps(self):
        response = json.loads(RESPONSE)
        response["human_present"]["evidence_frame_indices"] = [1, 0]
        modules, model, processor, batch = self.runtime(response=json.dumps(response))
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()):
            measured = self.infer()
        evidence = measured["human_present"]["evidence"][0]
        self.assertEqual(evidence["evidence_frame_indices"], [0, 1])
        self.assertEqual(evidence["observed_frame_timestamps_seconds"], [0.5, 1.5])

    def test_vlm_worker_diagnostics_do_not_discard_valid_private_records(self):
        media_root = self.root / "media"
        media_root.mkdir()
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": "dbt1-evidence-manifest-v1", "clips": [
            {"clip_id": "fixture_part_00000001", "source_id": "fixture", "part": 1}]}))
        output = self.root / "output.json"
        records = dbmetrics.legacy.empty_records("fixture_part_00000001", media_root / "fixture_part_00000001.mp4", "original")

        def worker(command, deadline, limit, **options):
            self.assertTrue(options.get("capture_stderr"))
            return json.dumps({"records": records, "failed": False}).encode(), b"bounded loader progress"

        with patch("dbmetrics.legacy.run_bounded", side_effect=worker):
            status = dbmetrics.legacy.extract_pilot(manifest, media_root, output, "original", vlm=True,
                local_vlm_dir=self.model_dir, model_revision=REVISION, device="cuda:1")
        self.assertEqual(status, 0)
        fields = {record["field_id"]: record for record in json.loads(output.read_text())["records"]}
        self.assertNotIn("run_provenance", fields["human_present"])
        self.assertEqual(fields["human_present"]["evidence"][0]["run_provenance"]["worker_diagnostics"], "bounded loader progress")

    def test_real_worker_reports_bad_local_model_without_importing_or_running_ml(self):
        media_root = self.root / "media"
        media_root.mkdir()
        (media_root / "fixture_part_00000001.mp4").write_bytes(b"software fixture")
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": "dbt1-evidence-manifest-v1", "clips": [
            {"clip_id": "fixture_part_00000001", "source_id": "fixture", "part": 1}]}))
        self.json_file("config.json", CONFIG | {"model_type": "not_approved"})
        probe = self.root / "fixture-ffprobe"
        probe.write_text("#!" + sys.executable + "\nprint(" + repr(json.dumps({"format": {"duration": "2"},
            "streams": [{"index": 0, "codec_type": "video", "width": 640, "height": 360,
                         "avg_frame_rate": "30/1"}]})) + ")\n")
        probe.chmod(0o700)
        output = self.root / "output.json"
        status = dbmetrics.legacy.extract_pilot(manifest, media_root, output, "original", ffprobe=str(probe), vlm=True,
            local_vlm_dir=self.model_dir, model_revision=REVISION, device="cuda:0")
        self.assertEqual(status, 1)
        fields = {record["field_id"]: record for record in json.loads(output.read_text())["records"]}
        self.assertEqual(set(fields), set(dbmetrics.legacy.SEMANTIC_FIELDS))
        self.assertEqual(fields["human_present"]["processing_status"], "error")
        self.assertIn("identity", fields["human_present"]["evidence"][0]["reason"])

    def test_explicit_cuda_index_version_and_token_bounds_are_enforced(self):
        for device in ("auto", "cuda", "cpu", "cuda:99", "cuda:-1", "cuda:2"):
            modules, model, processor, batch = self.runtime()
            with self.subTest(device=device), patch.dict(sys.modules, modules), self.assertRaises(ValueError):
                self.infer(device=device)
            model.generate.assert_not_called()
        for prompt_length, generated_length in ((4097, 2), (3, 1025)):
            modules, model, processor, batch = self.runtime(prompt_length=prompt_length, generated_length=generated_length)
            with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()), self.assertRaises(ValueError):
                self.infer()
        modules, model, processor, batch = self.runtime()
        modules["transformers"].__version__ = "5.0.0"
        with patch.dict(sys.modules, modules), self.assertRaises(ValueError):
            self.infer()

    def test_model_change_while_loading_is_rejected(self):
        modules, model, processor, batch = self.runtime()
        model.generate.side_effect = lambda **kwargs: self.json_file("config.json", CONFIG | {"changed": True}) or [[7, 7, 7, 9, 9]]
        with patch.dict(sys.modules, modules), patch("dbmetrics._vlm.run_bounded", return_value=self.decoder_output()), self.assertRaises(ValueError):
            self.infer()

    def test_cli_requires_all_vlm_flags_and_cannot_write_raw_responses_into_git(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps({"schema_version": "dbt1-evidence-manifest-v1", "clips": [
            {"clip_id": "fixture_part_00000001", "source_id": "fixture", "part": 1}]}))
        output = self.root / "output.json"
        command = [sys.executable, "-B", "-m", "dbmetrics.legacy", "--manifest", str(manifest),
                   "--media-root", str(self.root), "--media-kind", "original", "--output", str(output)]
        for flags in (["--vlm"], ["--local-vlm-dir", str(self.model_dir)],
                      ["--vlm", "--local-vlm-dir", str(self.model_dir), "--model-revision", REVISION]):
            result = subprocess.run(command + flags, capture_output=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
        repository = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            dbmetrics.legacy.extract_pilot(manifest, self.root, repository / "forbidden-vlm-output.json", "original",
                vlm=True, local_vlm_dir=self.model_dir, model_revision=REVISION, device="cuda:0")

    def test_module_and_cli_help_do_not_import_ml_packages(self):
        command = [sys.executable, "-B", "-c", "import sys, dbmetrics._vlm, dbmetrics.legacy; "
                   "assert not {'torch', 'transformers', 'PIL'} & set(sys.modules)"]
        result = subprocess.run(command, capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", "-m", "dbmetrics.legacy", "--help"],
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b"--local-vlm-dir", result.stdout)
        self.assertIn(b"--device", result.stdout)


if __name__ == "__main__":
    unittest.main()
