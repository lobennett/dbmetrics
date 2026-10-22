"""Explicit model identity survives the extractor's worker boundary."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import dbmetrics.legacy
from dbmetrics._vlm import QWEN3_MODEL_ID, QWEN3_REVISION
import test_compat_pilot_extract as support


class PilotVlmCliTests(unittest.TestCase):
    def setUp(self):
        self.fixture = support.PilotExtractTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_selected_model_requires_explicit_vlm_stage(self):
        with self.assertRaises(ValueError):
            self.fixture.run_extract(vlm_model_id=QWEN3_MODEL_ID)
        self.assertFalse(self.fixture.output.exists())

    def test_qwen3_identity_reaches_worker_without_loading_model(self):
        captured = []

        def worker(command, *args, **kwargs):
            captured.append(json.loads(command[-1]))
            return json.dumps({"records": [], "failed": False}).encode(), b""

        with patch("dbmetrics._vlm.snapshot_path", return_value=Path(self.fixture.root)) as preflight:
            with patch.object(dbmetrics.legacy, "run_bounded", side_effect=worker):
                self.assertEqual(self.fixture.run_extract(stage="vlm", vlm=True, vlm_model_id=QWEN3_MODEL_ID,
                                                         model_revision=QWEN3_REVISION,
                                                         local_vlm_dir=self.fixture.root, device="cuda:0"), 0)
        self.assertEqual(preflight.call_args.kwargs["model_id"], QWEN3_MODEL_ID)
        self.assertEqual(captured[0]["vlm_options"]["model_id"], QWEN3_MODEL_ID)
        self.assertEqual(captured[0]["vlm_options"]["revision"], QWEN3_REVISION)

    def test_descriptive_contract_reports_explicit_model_and_rejects_unknown(self):
        contract = dbmetrics.legacy.vlm_adapter_contract(self.fixture.root, QWEN3_REVISION, model_id=QWEN3_MODEL_ID)
        self.assertEqual(contract["model_id"], QWEN3_MODEL_ID)
        self.assertFalse(contract["execution_enabled"])
        with self.assertRaises(ValueError):
            dbmetrics.legacy.vlm_adapter_contract(self.fixture.root, QWEN3_REVISION, model_id="unknown/model")


if __name__ == "__main__":
    unittest.main()
