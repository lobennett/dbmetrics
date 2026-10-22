import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SimilarityComparisonsTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "notebooks" / "exploratory-metrics.ipynb"
        self.cells = {cell["id"]: "".join(cell["source"])
                      for cell in json.loads(path.read_text())["cells"]}
        self.addCleanup(plt.close, "all")

    def test_cut_errors_remain_unknown_in_the_comparison(self):
        continuity = {"coverage": {"first_timestamp_seconds": 5, "kind": "full"},
                      "trace": [{"timestamp_seconds": 5.1, "change": 0.2}]}
        failed = {"value": None, "processing_status": "error", "evidence": [{"reason": "decode_failed"}]}
        namespace = dict(VIDEO_PATH=Path("demo.mp4"), config=type("Config", (), {
            "ffmpeg": "ffmpeg", "ffprobe": "ffprobe"})(), plt=plt,
            embedding_changes=[], change_times=[], manual_boundaries=[],
            decoded={"duration_seconds": 10})
        with patch("dbmetrics.experimental.scene_continuity.measure", return_value=continuity), \
             patch("dbmetrics._cuts.detect_ffmpeg_cuts", return_value=failed), \
             patch("subprocess.run") as probe:
            probe.return_value.stdout = "{}"
            exec(self.cells["continuity-comparison"], namespace)
        self.assertIsNone(namespace["cut_times"])
        self.assertEqual(namespace["pixel_times"], [5.1 - 5])
        with patch.dict(namespace, run_bounded=lambda command, *args, **kwargs: command):
            command = ["ffmpeg", "-vsync", "0", "-f", "null", "pipe:1"]
            converted = namespace["run_cut_process"](command)
            self.assertEqual(converted, ["ffmpeg", "-fps_mode", "passthrough", "-f", "null", "pipe:1"])
            self.assertEqual(command[1:3], ["-vsync", "0"])

    def test_caption_comparison_is_optional_and_requires_half_descriptions(self):
        namespace = {}
        exec(self.cells["half-caption-settings"], namespace)
        exec(self.cells["half-caption-comparison"], namespace)
        self.assertIsNone(namespace["half_caption_similarity"])
        namespace["RUN_CAPTION_SIMILARITY"] = True
        with self.assertRaisesRegex(ValueError, "each half"):
            exec(self.cells["half-caption-comparison"], namespace)
