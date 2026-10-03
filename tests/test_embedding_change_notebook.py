import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from dbp_metrics.experimental.half_similarity import Config, cosine


class EmbeddingChangeNotebookTests(unittest.TestCase):
    def setUp(self):
        notebook = Path(__file__).parents[1] / "notebooks" / "exploratory-metrics.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        self.sources = {cell["id"]: "".join(cell["source"]) for cell in cells}

    def test_neighboring_windows_use_64_frames_and_midpoint_times(self):
        intervals = []
        vectors = iter(([1, 0], [1, 0], [0, 1]))

        def decode(path, config, *, interval):
            self.assertEqual(config.frames_per_half, 32)
            intervals.append(interval)
            return {"halves": [{"frames": [b"frame"] * 32}, {"frames": [b"frame"] * 32}]}

        def embed(frames):
            self.assertEqual(len(frames), 64)
            return next(vectors)

        axis = SimpleNamespace(**{name: lambda *args, **kwargs: None
                                  for name in ("plot", "axvline", "set", "set_ylim", "legend")})
        plotting = SimpleNamespace(subplots=lambda **kwargs: (None, axis), show=lambda: None)
        namespace = dict(np=np, decoded={"duration_seconds": 4}, Config=Config, cosine=cosine,
                         VIDEO_PATH="demo.mp4", DEVICE="cpu", RUN_VJEPA=True,
                         backend=SimpleNamespace(embed=embed), decode_media=decode, plt=plotting)
        exec(self.sources["embedding-change-settings"], namespace)
        namespace["RUN_EMBEDDING_CHANGE"] = True
        exec(self.sources["embedding-change-computation"], namespace)
        self.assertEqual(intervals, [(0, 2), (1, 3), (2, 4)])
        self.assertEqual(namespace["change_times"], [1.5, 2.5])
        self.assertEqual(namespace["embedding_changes"], [0, 1])

    def test_disabled_run_does_not_decode_or_infer(self):
        namespace = dict(np=np, decoded={"duration_seconds": 4})
        exec(self.sources["embedding-change-settings"], namespace)
        exec(self.sources["embedding-change-computation"], namespace)
        self.assertEqual(namespace["embedding_changes"], [])
