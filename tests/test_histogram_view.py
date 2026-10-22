import importlib.util
import json
from pathlib import Path
import unittest

from dbmetrics.experimental.half_similarity import histogram


class HistogramViewTests(unittest.TestCase):
    def test_frame_counts_sum_to_the_existing_half_histogram(self):
        path = Path(__file__).parents[1] / "src/dbmetrics/experimental/histogram_view.py"
        spec = importlib.util.spec_from_file_location("histogram_view", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        frames = [bytes([0, 127, 255]) * (256 * 256),
                  bytes([255, 127, 0]) * (256 * 256)]
        half = dict(frames=frames, timestamps_seconds=[0.1, 0.9])
        data = module.histogram_data(dict(halves=[half, half]), 16)
        counts = data["halves"][0]["counts"]
        self.assertEqual([sum(values) for values in zip(*counts)], histogram(frames, 16))
        self.assertEqual(counts[0][0], 65536)
        self.assertEqual(counts[0][16 + 7], 65536)
        self.assertEqual(counts[0][32 + 15], 65536)
        self.assertAlmostEqual(data["score"], 1.0)
        self.assertEqual(data["halves"][0]["times"], [0.1, 0.9])
        page = module.histogram_html(dict(halves=[half, half]), 16)
        self.assertIn('id="mode"', page)
        self.assertIn('id="sample"', page)
        self.assertIn("not confidence", page)

    def test_notebook_uses_the_interactive_view(self):
        path = Path(__file__).parents[1] / "notebooks/exploratory-metrics.ipynb"
        notebook = json.loads(path.read_text())
        self.assertIn("show_histograms", "".join(notebook["cells"][6]["source"]))


if __name__ == "__main__":
    unittest.main()
