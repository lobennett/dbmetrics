import importlib.util
from pathlib import Path
import unittest


class SplitSampleViewTests(unittest.TestCase):
    def test_view_uses_the_actual_sampling_evidence(self):
        if importlib.util.find_spec("PIL") is None:
            self.skipTest("Notebook Pillow dependency is not installed")
        path = Path(__file__).parents[1] / "src/dbp_metrics/experimental/split_sample_view.py"
        spec = importlib.util.spec_from_file_location("split_sample_view", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        half = dict(frames=[bytes(256 * 256 * 3)], requested_seconds=[0.5],
                    timestamps_seconds=[0.48], frame_indices=[12])
        page = module.split_sample_html(dict(duration_seconds=2, split_seconds=1, halves=[half, half]))
        self.assertIn('"requested": [0.5]', page)
        self.assertIn('"actual": [0.48]', page)
        self.assertIn('"indices": [12]', page)
        self.assertIn("data:image/jpeg;base64,", page)
        self.assertIn('type="range"', page)
        self.assertIn("Explain it back", page)

    def test_notebook_keeps_images_out_of_saved_outputs(self):
        import json

        path = Path(__file__).parents[1] / "notebooks/exploratory-metrics.ipynb"
        notebook = json.loads(path.read_text())
        self.assertTrue(all(not cell.get("outputs") for cell in notebook["cells"]))
        source = "".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("half-similarity-demo.mp4", source)
        self.assertNotIn("print(decoded)", source)


if __name__ == "__main__":
    unittest.main()
