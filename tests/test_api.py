import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


class PackageTests(unittest.TestCase):
    def test_package_exists(self):
        self.assertIsNotNone(importlib.util.find_spec('dbp_metrics'))

    def test_inventory_and_validation(self):
        from dbp_metrics import METRICS, measure_video

        self.assertEqual(len(METRICS), 16)
        for settings in ({'timeout': 0}, {'stages': ['unknown']}, {'cut_threshold': 2}):
            with self.assertRaises(ValueError):
                measure_video('missing.mp4', **settings)

    def test_missing_media_remains_unknown(self):
        from dbp_metrics import measure_video

        records = measure_video('missing.mp4', stages=['probe'])
        self.assertEqual(len(records), 5)
        self.assertTrue(all(record['value'] is None for record in records))
        self.assertTrue(all(record['processing_status'] == 'missing' for record in records))

    def test_media_symlinks_are_rejected(self):
        from dbp_metrics import measure_video

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target.mp4'
            target.write_bytes(b'not a video')
            link = root / 'link.mp4'
            link.symlink_to(target)
            records = measure_video(link, stages=['probe'])
            self.assertTrue(all(record['processing_status'] == 'error' for record in records))
            self.assertTrue(all(record['value'] is None for record in records))


if __name__ == '__main__':
    unittest.main()
