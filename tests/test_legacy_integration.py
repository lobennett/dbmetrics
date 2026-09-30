import importlib
import unittest


class LegacyIntegrationTests(unittest.TestCase):
    def test_preserved_worker_modules_are_packaged(self):
        for name in ('legacy', '_vlm', '_semantics', 'batch'):
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module('dbp_metrics.' + name))

    def test_vlm_and_overlay_share_snapshot_validation(self):
        from dbp_metrics import _local_model, _vlm
        self.assertIs(_vlm.validate_local_model, _local_model.validate_local_model)

    def test_cpu_compatibility_worker_uses_package_computations(self):
        from dbp_metrics import legacy
        self.assertIn('dbp_metrics._visual',
                      __import__('inspect').getsource(legacy.extract_clip).replace('from .', 'from dbp_metrics.'))
