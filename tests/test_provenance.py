import unittest
import sys
from dbp_metrics.provenance import code_hashes, verify_code, verify_interpreter


class ProvenanceTests(unittest.TestCase):
    def test_binds_all_computation_modules(self):
        hashes = code_hashes()
        for name in ('legacy.py', '_vlm.py', '_local_model.py', '_ocr.py',
                     '_technical.py', 'overlay.py', 'captions.py',
                     'experimental/half_similarity.py', 'cli/research.py'):
            self.assertEqual(len(hashes[name]), 64)
        verify_code(hashes)

    def test_rejects_changed_or_missing_module(self):
        hashes = code_hashes()
        hashes.pop('captions.py')
        with self.assertRaisesRegex(ValueError, 'new run'):
            verify_code(hashes)

    def test_worker_interpreter_has_matching_package(self):
        verify_interpreter(sys.executable, code_hashes())
