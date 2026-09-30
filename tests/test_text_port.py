import importlib.util
import sys
import unittest


class TextPortTests(unittest.TestCase):
    def test_remaining_computation_modules_exist(self):
        for name in ('ocr', 'overlay', 'captions', 'evidence', '_local_model',
                     'experimental.half_similarity', 'experimental.scene_continuity',
                     'experimental.affect'):
            with self.subTest(module=name):
                try:
                    found = importlib.util.find_spec('dbp_metrics.' + name)
                except ModuleNotFoundError:
                    found = None
                self.assertIsNotNone(found)

    def test_default_overlay_prompt_and_model_are_preserved(self):
        from dbp_metrics.overlay import method_config, model_identity

        self.assertEqual(method_config()['prompt_sha256'],
                         '012f080da28451410c581dfe074e780ae16eeb5f8a213e8323c2c853d49a9c37')
        self.assertEqual(model_identity('8b'),
                         ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'))

    def test_compact_caption_prompt_is_preserved(self):
        from dbp_metrics.captions import PROMPT_SHA256

        self.assertEqual(PROMPT_SHA256,
                         '5ce11554553235b631b3a2617ee6018cb774905e628de3d3cce4f21ddc3998fb')


if __name__ == '__main__':
    unittest.main()
