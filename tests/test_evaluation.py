import unittest
from dbp_metrics.evaluation import score_target, target_label


class EvaluationTests(unittest.TestCase):
    def test_counts_and_denominators_exclude_unknown(self):
        examples = [('present', 'present'), ('absent', 'present'),
                    ('absent', 'absent'), ('present', 'absent'), ('uncertain', 'present')]
        rows = [{'clip_id': str(index), 'review': {
                    'labels': {'added_text': truth}, 'watched_full': True},
                 'baseline': predicted}
                for index, (truth, predicted) in enumerate(examples)]
        result = score_target(rows, 'added_text', 'baseline')
        self.assertEqual(result['counts'], dict(tp=1, fp=1, tn=1, fn=1))
        self.assertEqual(result['denominators']['scored'], 4)
        self.assertEqual(result['excluded_reasons'], {'human_uncertain': 1})
        self.assertEqual(result['precision'], 0.5)

    def test_union_does_not_convert_missing_to_absence(self):
        self.assertIsNone(target_label({'added_text': 'absent'}, 'added_combined'))
