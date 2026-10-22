import unittest

import numpy as np

from dbmetrics.experimental.candidate_metrics import transition_candidates, window_scores, validate_hypothesis, compare_boundaries


class CandidateMetricsTests(unittest.TestCase):
    def test_boundary_comparison_is_one_to_one_and_not_reviewed_is_unknown(self):
        self.assertIsNone(compare_boundaries([1], None))
        result = compare_boundaries([1, 1.1, 4], [1.05, 7], .2)
        self.assertEqual((result['matched'], result['extra'], result['missed']), (1, 2, 1))
        self.assertEqual(result['precision'], 1 / 3)
        self.assertEqual(result['recall'], .5)
    def test_adjacent_transition_frames_form_one_candidate(self):
        rows = transition_candidates([0, .6, .9, 0, .8], [0, .1, .2, .3, .4], .5)
        self.assertEqual([row['time_seconds'] for row in rows], [.2, .4])
        self.assertEqual(rows[0]['start_seconds'], .1)

    def test_incomplete_or_invalid_scores_are_rejected(self):
        for scores, times in (([.2], [0, 1]), ([float('nan')], [0]), ([1.1], [0])):
            with self.assertRaises(ValueError):
                transition_candidates(scores, times)

    def test_windows_keep_only_the_center_50_and_restore_length(self):
        frames = np.arange(53, dtype=np.uint8).reshape(53, 1, 1, 1)
        windows = []

        def predict(window):
            windows.append(window)
            return window[:, 0, 0, 0] / 100

        scores = window_scores(frames, predict)
        self.assertEqual(len(windows), 2)
        self.assertEqual([len(window) for window in windows], [100, 100])
        np.testing.assert_allclose(scores, np.arange(53) / 100)

    def test_affect_remains_unknown_and_engagement_is_only_a_proxy(self):
        record = dict(depicted_valence=None, depicted_arousal=5, evidence='calm movement', limitations='no audio')
        self.assertIsNone(validate_hypothesis(record, 'affect')['depicted_valence'])
        with self.assertRaises(ValueError):
            validate_hypothesis({**record, 'depicted_valence': 10}, 'affect')
        with self.assertRaises(ValueError):
            validate_hypothesis(dict(engagement=7, evidence='', limitations=''), 'interest')

    def test_event_types_are_separate_and_uncertainty_is_retained(self):
        changes = dict(place='no', time='uncertain', entities='no', action='yes', goal='uncertain')
        record = dict(changes=changes, evidence='picking becomes carrying', limitations='goal inferred')
        self.assertEqual(validate_hypothesis(record, 'event')['changes']['goal'], 'uncertain')
        with self.assertRaises(ValueError):
            validate_hypothesis({**record, 'changes': dict(changes, action='probably')}, 'event')
