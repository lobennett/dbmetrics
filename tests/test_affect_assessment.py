import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import dbp_metrics.experimental.affect as affect


ROOT = Path(__file__).resolve().parents[1]


def review(**changes):
    row = dict(video_id="video-a", reviewer_id="reviewer-1", target="viewer_experienced",
               dimension="valence", rating="5", scale_min="1", scale_max="9",
               recorded_at="2026-09-29T12:00:00Z", provenance="pilot-v1; full clip with audio",
               split="heldout")
    row.update(changes)
    return row


def prediction(**changes):
    row = review()
    del row["reviewer_id"]
    del row["split"]
    row["model_id"] = "experimental-model@revision;prompt-v1"
    row.update(changes)
    return row


def encoded(rows, kind="reviews"):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=affect.FIELDS[kind])
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    return stream


def parsed(rows, kind="reviews"):
    return affect.read_csv(encoded(rows, kind), kind)


class ValidationTests(unittest.TestCase):
    def test_nullable_and_bounds_preserve_provenance(self):
        for value in ("", "1", "9"):
            result = parsed([review(rating=value)])[0]
            self.assertEqual(result["rating"], None if value == "" else float(value))
            self.assertEqual(result["provenance"], review()["provenance"])

    def test_rejects_bad_ratings_scales_metadata_and_constructs(self):
        for changes in (
            {"rating": "NaN"}, {"rating": "inf"}, {"rating": "0"}, {"rating": "10"},
            {"rating": "3.5"}, {"rating": "null"}, {"scale_min": "0"},
            {"scale_max": "100"}, {"reviewer_id": " "}, {"video_id": ""},
            {"provenance": ""}, {"recorded_at": "2026-09-29"},
            {"recorded_at": "2026-09-29T12:00:00"}, {"recorded_at": "invalid"},
            {"target": "emotion"}, {"dimension": "views"}, {"split": "test"},
            {"target": "stimulus_depicted", "dimension": "engagement"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parsed([review(**changes)])

    def test_engagement_scale_and_continuous_predictions(self):
        self.assertEqual(parsed([review(dimension="engagement", rating="9")])[0]["rating"], 9)
        self.assertEqual(parsed([prediction(rating="3.5")], "predictions")[0]["rating"], 3.5)

    def test_engagement_is_human_only(self):
        with self.assertRaisesRegex(ValueError, "human-only"):
            parsed([prediction(dimension="engagement")], "predictions")

    def test_all_ui_fields_have_nullable_integer_one_to_nine_scales(self):
        self.assertEqual(set(affect.UI_FIELDS), {"depicted_valence", "depicted_arousal",
                                              "experienced_valence", "experienced_arousal", "engagement"})
        for target, dimension in affect.UI_FIELDS.values():
            for value in ("", "1", "5", "9"):
                with self.subTest(target=target, dimension=dimension, value=value):
                    result = parsed([review(target=target, dimension=dimension, rating=value)])[0]
                    self.assertEqual(result["rating"], int(value) if value else None)
                    if value:
                        self.assertIsInstance(result["rating"], int)

    def test_prediction_bounds_and_missing_identity(self):
        for changes in ({"rating": "NaN"}, {"rating": "9.1"}, {"rating": "0"},
                        {"model_id": ""}, {"scale_max": "5"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                parsed([prediction(**changes)], "predictions")

    def test_duplicate_reviews_and_predictions_rejected(self):
        for kind, row in (("reviews", review()), ("predictions", prediction())):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "duplicate"):
                parsed([row, row], kind)

    def test_video_cannot_cross_splits_even_across_constructs(self):
        with self.assertRaisesRegex(ValueError, "split"):
            parsed([review(), review(reviewer_id="other", dimension="arousal", split="train")])

    def test_headers_and_row_width_strict(self):
        header = encoded([]).getvalue()
        for payload in ("", "video_id\nfoo\n", header.replace("rating,", "rating,rating,"),
                        header + "only-one-cell\n", encoded([review()]).getvalue().rstrip() + ",extra\n"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                affect.read_csv(io.StringIO(payload))


class SummaryTests(unittest.TestCase):
    def test_disagreement_missingness_and_separate_targets(self):
        rows = parsed([review(rating="1"), review(reviewer_id="two", rating="9"),
                       review(reviewer_id="three", rating=""),
                       review(target="stimulus_depicted", rating="7")])
        result = affect.summarize(rows)
        experienced = next(item for item in result if item["target"] == "viewer_experienced")
        self.assertEqual(experienced["n_reviews"], 3)
        self.assertEqual(experienced["n_ratings"], 2)
        self.assertEqual(experienced["n_missing"], 1)
        self.assertEqual(experienced["n_reviewers"], 3)
        self.assertEqual(experienced["mean"], 5)
        self.assertEqual(experienced["median"], 5)
        self.assertEqual(experienced["range"], 8)
        self.assertEqual(experienced["mean_pairwise_absolute_difference"], 8)
        self.assertAlmostEqual(experienced["sample_sd"], 32 ** 0.5)
        self.assertEqual(len(result), 2)

    def test_single_and_all_missing_not_false_agreement(self):
        for value in ("", "5"):
            result = affect.summarize(parsed([review(rating=value)]))[0]
            self.assertIsNone(result["sample_sd"])
            self.assertIsNone(result["mean_pairwise_absolute_difference"])
            self.assertEqual(result["mean"], None if value == "" else 5)

    def test_pairwise_disagreement_and_input_order(self):
        rows = parsed([review(reviewer_id=str(index), rating=str(value))
                       for index, value in enumerate((1, 3, 8))])
        summaries = affect.summarize(rows)
        self.assertAlmostEqual(summaries[0]["mean_pairwise_absolute_difference"], 14 / 3)
        self.assertEqual(summaries, affect.summarize(list(reversed(rows))))


class EvaluationTests(unittest.TestCase):
    def test_heldout_equal_video_weight_counts_and_unmatched(self):
        reviews = parsed([review(rating="1"), review(reviewer_id="two", rating="3"),
                          review(video_id="video-b", rating="8"),
                          review(video_id="train", rating="9", split="train"),
                          review(video_id="missing", rating=""), review(video_id="no-prediction")])
        predictions = parsed([prediction(rating="3"), prediction(video_id="video-b", rating="6"),
                              prediction(video_id="train"), prediction(video_id="unknown")], "predictions")
        result = affect.evaluate(reviews, predictions)
        metric = result["metrics"][0]
        self.assertEqual(metric["mae"], 1.5)
        self.assertAlmostEqual(metric["spearman"], 1)
        self.assertEqual(metric["n_pairs"], 2)
        self.assertEqual(metric["n_human_ratings"], 3)
        self.assertEqual(metric["n_heldout_videos"], 4)
        self.assertEqual(metric["n_rated_heldout_videos"], 3)
        self.assertEqual(metric["n_unpaired_heldout_videos"], 2)
        self.assertEqual(metric["n_predictions_outside_heldout"], 2)
        self.assertNotIn("confidence", metric)

    def test_ties_and_degenerate_spearman(self):
        self.assertAlmostEqual(affect.spearman([1, 1, 3], [1, 2, 3]), 3 ** 0.5 / 2)
        self.assertAlmostEqual(affect.spearman([1, 2, 3], [3, 2, 1]), -1)
        for left, right in (([], []), ([1], [2]), ([1, 1], [1, 2])):
            self.assertIsNone(affect.spearman(left, right))
        with self.assertRaises(ValueError):
            affect.spearman([1], [])

    def test_no_pairs_has_null_metrics_and_reason(self):
        result = affect.evaluate(parsed([review()]), parsed([prediction(rating="")], "predictions"))
        metric = result["metrics"][0]
        self.assertEqual(metric["n_pairs"], 0)
        self.assertIsNone(metric["mae"])
        self.assertIsNone(metric["spearman"])
        self.assertTrue(metric["spearman_unavailable_reason"])

    def test_no_heldout_or_no_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError, "heldout"):
            affect.evaluate(parsed([review(split="train")]), parsed([prediction()], "predictions"))
        with self.assertRaisesRegex(ValueError, "prediction"):
            affect.evaluate(parsed([review()]), [])

    def test_model_and_construct_separation(self):
        reviews = parsed([review(), review(target="stimulus_depicted", rating="9")])
        predictions = parsed([prediction(), prediction(model_id="second", rating="2"),
                              prediction(target="stimulus_depicted", rating="8")], "predictions")
        metrics = affect.evaluate(reviews, predictions)["metrics"]
        self.assertEqual(len(metrics), 4)
        self.assertEqual(sorted(item["mae"] for item in metrics if item["mae"] is not None), [0, 1, 3])


class CLITests(unittest.TestCase):
    def cli(self, *arguments):
        return subprocess.run([sys.executable, "-m", "dbp_metrics.cli.affect", *arguments],
                              cwd=tempfile.gettempdir(), text=True, capture_output=True, check=False)

    def test_templates_and_prompt_export_no_inference(self):
        for kind in affect.FIELDS:
            result = self.cli("template", "--kind", kind)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(next(csv.reader(io.StringIO(result.stdout))), list(affect.FIELDS[kind]))
        result = self.cli("export-prompts", "--video-id", "demo-video")
        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["inference_performed"])
        self.assertEqual(payload["jobs"][0]["video_id"], "demo-video")
        self.assertIn("hypotheses", payload["jobs"][0]["prompt"])
        self.assertIn("null", payload["jobs"][0]["prompt"])
        self.assertEqual(self.cli("export-prompts", "--video-id", " ").returncode, 2)

    def test_validation_summary_evaluation_and_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            reviews = Path(directory) / "reviews.csv"
            predictions = Path(directory) / "predictions.csv"
            reviews.write_text(encoded([review()]).getvalue(), encoding="utf-8")
            predictions.write_text(encoded([prediction()], "predictions").getvalue(), encoding="utf-8")
            for arguments in (("validate", str(reviews)), ("summarize", str(reviews)),
                              ("import-reviews", str(reviews)),
                              ("validate", str(predictions), "--kind", "predictions"),
                              ("evaluate", str(reviews), str(predictions))):
                result = self.cli(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)
                self.assertEqual(len(report["inputs"][0]["sha256"]), 64)
                if arguments[0] == "import-reviews":
                    self.assertEqual(report["reviews"][0]["provenance"], review()["provenance"])
            reviews.write_text(encoded([review(rating="100")]).getvalue(), encoding="utf-8")
            result = self.cli("validate", str(reviews))
            self.assertEqual(result.returncode, 2)
            self.assertIn("row 2", result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(self.cli("validate", str(reviews) + ".missing").returncode, 2)

    def test_bom_and_encoding_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            path.write_text(encoded([review()]).getvalue(), encoding="utf-8-sig")
            self.assertEqual(self.cli("validate", str(path)).returncode, 0)
            path.write_bytes(b"\xff\xfe")
            result = self.cli("validate", str(path))
            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
