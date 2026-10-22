"""Classification counts with explicit missing/uncertain exclusions."""
from collections import Counter
from .evidence import FIELDS

COMBINED = {"added_combined": ("added_text", "added_logo"), "any_text": ("added_text", "scene_text")}


def target_label(labels, target):
    if target in FIELDS:
        return labels.get(target)
    values = [labels.get(field) for field in COMBINED[target]]
    if "present" in values:
        return "present"
    if all(value == "absent" for value in values):
        return "absent"
    return None if None in values else "uncertain"


def score_target(clips, target, method, label_key="labels"):
    counts = dict.fromkeys(("tp", "fp", "tn", "fn"), 0)
    exclusions = Counter()
    decisions = []
    for clip in clips:
        review = clip["review"]
        truth = target_label(review.get("labels", {}), target) if review is not None else None
        prediction = clip[method]
        predicted = None
        if method == "baseline":
            predicted = prediction
        elif prediction is not None and prediction["processing_status"] == "complete":
            predicted = target_label(prediction.get(label_key, {}), target)
        reason = None
        if truth is None:
            reason = "human_missing"
        elif truth == "uncertain":
            reason = "human_uncertain"
        elif review.get("watched_full") is not True:
            reason = "human_not_full_watched"
        elif prediction is None:
            reason = "prediction_missing"
        elif method != "baseline" and prediction["processing_status"] == "error":
            reason = "prediction_error"
        elif predicted == "uncertain":
            reason = "prediction_uncertain"
        elif predicted is None:
            reason = "prediction_missing"
        if reason:
            exclusions[reason] += 1
            outcome = None
        else:
            outcome = ("t" if truth == predicted else "f") + ("p" if predicted == "present" else "n")
            counts[outcome] += 1
        decisions.append({"clip_id": clip["clip_id"], "human": truth, "prediction": predicted,
                          "outcome": outcome, "excluded_reason": reason})
    scored = sum(counts.values())
    denominators = {"total": len(clips), "scored": scored, "excluded": len(clips) - scored,
                    "precision": counts["tp"] + counts["fp"], "recall": counts["tp"] + counts["fn"],
                    "specificity": counts["tn"] + counts["fp"]}
    return {"counts": counts, "denominators": denominators, "excluded_reasons": dict(exclusions),
            **{name: numerator / denominators[name] if denominators[name] else None
               for name, numerator in (("precision", counts["tp"]), ("recall", counts["tp"]), ("specificity", counts["tn"]))},
            "decisions": decisions}
