"""Small notebook helpers; not website metrics or a production model pipeline."""

import numpy as np


def transition_candidates(scores, timestamps, threshold=.5):
    """Group consecutive above-threshold frames, keeping each group's peak."""
    scores = np.asarray(scores, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    if (scores.ndim != 1 or len(scores) == 0 or timestamps.shape != scores.shape
            or not np.isfinite(scores).all() or not np.isfinite(timestamps).all()
            or np.any((scores < 0) | (scores > 1)) or np.any(np.diff(timestamps) <= 0)
            or not 0 <= threshold <= 1):
        raise ValueError('Expected aligned scores in 0–1 and increasing frame timestamps')
    selected = np.flatnonzero(scores > threshold)
    groups = np.split(selected, np.flatnonzero(np.diff(selected) > 1) + 1)
    result = []
    for group in groups:
        if not len(group):
            continue
        peak = int(group[np.argmax(scores[group])])
        result.append(dict(frame_index=peak, time_seconds=float(timestamps[peak]),
                           start_seconds=float(timestamps[group[0]]),
                           end_seconds=float(timestamps[group[-1]]), score=float(scores[peak])))
    return result


def window_scores(frames, predict):
    """TransNet inference: 100-frame windows, retaining the central 50 scores."""
    if len(frames) == 0:
        raise ValueError('No frames')
    ending = 25 + (-len(frames) % 50)
    padded = np.concatenate([np.repeat(frames[:1], 25, axis=0), frames,
                             np.repeat(frames[-1:], ending, axis=0)])
    retained = []
    for start in range(0, len(padded) - 99, 50):
        scores = np.asarray(predict(padded[start:start + 100]), dtype=float)
        if scores.shape != (100,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError('TransNet must return 100 finite scores in 0–1')
        retained.append(scores[25:75])
    return np.concatenate(retained)[:len(frames)]


def compare_boundaries(predicted, reviewed, tolerance=.5):
    """Match chronologically, at most once per reviewed and predicted boundary."""
    if predicted is None or reviewed is None:
        return None
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Use a nonnegative timestamp tolerance')
    predicted, reviewed = sorted(predicted), sorted(reviewed)
    if any(not np.isfinite(value) or value < 0 for value in predicted + reviewed):
        raise ValueError('Boundary timestamps must be finite and nonnegative')
    prediction_index = review_index = matched = 0
    while prediction_index < len(predicted) and review_index < len(reviewed):
        distance = predicted[prediction_index] - reviewed[review_index]
        if abs(distance) <= tolerance:
            matched += 1
            prediction_index += 1
            review_index += 1
        elif distance < 0:
            prediction_index += 1
        else:
            review_index += 1
    return dict(matched=matched, extra=len(predicted) - matched, missed=len(reviewed) - matched,
                precision=matched / len(predicted) if predicted else None,
                recall=matched / len(reviewed) if reviewed else None, tolerance_seconds=tolerance)


def validate_hypothesis(record, kind):
    """Validate output structure and ranges, not whether a model is correct."""
    fields = {'affect': {'depicted_valence', 'depicted_arousal'},
              'interest': {'anticipated_interest'}, 'event': {'changes'}}[kind]
    if type(record) is not dict or set(record) != fields | {'evidence', 'limitations'}:
        raise ValueError('Unexpected model fields')
    for field in ('evidence', 'limitations'):
        if type(record[field]) is not str or len(record[field]) > 1000:
            raise ValueError('Use short evidence and limitations strings')
    if kind == 'event':
        changes = record['changes']
        if (type(changes) is not dict or set(changes) != {'place', 'time', 'entities', 'action', 'goal'}
                or any(value not in ('yes', 'no', 'uncertain') for value in changes.values())):
            raise ValueError('Expected separate yes/no/uncertain event-change types')
    else:
        for field in fields:
            value = record[field]
            if value is not None and (type(value) not in (int, float)
                                      or not np.isfinite(value) or not 1 <= value <= 9):
                raise ValueError('Ratings must be 1–9 or null')
    return record
