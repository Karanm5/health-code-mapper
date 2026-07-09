"""Tests for the conformal layer: coverage guarantee, triage, monotonicity."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from concordance.conformal import (          # noqa: E402
    build_prediction_sets,
    calibrate_threshold,
    empirical_coverage,
)


def _synthetic_probs(n, k, rng, sharpness=6.0):
    """Make a well-calibrated-ish classifier: true label gets higher logits."""
    true = rng.integers(0, k, size=n)
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), true] += sharpness * rng.random(n)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    return probs, true


def test_marginal_coverage_holds():
    rng = np.random.default_rng(0)
    k = 8
    label_ids = list(range(k))
    alpha = 0.1
    covers = []
    for _ in range(300):
        probs, true = _synthetic_probs(400, k, rng)
        cal, test = slice(0, 200), slice(200, 400)
        q = calibrate_threshold(probs[cal], true[cal], alpha=alpha)
        sets = build_prediction_sets(probs[test], q, label_ids)
        covers.append(empirical_coverage(sets, [label_ids[y] for y in true[test]]))
    # Average coverage should meet or exceed the 1 - alpha target.
    assert np.mean(covers) >= (1 - alpha) - 0.02


def test_triage_decisions():
    label_ids = [10, 20, 30]
    # Row A: one candidate clears bar -> AUTO_ACCEPT
    # Row B: two candidates clear bar -> REVIEW
    # Row C: none clear bar -> NO_MATCH
    probs = np.array([
        [0.95, 0.03, 0.02],
        [0.50, 0.45, 0.05],
        [0.35, 0.34, 0.31],
    ])
    q_hat = 0.6  # keep any label with prob >= 1 - q_hat = 0.4
    sets = build_prediction_sets(probs, q_hat, label_ids)
    assert sets[0].decision == "AUTO_ACCEPT" and sets[0].member_labels == [10]
    assert sets[1].decision == "REVIEW" and set(sets[1].member_labels) == {10, 20}
    assert sets[2].decision == "NO_MATCH" and sets[2].member_labels == []


def test_lower_alpha_gives_larger_sets():
    rng = np.random.default_rng(1)
    k = 8
    label_ids = list(range(k))
    probs, true = _synthetic_probs(600, k, rng)
    cal, test = slice(0, 300), slice(300, 600)

    def avg_size(alpha):
        q = calibrate_threshold(probs[cal], true[cal], alpha=alpha)
        sets = build_prediction_sets(probs[test], q, label_ids)
        return np.mean([len(s.member_labels) for s in sets])

    assert avg_size(0.05) >= avg_size(0.20)
