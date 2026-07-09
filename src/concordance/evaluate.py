"""
Evaluation: quantify what the conformal triage layer buys you over a naive
"auto-accept the top-1 match" strategy (the Usagi-style default).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .conformal import (
    PredictionSet,
    build_prediction_sets,
    calibrate_threshold,
    empirical_coverage,
)


@dataclass
class TriageMetrics:
    alpha: float
    n_test: int
    empirical_coverage: float       # should be >= 1 - alpha
    avg_set_size: float             # efficiency (smaller is better)
    auto_accept_rate: float         # fraction sent straight through
    auto_accept_accuracy: float     # accuracy among auto-accepted
    review_rate: float              # fraction flagged as ambiguous
    no_match_rate: float            # fraction with empty set
    naive_top1_accuracy: float      # accept-everything baseline accuracy


def _accuracy(pred_labels: list[int], true_labels: list[int]) -> float:
    if not pred_labels:
        return float("nan")
    return float(np.mean([p == y for p, y in zip(pred_labels, true_labels)]))


def evaluate_single_split(
    probs_cal: np.ndarray, true_cal: np.ndarray,
    probs_test: np.ndarray, true_test: list[int],
    label_ids: list[int], alpha: float,
) -> tuple[TriageMetrics, list[PredictionSet]]:
    q_hat = calibrate_threshold(probs_cal, true_cal, alpha=alpha)
    sets = build_prediction_sets(probs_test, q_hat, label_ids)

    cover = empirical_coverage(sets, true_test)
    sizes = [len(s.member_labels) for s in sets]

    auto = [s for s in sets if s.decision == "AUTO_ACCEPT"]
    auto_idx = [i for i, s in enumerate(sets) if s.decision == "AUTO_ACCEPT"]
    auto_acc = _accuracy([s.top_label for s in auto],
                         [true_test[i] for i in auto_idx])

    naive_top1 = _accuracy([s.top_label for s in sets], true_test)

    m = TriageMetrics(
        alpha=alpha,
        n_test=len(sets),
        empirical_coverage=round(cover, 4),
        avg_set_size=round(float(np.mean(sizes)), 3),
        auto_accept_rate=round(len(auto) / len(sets), 4),
        auto_accept_accuracy=round(auto_acc, 4) if auto else float("nan"),
        review_rate=round(sum(s.decision == "REVIEW" for s in sets) / len(sets), 4),
        no_match_rate=round(sum(s.decision == "NO_MATCH" for s in sets) / len(sets), 4),
        naive_top1_accuracy=round(naive_top1, 4),
    )
    return m, sets


def repeated_split_coverage(
    probs: np.ndarray, true_idx: np.ndarray, label_ids: list[int],
    alpha: float, n_repeats: int = 200, cal_frac: float = 0.5, seed: int = 0,
) -> dict:
    """Average empirical coverage over many random calibration/test splits.

    A single split of a small dataset gives a noisy coverage estimate; averaging
    over many splits shows the guarantee holds on average (>= 1 - alpha).
    """
    rng = np.random.default_rng(seed)
    n = len(true_idx)
    true_labels_all = [label_ids[j] for j in true_idx]
    covers, sizes = [], []
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        k = int(n * cal_frac)
        cal, test = perm[:k], perm[k:]
        q_hat = calibrate_threshold(probs[cal], true_idx[cal], alpha=alpha)
        sets = build_prediction_sets(probs[test], q_hat, label_ids)
        covers.append(empirical_coverage(sets, [true_labels_all[i] for i in test]))
        sizes.append(np.mean([len(s.member_labels) for s in sets]))
    return {
        "alpha": alpha,
        "target_coverage": round(1 - alpha, 4),
        "mean_empirical_coverage": round(float(np.mean(covers)), 4),
        "std_empirical_coverage": round(float(np.std(covers)), 4),
        "mean_avg_set_size": round(float(np.mean(sizes)), 3),
        "n_repeats": n_repeats,
    }


def metrics_to_dict(m: TriageMetrics) -> dict:
    return asdict(m)
