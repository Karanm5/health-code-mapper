"""
Split (inductive) conformal prediction for vocabulary mapping.

The mapping task is a classification problem: given a source term, choose the
correct standard concept from K candidates. A plain matcher returns the top-1
candidate and hands *everything* to a human, or auto-accepts everything and
makes silent errors. Neither quantifies confidence.

Conformal prediction converts the matcher's scores into prediction *sets* with a
distribution-free coverage guarantee: for a chosen error rate alpha, the true
target is contained in the set with probability >= 1 - alpha (marginally, under
exchangeability of calibration and test data). Vovk et al. (2005);
Angelopoulos & Bates (2021).

We use the LAC / least-ambiguous-set score: nonconformity s(x, y) = 1 - p_hat(y|x).

Triage rule built on the set:
  * |C(x)| == 1  -> AUTO_ACCEPT   (one confident candidate)
  * |C(x)| >= 2  -> REVIEW        (ambiguous; route to a human curator)
  * |C(x)| == 0  -> NO_MATCH      (no candidate clears the bar; likely no valid
                                   target or out-of-distribution code -> flag)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionSet:
    query_index: int
    member_labels: list[int]        # candidate label ids in the set
    top_label: int                  # argmax label (the point prediction)
    top_prob: float                 # its probability
    decision: str                   # AUTO_ACCEPT | REVIEW | NO_MATCH


def calibrate_threshold(
    cal_probs: np.ndarray,      # (n_cal, K) predicted probabilities
    cal_true_idx: np.ndarray,   # (n_cal,) index of the true label within columns
    alpha: float = 0.1,
) -> float:
    """Return q_hat, the conformal threshold on nonconformity scores."""
    n = len(cal_true_idx)
    true_probs = cal_probs[np.arange(n), cal_true_idx]
    scores = 1.0 - true_probs                       # nonconformity of true label
    # Finite-sample corrected quantile level.
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q_hat = float(np.quantile(scores, level, method="higher"))
    return q_hat


def build_prediction_sets(
    probs: np.ndarray,          # (n, K)
    q_hat: float,
    label_ids: list[int],       # maps column index -> label id
) -> list[PredictionSet]:
    """Form conformal sets and triage decisions for each query row."""
    label_ids = list(label_ids)
    out: list[PredictionSet] = []
    for i, row in enumerate(probs):
        members = [label_ids[j] for j, p in enumerate(row) if (1.0 - p) <= q_hat]
        top_col = int(np.argmax(row))
        top_label = label_ids[top_col]
        top_prob = float(row[top_col])
        if len(members) == 0:
            decision = "NO_MATCH"
        elif len(members) == 1:
            decision = "AUTO_ACCEPT"
        else:
            decision = "REVIEW"
        out.append(PredictionSet(i, members, top_label, top_prob, decision))
    return out


def empirical_coverage(
    pred_sets: list[PredictionSet],
    true_labels: list[int],
) -> float:
    """Fraction of queries whose true label falls inside the prediction set."""
    hits = sum(1 for ps, y in zip(pred_sets, true_labels) if y in ps.member_labels)
    return hits / len(pred_sets) if pred_sets else float("nan")
