"""
Run the full mapping + conformal-triage pipeline and write results.

Outputs (under outputs/):
  - mappings.csv    per source code: proposed target, confidence, set size, decision
  - metrics.json    triage metrics + repeated-split coverage across alphas
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from concordance.mapper import VocabularyMapper          # noqa: E402
from concordance.evaluate import (                        # noqa: E402
    evaluate_single_split,
    metrics_to_dict,
    repeated_split_coverage,
)

ROOT = os.path.join(os.path.dirname(__file__), "..")
VOCAB = os.path.join(ROOT, "data", "vocab")
OUT = os.path.join(ROOT, "outputs")
ALPHA = 0.10
CAL_FRAC = 0.5
SEED = 7


def load_vocab():
    standards, standard_names = [], []
    id_to_name = {}
    with open(os.path.join(VOCAB, "concept.csv")) as f:
        for r in csv.DictReader(f):
            cid = int(r["concept_id"])
            id_to_name[cid] = r["concept_name"]
            if r["standard_concept"] == "S":
                standards.append(cid)
                standard_names.append(r["concept_name"])

    src_terms, src_true, src_meta = [], [], []
    with open(os.path.join(VOCAB, "source_codes.csv")) as f:
        for r in csv.DictReader(f):
            src_terms.append(r["source_name"])
            src_true.append(int(r["true_target_concept_id"]))
            src_meta.append((r["source_vocabulary"], r["source_code"]))
    return standards, standard_names, id_to_name, src_terms, src_true, src_meta


def main():
    os.makedirs(OUT, exist_ok=True)
    (standards, standard_names, id_to_name,
     src_terms, src_true, src_meta) = load_vocab()

    mapper = VocabularyMapper(temperature=0.08)
    mapper.fit(standards, standard_names, extra_corpus=src_terms)
    probs = mapper.predict_proba(src_terms)
    label_ids = mapper.label_ids()
    col_of = {cid: j for j, cid in enumerate(label_ids)}
    true_idx = np.array([col_of[c] for c in src_true])

    # One reproducible split for the detailed per-code output table.
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(src_terms))
    k = int(len(src_terms) * CAL_FRAC)
    cal, test = perm[:k], perm[k:]

    metrics, sets = evaluate_single_split(
        probs[cal], true_idx[cal],
        probs[test], [src_true[i] for i in test],
        label_ids, alpha=ALPHA,
    )

    # Per-code results table (test split).
    with open(os.path.join(OUT, "mappings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_vocabulary", "source_code", "source_term",
                    "proposed_standard", "confidence", "set_size",
                    "decision", "correct"])
        for pos, i in enumerate(test):
            ps = sets[pos]
            correct = src_true[i] in ps.member_labels
            w.writerow([
                src_meta[i][0], src_meta[i][1], src_terms[i],
                id_to_name[ps.top_label], f"{ps.top_prob:.3f}",
                len(ps.member_labels), ps.decision, correct,
            ])

    # Coverage across a range of alphas (repeated splits -> stable estimate).
    coverage_curve = [
        repeated_split_coverage(probs, true_idx, label_ids, alpha=a,
                                n_repeats=300, cal_frac=CAL_FRAC, seed=1)
        for a in (0.20, 0.15, 0.10, 0.05)
    ]

    report = {
        "n_source_codes": len(src_terms),
        "n_standard_candidates": len(standards),
        "single_split": metrics_to_dict(metrics),
        "coverage_curve": coverage_curve,
    }
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    m = metrics
    print("=== Concordance mapping pipeline ===")
    print(f"source codes: {len(src_terms)} | standard candidates: {len(standards)}")
    print(f"alpha = {m.alpha}  (target coverage {1 - m.alpha:.0%})")
    print(f"empirical coverage (test split): {m.empirical_coverage:.1%}")
    print(f"avg conformal set size:          {m.avg_set_size}")
    print(f"naive top-1 accuracy:            {m.naive_top1_accuracy:.1%}")
    print(f"auto-accept rate:                {m.auto_accept_rate:.1%}")
    print(f"accuracy within auto-accepted:   {m.auto_accept_accuracy:.1%}")
    print(f"routed to human review:          {m.review_rate:.1%}")
    print(f"flagged no-match:                {m.no_match_rate:.1%}")
    print("\nRepeated-split coverage (300 splits/alpha):")
    for c in coverage_curve:
        print(f"  alpha={c['alpha']:.2f}  target={c['target_coverage']:.0%}  "
              f"empirical={c['mean_empirical_coverage']:.1%} "
              f"(+/-{c['std_empirical_coverage']:.1%})  "
              f"avg set size={c['mean_avg_set_size']}")


if __name__ == "__main__":
    main()
