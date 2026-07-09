"""
Concordance — interactive demo.

Browse how source clinical codes (ICD-10 from hospital data, Read v2 from primary
care) are mapped to standard SNOMED concepts, each with a calibrated confidence
and a conformal triage decision. Move the error-rate slider to watch the
coverage / human-effort tradeoff in real time.

Run:  streamlit run app/streamlit_app.py
"""
import csv
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from concordance.mapper import VocabularyMapper                    # noqa: E402
from concordance.conformal import (                                # noqa: E402
    build_prediction_sets,
    calibrate_threshold,
    empirical_coverage,
)

VOCAB = os.path.join(ROOT, "data", "vocab")


@st.cache_data
def load_vocab():
    # Self-heal on a fresh deploy: build the curated vocab if it is missing.
    if not os.path.exists(os.path.join(VOCAB, "concept.csv")):
        import subprocess
        subprocess.run([sys.executable, os.path.join(ROOT, "data", "build_vocabulary.py")],
                       check=True)
    standards, standard_names, id_to_name = [], [], {}
    with open(os.path.join(VOCAB, "concept.csv")) as f:
        for r in csv.DictReader(f):
            cid = int(r["concept_id"])
            id_to_name[cid] = r["concept_name"]
            if r["standard_concept"] == "S":
                standards.append(cid)
                standard_names.append(r["concept_name"])
    src = []
    with open(os.path.join(VOCAB, "source_codes.csv")) as f:
        for r in csv.DictReader(f):
            src.append((r["source_vocabulary"], r["source_code"],
                        r["source_name"], int(r["true_target_concept_id"])))
    return standards, standard_names, id_to_name, src


@st.cache_resource
def build_mapper(standards, standard_names, src_terms):
    m = VocabularyMapper(temperature=0.08)
    m.fit(standards, standard_names, extra_corpus=src_terms)
    return m


def main():
    st.set_page_config(page_title="Health-Code-Mapper", layout="wide")
    st.title("Health-Code-Mapper")
    st.caption("Uncertainty-calibrated mapping of source clinical codes to "
               "standard OMOP (SNOMED) concepts. Synthetic data.")

    standards, standard_names, id_to_name, src = load_vocab()
    src_vocabs = [s[0] for s in src]
    src_codes = [s[1] for s in src]
    src_terms = [s[2] for s in src]
    src_true = [s[3] for s in src]

    mapper = build_mapper(standards, standard_names, src_terms)
    probs = mapper.predict_proba(src_terms)
    label_ids = mapper.label_ids()
    col_of = {cid: j for j, cid in enumerate(label_ids)}
    true_idx = np.array([col_of[c] for c in src_true])

    with st.sidebar:
        st.header("Controls")
        alpha = st.slider("Error rate α (target miss rate)", 0.01, 0.30, 0.10, 0.01)
        seed = st.number_input("Calibration split seed", 0, 9999, 7)
        st.markdown(
            "**Triage**\n\n"
            "- 1 candidate in set → auto-accept\n"
            "- ≥2 → route to human review\n"
            "- 0 → flag as no-match"
        )

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(src_terms))
    k = len(src_terms) // 2
    cal, test = perm[:k], perm[k:]

    q_hat = calibrate_threshold(probs[cal], true_idx[cal], alpha=alpha)
    sets = build_prediction_sets(probs[test], q_hat, label_ids)
    true_test = [src_true[i] for i in test]

    cover = empirical_coverage(sets, true_test)
    auto = [s for s in sets if s.decision == "AUTO_ACCEPT"]
    naive_correct = sum(1 for s, y in zip(sets, true_test) if s.top_label == y)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Empirical coverage", f"{cover:.0%}", f"target {1-alpha:.0%}")
    c2.metric("Auto-accepted", f"{len(auto)/len(sets):.0%}")
    auto_acc = (sum(1 for i, s in enumerate(sets) if s.decision == "AUTO_ACCEPT"
                    and s.top_label == true_test[i]) / len(auto)) if auto else float("nan")
    c3.metric("Auto-accept accuracy", f"{auto_acc:.0%}" if auto else "—")
    c4.metric("Naive top-1 accuracy", f"{naive_correct/len(sets):.0%}")

    st.subheader("Per-code mapping decisions (test split)")
    colour = {"AUTO_ACCEPT": "🟢", "REVIEW": "🟡", "NO_MATCH": "🔴"}
    table = []
    for pos, i in enumerate(test):
        ps = sets[pos]
        table.append({
            "source": f"{src_vocabs[i]} {src_codes[i]}",
            "source term": src_terms[i],
            "proposed standard": id_to_name[ps.top_label],
            "confidence": round(ps.top_prob, 3),
            "set size": len(ps.member_labels),
            "decision": f"{colour[ps.decision]} {ps.decision}",
            "correct": "✓" if src_true[i] in ps.member_labels else "✗",
        })
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    st.subheader("Coverage vs error rate (repeated splits)")
    from concordance.evaluate import repeated_split_coverage
    curve = [repeated_split_coverage(probs, true_idx, label_ids, alpha=a,
                                     n_repeats=200, seed=1)
             for a in (0.20, 0.15, 0.10, 0.05)]
    cov_df = pd.DataFrame([{
        "α": c["alpha"],
        "target coverage": c["target_coverage"],
        "empirical coverage": c["mean_empirical_coverage"],
        "avg set size": c["mean_avg_set_size"],
    } for c in curve])
    st.dataframe(cov_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
