"""Tests for the similarity mapper."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from concordance.mapper import VocabularyMapper       # noqa: E402


STANDARD_IDS = [201826, 320128, 317009]
STANDARD_NAMES = ["Type 2 diabetes mellitus", "Essential hypertension", "Asthma"]


def test_predict_proba_shape_and_normalisation():
    m = VocabularyMapper().fit(STANDARD_IDS, STANDARD_NAMES)
    probs = m.predict_proba(["asthma", "high blood pressure"])
    assert probs.shape == (2, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_obvious_match_is_top1():
    m = VocabularyMapper().fit(STANDARD_IDS, STANDARD_NAMES)
    probs = m.predict_proba(["Type 2 diabetes mellitus"])
    top = STANDARD_IDS[int(np.argmax(probs[0]))]
    assert top == 201826
