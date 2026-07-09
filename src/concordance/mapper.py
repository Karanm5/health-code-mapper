"""
Vocabulary mapper: turn source terms into probability vectors over the set of
standard (SNOMED) candidate concepts.

This is the "model" that the conformal layer calibrates. It:
  1. fits the similarity backbone on all concept names,
  2. scores each source term against every standard candidate,
  3. converts scores to a probability distribution via temperature softmax.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .similarity import SimilarityModel, softmax


@dataclass
class Candidates:
    label_ids: list[int]        # standard concept_id per column
    names: list[str]            # standard concept_name per column


class VocabularyMapper:
    def __init__(self, temperature: float = 0.08, ngram_range=(2, 4)) -> None:
        self.temperature = temperature
        self.sim = SimilarityModel(ngram_range=ngram_range)
        self.candidates: Candidates | None = None

    def fit(self, standard_ids: list[int], standard_names: list[str],
            extra_corpus: list[str] | None = None) -> "VocabularyMapper":
        """Fit similarity on standard names (+ any extra source strings)."""
        self.candidates = Candidates(list(standard_ids), list(standard_names))
        corpus = list(standard_names) + list(extra_corpus or [])
        self.sim.fit(corpus)
        return self

    def predict_proba(self, source_terms: list[str]) -> np.ndarray:
        """Return (n_terms, n_candidates) probability matrix."""
        if self.candidates is None:
            raise RuntimeError("call fit() first")
        sims = self.sim.similarity(source_terms, self.candidates.names)
        return softmax(sims, temperature=self.temperature, axis=1)

    def label_ids(self) -> list[int]:
        assert self.candidates is not None
        return self.candidates.label_ids
