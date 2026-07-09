"""
String-similarity backbone for candidate mapping.

We use TF-IDF over character n-grams and cosine similarity. This is deliberately
the same family of method that OHDSI's Usagi uses for term matching, so the
comparison is fair: the contribution of this project is *not* a fancier matcher,
it is the calibrated conformal layer that sits on top (see conformal.py).

The similarity model is unsupervised: it is fit on concept-name strings only and
never sees the ground-truth source->target labels, so using the labelled pairs
purely to calibrate the conformal threshold introduces no leakage.

Swapping in dense sentence embeddings (e.g. a biomedical sentence-transformer)
is a drop-in change: replace `SimilarityModel` with one that returns cosine
similarities from dense vectors. Everything downstream is model-agnostic.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityModel:
    """Fit on a corpus of concept names; score a query against candidate targets."""

    def __init__(self, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ngram_range,
            lowercase=True,
        )
        self._fitted = False

    def fit(self, corpus: list[str]) -> "SimilarityModel":
        self.vectorizer.fit(corpus)
        self._fitted = True
        return self

    def similarity(self, queries: list[str], targets: list[str]) -> np.ndarray:
        """Return an (n_queries, n_targets) matrix of cosine similarities."""
        if not self._fitted:
            raise RuntimeError("call fit() before similarity()")
        q = self.vectorizer.transform(queries)
        t = self.vectorizer.transform(targets)
        return cosine_similarity(q, t)


def softmax(scores: np.ndarray, temperature: float = 0.1, axis: int = -1) -> np.ndarray:
    """Temperature-scaled softmax turning similarities into a probability vector.

    Lower temperature -> sharper distribution. Cosine similarities live in a
    narrow band, so a small temperature is needed to separate candidates.
    """
    z = scores / max(temperature, 1e-6)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)
