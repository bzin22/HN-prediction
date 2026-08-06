"""Turn a title's word vectors into one title vector.

Mean pooling is the default. SIF (smooth inverse frequency weighting plus removal of the
first principal component, Arora et al. 2017, "A Simple but Tough-to-Beat Baseline for
Sentence Embeddings") is the upgrade. Both sit behind one interface so the fusion models
do not know or care which is active.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class TitlePooler(Protocol):
    """Pools a sequence of token indices into a single fixed-length vector."""

    def fit(self, token_ids: list[list[int]], embeddings: np.ndarray) -> TitlePooler:
        """Learn whatever the pooler needs from the training titles.

        Mean pooling learns nothing. SIF learns the word frequencies and the first
        principal component, and must learn them from training rows only.
        """
        ...

    def pool(self, token_ids: list[list[int]], embeddings: np.ndarray) -> np.ndarray:
        """Return one row per title, of width ``embeddings.shape[1]``."""
        ...


class MeanPooler:
    """Unweighted mean of the token vectors. Empty titles pool to zeros."""

    def fit(self, token_ids: list[list[int]], embeddings: np.ndarray) -> MeanPooler:
        """No-op. Present so the two poolers share an interface."""
        raise NotImplementedError

    def pool(self, token_ids: list[list[int]], embeddings: np.ndarray) -> np.ndarray:
        """Average the token vectors of each title."""
        raise NotImplementedError


class SIFPooler:
    """Smooth inverse frequency weighting, then first principal component removal.

    Each token is weighted ``a / (a + p(w))``, which damps common words without a stop
    list. The common component is then projected out. ``a = 1e-3`` is the paper's value.
    """

    def __init__(self, a: float = 1e-3) -> None:
        self.a = a
        self.word_probabilities: np.ndarray | None = None
        self.common_component: np.ndarray | None = None

    def fit(self, token_ids: list[list[int]], embeddings: np.ndarray) -> SIFPooler:
        """Estimate word probabilities and the first principal component.

        Both are fitted on training rows only. Fitting on the full frame would leak
        test-period vocabulary statistics backwards into training.
        """
        raise NotImplementedError

    def pool(self, token_ids: list[list[int]], embeddings: np.ndarray) -> np.ndarray:
        """Weight, average, then subtract the projection onto the common component."""
        raise NotImplementedError
