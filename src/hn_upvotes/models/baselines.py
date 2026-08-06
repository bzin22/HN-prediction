"""The baseline ladder.

Built before any neural network, so the fusion models have a real bar to clear. Five
rungs, in increasing order of what they know:

1. Predict the trailing baseline, which in normalised space is just zero
2. Author historical mean, time aware
3. Domain historical mean, time aware
4. TF-IDF plus Ridge
5. TF-IDF plus gradient boosting

If a fusion model does not beat rung 5, the README says so. Rungs 4 and 5 need
scikit-learn from the ``train`` extra. Rungs 1 to 3 are numpy only.

Every estimator here predicts in normalised target space, the same space the fusion
models train in, so the comparison table is like for like.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Estimator(Protocol):
    """The interface every rung of the ladder and every fusion model presents."""

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> Estimator:
        """Fit on the training split. ``frame`` holds allowlisted columns only."""
        ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict normalised targets, one per row."""
        ...


class TrailingBaselinePredictor:
    """Rung 1. Predicts zero for everything.

    Zero in normalised space means "exactly the trailing average", so this is the
    do-nothing model. Any model that cannot beat it has learned nothing. Non-trivial to
    beat on Spearman, because it is a constant and therefore has no ranking at all.
    """

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> TrailingBaselinePredictor:
        """No parameters to fit."""
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Return zeros."""
        raise NotImplementedError


class AuthorMeanPredictor:
    """Rung 2. The author's mean normalised target over their strictly earlier posts.

    Uses ``features.author.expanding_author_stats``, so an author's March prediction is
    never informed by their April posts. Falls back to zero below ``min_posts``.
    """

    def __init__(self, min_posts: int = 3) -> None:
        self.min_posts = min_posts

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> AuthorMeanPredictor:
        """Record the training-period author statistics."""
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Look up each row's author prior."""
        raise NotImplementedError


class DomainMeanPredictor:
    """Rung 3. The domain's mean normalised target over strictly earlier posts."""

    def __init__(self, min_posts: int = 5) -> None:
        self.min_posts = min_posts

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> DomainMeanPredictor:
        """Record the training-period domain statistics."""
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Look up each row's domain prior."""
        raise NotImplementedError


class TfidfRidge:
    """Rung 4. TF-IDF over titles into Ridge regression.

    The bar the from-scratch embeddings have to clear. A linear model over word counts
    is a strong baseline on short text, and saying so up front is more useful than
    discovering it after training three fusion architectures.
    """

    def __init__(self, max_features: int = 50_000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> TfidfRidge:
        """Fit the vectoriser and the regressor on the training split only."""
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Vectorise and predict."""
        raise NotImplementedError


class TfidfGradientBoosting:
    """Rung 5. TF-IDF plus author, domain and temporal features into gradient boosting.

    The hardest rung, and the honest comparison for the fusion models: it sees the same
    four modalities, just without learned representations.
    """

    def __init__(self, max_features: int = 50_000, n_estimators: int = 500) -> None:
        self.max_features = max_features
        self.n_estimators = n_estimators

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> TfidfGradientBoosting:
        """Fit on the training split only."""
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict normalised targets."""
        raise NotImplementedError
