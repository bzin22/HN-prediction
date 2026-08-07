"""Evaluation metrics.

Four numbers per model, and they answer different questions on purpose.

* **RMSE and MAE on ``log1p(score)``**, the space the model trains in. Phase 1 switched
  the trailing z-score off, so the target *is* ``log1p(score)`` and there is only one
  error space. If that transform is ever switched back on, these two need the inverse
  mapping applied first and this module needs changing with it.
* **Spearman rank correlation on raw score.** This is the one that matters. The tail
  inflated across the split, so absolute error partly measures inflation rather than
  model quality. Rank is immune to that. Spearman on raw score and Spearman on
  ``log1p(score)`` are the same number, because ``log1p`` is strictly increasing and
  rank correlation only sees order, so this is computed in ``log1p`` space and quoted on
  raw score without any conversion.
* **Precision@100.** Of the 100 posts the model ranks highest, how many really landed in
  the top 100. Absolute error can look respectable while the ranking at the top is
  worthless, and this is the metric that catches it.

Ties are not a footnote here. 57.6% of stories score 1 or 2, so the actuals have very
long tie runs at the bottom, and Spearman uses average ranks to handle them.

numpy only, no scipy, so the metrics import with the base install and CI does not need
the ``train`` extra to test them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class MetricReport:
    """Every metric for one model on one evaluation slice."""

    rmse_log1p_score: float
    mae_log1p_score: float
    spearman_raw_score: float
    precision_at_100: float
    n_rows: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Root mean squared error."""
    predictions, actuals = _aligned(predictions, actuals)
    return float(np.sqrt(np.mean((predictions - actuals) ** 2)))


def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean absolute error."""
    predictions, actuals = _aligned(predictions, actuals)
    return float(np.mean(np.abs(predictions - actuals)))


def spearman(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Spearman rank correlation, with average ranks for ties.

    Returns 0.0 when either side is constant, which is the honest answer for a model
    that predicts one number for everything: it has no ranking, so it has no rank
    agreement. Pearson on the ranks is undefined there rather than zero, and reporting
    ``nan`` in that cell would just push the question into the README.
    """
    predictions, actuals = _aligned(predictions, actuals)
    if len(predictions) < 2:
        return 0.0
    predicted_ranks = average_ranks(predictions)
    actual_ranks = average_ranks(actuals)
    if predicted_ranks.std() == 0.0 or actual_ranks.std() == 0.0:
        return 0.0
    return float(np.corrcoef(predicted_ranks, actual_ranks)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks of ``values``, with tied entries sharing their average rank.

    Three values ``[5, 5, 9]`` rank ``[1.5, 1.5, 3]``: the two fives split ranks 1 and 2.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)

    # Each run of equal values gets the mean of the positions it occupies.
    run_start = np.flatnonzero(np.concatenate([[True], ordered[1:] != ordered[:-1], [True]]))
    for start, stop in zip(run_start[:-1], run_start[1:], strict=True):
        ranks[order[start:stop]] = (start + stop + 1) / 2.0
    return ranks


def precision_at_k(predictions: np.ndarray, actuals: np.ndarray, k: int = 100) -> float:
    """Fraction of the model's top ``k`` that really is in the top ``k`` by actual value.

    "In the top ``k``" means an actual value at least as high as the ``k``-th largest.
    Scores tie, so a post level with the 100th-placed post counts as being in the top
    100. Drawing the line by row order instead would make the number depend on the order
    the rows happened to arrive in.
    """
    predictions, actuals = _aligned(predictions, actuals)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(predictions) < k:
        raise ValueError(f"cannot take the top {k} of {len(predictions)} rows")

    threshold = np.partition(actuals, -k)[-k]
    chosen = np.argpartition(-predictions, k - 1)[:k]
    return float(np.mean(actuals[chosen] >= threshold))


def report(predictions: np.ndarray, actuals: np.ndarray, k: int = 100) -> MetricReport:
    """Compute every metric for one evaluation slice.

    Both arguments are in ``log1p(score)`` space. Spearman is quoted against raw score
    and needs no conversion to get there; see the module docstring.
    """
    predictions, actuals = _aligned(predictions, actuals)
    return MetricReport(
        rmse_log1p_score=rmse(predictions, actuals),
        mae_log1p_score=mae(predictions, actuals),
        spearman_raw_score=spearman(predictions, actuals),
        precision_at_100=precision_at_k(predictions, actuals, k=k),
        n_rows=len(actuals),
    )


def _aligned(predictions: np.ndarray, actuals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float64)
    actuals = np.asarray(actuals, dtype=np.float64)
    if predictions.shape != actuals.shape:
        raise ValueError(
            f"predictions and actuals differ in shape: {predictions.shape} vs {actuals.shape}"
        )
    if predictions.size == 0:
        raise ValueError("no rows to score")
    return predictions, actuals
