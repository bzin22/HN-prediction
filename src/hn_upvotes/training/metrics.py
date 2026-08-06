"""Evaluation metrics.

Primary metrics are RMSE and MAE in the normalised target space, because that is the
space the model trains in. A normalised error is hard to read on its own, so three more
are reported alongside:

* RMSE on ``log1p(score)``, obtained by mapping predictions back through the test
  period's trailing baseline. Quotable in real score terms.
* Spearman correlation on raw score. Ranking quality is what a submission-time
  predictor is actually for, and Spearman is invariant to the whole normalisation, so
  it is the metric that cannot be flattered by the target transform.
* Precision@k. Of the top 100 posts the model predicts, how many really landed high.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hn_upvotes.target.normalise import Baseline


@dataclass(frozen=True)
class MetricReport:
    """Every metric for one model on one evaluation slice."""

    rmse_normalised: float
    mae_normalised: float
    rmse_log1p_score: float
    spearman_raw_score: float
    precision_at_100: float
    n_rows: int


def rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Root mean squared error."""
    raise NotImplementedError


def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Mean absolute error."""
    raise NotImplementedError


def spearman(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Spearman rank correlation, with average ranks for ties.

    Ties matter here rather than being a footnote. A large share of HN posts score 1 or
    2, so the actuals have very long tie runs at the bottom.
    """
    raise NotImplementedError


def precision_at_k(predictions: np.ndarray, actuals: np.ndarray, k: int = 100) -> float:
    """Overlap between the model's top ``k`` and the true top ``k``, as a fraction."""
    raise NotImplementedError


def rmse_in_log1p_space(
    predictions: np.ndarray,
    actuals: np.ndarray,
    baseline: Baseline,
) -> float:
    """RMSE after mapping both sides back through the trailing baseline.

    Uses ``target.normalise.inverse`` and then ``log1p`` again, so the number is in
    score units rather than normalised units and can be quoted to someone who has not
    read the target section.
    """
    raise NotImplementedError


def report(
    predictions: np.ndarray,
    actuals: np.ndarray,
    baseline: Baseline,
    k: int = 100,
) -> MetricReport:
    """Compute every metric for one evaluation slice."""
    raise NotImplementedError


def aggregate_across_seeds(reports: list[MetricReport]) -> dict[str, tuple[float, float]]:
    """Mean and standard deviation of every metric across seeds.

    Returns ``{metric_name: (mean, std)}``. The README quotes both. Where two models
    differ by less than this standard deviation, they are reported as indistinguishable
    rather than ranked.
    """
    raise NotImplementedError
