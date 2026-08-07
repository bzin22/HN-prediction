"""Author features: an expanding-window track record, and a learned embedding table.

Authors are high cardinality and long tailed. Two things are built from the column:

* Rolling statistics over the author's *strictly earlier* posts. A post from March never
  sees a statistic computed with April data.
* An index into a learned embedding table, with authors below a minimum post count
  bucketed to a shared out-of-vocabulary row so the table does not memorise one-post
  accounts.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

#: Row reserved for authors below the minimum post count, and for unseen authors.
OOV_INDEX = 0


@dataclass(frozen=True)
class AuthorStatsConfig:
    """Settings for the expanding author statistics.

    window
        ``None`` means expanding from the start of the data. A finite window makes the
        statistic recency weighted instead.
    lag
        Settling lag, matching ``target.normalise``. A post from two hours ago has not
        finished scoring, so it should not yet count towards its author's track record.
    min_posts
        Below this many prior posts, the statistic falls back to the global prior rather
        than to a mean of one.
    """

    window: timedelta | None = None
    lag: timedelta = timedelta(hours=48)
    min_posts: int = 3


def expanding_author_stats(
    times: pd.Series,
    authors: pd.Series,
    targets: pd.Series,
    config: AuthorStatsConfig | None = None,
) -> pd.DataFrame:
    """Per-row author track record, as feature columns, from strictly earlier posts only.

    Returns columns ``author_prior_count``, ``author_prior_mean`` and
    ``author_prior_std``, aligned to the input rows.

    Not needed yet. The statistic itself is implemented and tested in
    ``features.history.prior_mean``, which rung 2 of the baseline ladder calls directly.
    This wrapper is the feature-column form the fusion models will want.
    """
    raise NotImplementedError("the statistic is implemented in features.history.prior_mean")


class AuthorEncoder:
    """Maps author names to embedding table rows, fitted on the training split."""

    def __init__(self, min_posts: int = 5) -> None:
        self.min_posts = min_posts
        self.author_to_index: dict[str, int] = {}

    def fit(self, authors: pd.Series) -> AuthorEncoder:
        """Assign a row to every author with at least ``min_posts`` training posts."""
        raise NotImplementedError

    def transform(self, authors: pd.Series) -> np.ndarray:
        """Map authors to row indices. Unknown and rare authors give ``OOV_INDEX``."""
        raise NotImplementedError

    @property
    def vocabulary_size(self) -> int:
        """Number of embedding rows, including the out-of-vocabulary row."""
        raise NotImplementedError
