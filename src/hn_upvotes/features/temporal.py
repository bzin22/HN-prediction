"""Time features that mean the same thing in any year.

Deliberately no ``year`` feature and no absolute time index. Training stops years before
the test period, so a model has no branch for a year it never saw: every future post
falls into the last training bucket and gets that era's numbers. Era drift is handled by
the target transform in ``target.normalise``, not by a feature.

What is left is the part of a timestamp that repeats. Submitting at 09:00 UTC on a
Tuesday means roughly the same thing in 2013 and in 2025, because it is about who is
awake and how crowded the front page is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Feature columns this module produces, in a fixed order the models rely on.
TEMPORAL_FEATURE_NAMES: tuple[str, ...] = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
)


def cyclical_encode(values: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode a cyclical integer as a sine and cosine pair.

    Hour 23 and hour 0 are one apart, but as raw integers they are 23 apart. Projecting
    onto a circle fixes that, and costs one extra column.
    """
    raise NotImplementedError


def hour_of_day(times: pd.Series) -> np.ndarray:
    """UTC hour of submission, 0 to 23.

    Left in UTC rather than converted to a local timezone. The audience for a post is
    the site's readership, which is global, so the site's own clock is the right one.
    """
    raise NotImplementedError


def day_of_week(times: pd.Series) -> np.ndarray:
    """Day of week, Monday as 0."""
    raise NotImplementedError


def build_temporal_features(times: pd.Series) -> pd.DataFrame:
    """Build every temporal feature, in ``TEMPORAL_FEATURE_NAMES`` order.

    The only input is the ``time`` column, which is on the submission-time allowlist in
    ``features.schema``.
    """
    raise NotImplementedError
