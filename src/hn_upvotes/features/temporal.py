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
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")
    angle = 2.0 * np.pi * np.asarray(values, dtype=np.float64) / period
    return np.sin(angle), np.cos(angle)


def hour_of_day(times: pd.Series) -> np.ndarray:
    """UTC hour of submission, 0 to 23.

    Left in UTC rather than converted to a local timezone. The audience for a post is
    the site's readership, which is global, so the site's own clock is the right one.
    """
    return _times(times).dt.hour.to_numpy(dtype=np.int64)


def day_of_week(times: pd.Series) -> np.ndarray:
    """Day of week, Monday as 0."""
    return _times(times).dt.dayofweek.to_numpy(dtype=np.int64)


def build_temporal_features(times: pd.Series) -> pd.DataFrame:
    """Build every temporal feature, in ``TEMPORAL_FEATURE_NAMES`` order.

    The only input is the ``time`` column, which is on the submission-time allowlist in
    ``features.schema``.
    """
    hour = hour_of_day(times)
    day = day_of_week(times)
    hour_sin, hour_cos = cyclical_encode(hour, 24)
    dow_sin, dow_cos = cyclical_encode(day, 7)
    return pd.DataFrame(
        {
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
            "is_weekend": (day >= 5).astype(np.float64),
        },
        columns=list(TEMPORAL_FEATURE_NAMES),
    )


def _times(times: pd.Series) -> pd.Series:
    """The one coercion, shared with the target transform."""
    from hn_upvotes.target.normalise import to_datetime

    return to_datetime(times)
