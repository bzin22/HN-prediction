"""Target transform: ``log1p(score)`` normalised against a trailing baseline.

Forward and inverse:

    target = (log1p(score) - baseline_mean) / baseline_spread
    score  = expm1(target * baseline_spread + baseline_mean)

The baseline for a row is computed from a window of rows that ended before that row was
submitted. Two rules make it usable at inference time as well as in a backtest:

* **Strictly earlier.** A row's baseline never sees a row at or after its own timestamp.
* **Settling lag.** Rows from the last ``lag`` are excluded as well, because a post from
  three hours ago is still collecting votes and its score is not final.

**Phase 1 found the drift this machinery exists to remove, and found that this machinery
cannot see it.** Era drift is real and it lives in the tail: the 99th percentile of raw
score went from 38 points in 2007 to 355 in 2025. Over the same span the median of
``log1p(score)`` did not move at all and the standard deviation moved 2.9%. An affine
correction fitted to a flat location and a flat scale leaves the tail where it was, so
switching this on would not have corrected the drift.

The defaults are therefore ``centre="zero"`` and ``scale=False``, and the forward
transform reduces to plain ``log1p(score)``. Each default carries the measurement that
set it. The window stays 30 days and the lag is 24 hours, down from 48.

The machinery is kept, not deleted. It is one config change to turn back on and the
trailing window is still what the author and domain statistics use. Tail drift remains
unhandled and is an open risk for Phase 2. See ``docs/design.md``.

Nothing here is a stub. ``tests/test_target_normalise.py`` covers the round trip and the
strictly-earlier rule, and ``tests/test_baseline_defaults.py`` pins the measured values.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import numpy as np
import pandas as pd

Centre = Literal["mean", "median", "zero"]
Spread = Literal["std", "iqr"]


@dataclass(frozen=True)
class BaselineConfig:
    """How the trailing baseline is computed.

    window
        Length of the trailing window.
    lag
        Settling lag. Rows newer than ``row_time - lag`` are excluded.
    centre
        Statistic subtracted from ``log1p(score)``. ``"zero"`` disables centring.
    scale
        Whether to divide by the spread. Set ``False`` to use a subtraction-only
        transform.
    spread
        Statistic used when ``scale`` is true. ``"iqr"`` is the robust option and is
        rescaled to be comparable with a standard deviation under normality.
    min_periods
        Minimum number of rows in the window before its statistics are trusted. Below
        this, the fallbacks are used.
    fallback_centre, fallback_spread
        Values used for rows with too little history, which is unavoidable for the first
        ``window + lag`` of any dataset. A fallback spread of 1.0 makes the transform a
        no-op scale rather than a divide by zero.
    """

    window: timedelta = timedelta(days=30)

    # Gate 3, measured. Live scores were read for 13,852 stories at one instant
    # (2026-08-06 06:22 UTC) and bucketed by age. Balanced across UTC hour of day, mean
    # log1p(score) is 1.490 [1.377, 1.603] for stories under 12 hours old, against a
    # settled reference of 1.683 [1.656, 1.712]. The 12 to 24 hour band is 1.733
    # [1.662, 1.816], which already overlaps the reference. Scores arrive at their final
    # level between 12 and 24 hours; 24 is the top of that interval. Was 48 hours.
    lag: timedelta = timedelta(hours=24)

    # Gate 1, measured. Median log1p(score) per year takes exactly one value,
    # 1.0986 = log1p(2), for every year from 2007 to 2024. 2025 is the single exception,
    # at log1p(3). That flatness is largely a floor artefact: 57.6% of stories score 1 or
    # 2, so the median is pinned there whatever happens above it. It is not evidence that
    # scoring is stable. It does establish there is no centre for a trailing subtraction
    # to remove, so centring is off. Was "mean".
    centre: Centre = "zero"

    # Gate 2, measured. Over 2010-2025 the standard deviation of log1p(score) moves
    # 1.1386 to 1.1721, which is 2.9%, and the interquartile range moves the other way,
    # 1.2528 to 0.8473 with a year correlation of -0.098. Two spread measures disagreeing
    # in sign means neither tracks a real trend. The drift is in the tail, where neither
    # can reach it: the 99th percentile of raw score moved 9.3x over the same span.
    # Was True.
    scale: bool = False

    # Unused while scale is False. Kept as "std" because it is the one to re-enable if a
    # later phase finds drift: the IQR of log1p(score) is pinned by the floor spike
    # (57.6% of stories score 1 or 2, so the 25th percentile is log1p(1) in every year)
    # and measures the floor rather than the spread.
    spread: Spread = "std"
    min_periods: int = 30
    fallback_centre: float = 0.0
    fallback_spread: float = 1.0

    def __post_init__(self) -> None:
        if self.window <= timedelta(0):
            raise ValueError(f"window must be positive, got {self.window}")
        if self.lag < timedelta(0):
            raise ValueError(f"lag cannot be negative, got {self.lag}")
        if self.min_periods < 1:
            raise ValueError(f"min_periods must be at least 1, got {self.min_periods}")
        if self.fallback_spread <= 0:
            raise ValueError(f"fallback_spread must be positive, got {self.fallback_spread}")


@dataclass(frozen=True)
class Baseline:
    """Per-row trailing statistics, aligned to the rows they were computed for.

    ``spread`` is all ones when ``BaselineConfig.scale`` is false, so ``forward`` and
    ``inverse`` need no branch on the configuration.
    """

    centre: np.ndarray
    spread: np.ndarray
    n_prior: np.ndarray
    config: BaselineConfig

    def __len__(self) -> int:
        return int(self.centre.shape[0])


#: Scaling that puts an interquartile range on the same footing as a standard deviation
#: for normally distributed data. 1 / (2 * Phi^-1(0.75)).
_IQR_TO_SIGMA = 0.7413


def to_datetime(times: pd.Series | np.ndarray) -> pd.Series:
    """Coerce a time column to timezone-naive UTC datetimes.

    The Hacker News dump stores ``time`` as unix seconds, so integer input is read as
    seconds since the epoch. Datetime input is passed through, with any timezone
    converted to UTC and then dropped so comparisons stay cheap.
    """
    series = pd.Series(times).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s")
    converted = pd.to_datetime(series)
    if isinstance(converted.dtype, pd.DatetimeTZDtype):
        converted = converted.dt.tz_convert("UTC").dt.tz_localize(None)
    return converted


def compute_trailing_baseline(
    times: pd.Series | np.ndarray,
    scores: pd.Series | np.ndarray,
    config: BaselineConfig | None = None,
) -> Baseline:
    """Compute the trailing baseline of ``log1p(score)`` for every row.

    For a row at time ``t``, the window is every row whose time lies in
    ``[t - lag - window, t - lag)``. The upper bound is exclusive, so a row can never
    contribute to its own baseline, and neither can any row submitted at the same
    instant or later.

    Input order does not matter. Results come back aligned to the input, not sorted.
    """
    config = config or BaselineConfig()

    t = to_datetime(times)
    y = np.log1p(np.asarray(scores, dtype=np.float64))
    if len(t) != len(y):
        raise ValueError(f"times and scores differ in length: {len(t)} vs {len(y)}")

    n = len(y)
    if n == 0:
        empty = np.empty(0, dtype=np.float64)
        return Baseline(empty, empty, np.empty(0, dtype=np.int64), config)

    # Work in sorted time order so the window is a contiguous slice, then undo the sort.
    order = np.argsort(t.to_numpy(), kind="stable")
    t_sorted = t.to_numpy()[order]
    y_sorted = y[order]

    lag = np.timedelta64(config.lag)
    window = np.timedelta64(config.window)
    hi = np.searchsorted(t_sorted, t_sorted - lag, side="left")
    lo = np.searchsorted(t_sorted, t_sorted - lag - window, side="left")

    counts = (hi - lo).astype(np.int64)
    centre, spread = _window_statistics(y_sorted, lo, hi, counts, config)

    enough = counts >= config.min_periods
    centre = np.where(enough, centre, config.fallback_centre)
    spread = np.where(enough, spread, config.fallback_spread)

    # A degenerate window (every score identical) would divide by zero.
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, config.fallback_spread)

    if config.centre == "zero":
        centre = np.zeros(n, dtype=np.float64)
    if not config.scale:
        spread = np.ones(n, dtype=np.float64)

    unsort = np.empty(n, dtype=np.int64)
    unsort[order] = np.arange(n)
    return Baseline(centre[unsort], spread[unsort], counts[unsort], config)


def _window_statistics(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    counts: np.ndarray,
    config: BaselineConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Centre and spread of ``y[lo[i]:hi[i]]`` for every ``i``. ``y`` is time sorted."""
    n = len(y)
    safe = np.maximum(counts, 1)

    if config.centre == "median" or (config.scale and config.spread == "iqr"):
        # Order statistics do not decompose into prefix sums, so these are per row.
        centre = np.full(n, config.fallback_centre, dtype=np.float64)
        spread = np.full(n, config.fallback_spread, dtype=np.float64)
        for i in range(n):
            if counts[i] == 0:
                continue
            window = y[lo[i] : hi[i]]
            centre[i] = np.median(window) if config.centre == "median" else window.mean()
            if config.spread == "iqr":
                q75, q25 = np.percentile(window, [75, 25])
                spread[i] = (q75 - q25) * _IQR_TO_SIGMA
            else:
                spread[i] = window.std()
        return centre, spread

    # mean and standard deviation, from prefix sums, O(n).
    cumulative = np.concatenate([[0.0], np.cumsum(y)])
    cumulative_sq = np.concatenate([[0.0], np.cumsum(y * y)])
    total = cumulative[hi] - cumulative[lo]
    total_sq = cumulative_sq[hi] - cumulative_sq[lo]

    centre = total / safe
    variance = np.maximum(total_sq / safe - centre**2, 0.0)
    return centre, np.sqrt(variance)


def forward(
    scores: np.ndarray | pd.Series | float,
    baseline: Baseline | tuple[float, float],
) -> np.ndarray:
    """Map raw scores into the normalised training target.

    ``baseline`` is either a :class:`Baseline` aligned row for row with ``scores``, or a
    plain ``(centre, spread)`` pair when one baseline applies to everything.
    """
    centre, spread = _unpack(baseline)
    return (np.log1p(np.asarray(scores, dtype=np.float64)) - centre) / spread


def inverse(
    targets: np.ndarray | pd.Series | float,
    baseline: Baseline | tuple[float, float],
) -> np.ndarray:
    """Map normalised targets back to raw scores, for reporting.

    Exact inverse of :func:`forward` up to floating point. Both trailing statistics are
    knowable at prediction time, so this works in production, not only in a backtest.
    """
    centre, spread = _unpack(baseline)
    return np.expm1(np.asarray(targets, dtype=np.float64) * spread + centre)


def _unpack(baseline: Baseline | tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(baseline, Baseline):
        return baseline.centre, baseline.spread
    centre, spread = baseline
    if spread == 0:
        raise ValueError("baseline spread of zero would divide by zero")
    return np.float64(centre), np.float64(spread)
