"""Target round-trip guard, and the strictly-earlier rule for the trailing baseline.

Two things are being defended:

1. ``inverse(forward(x)) == x``. If the transform is not invertible, every score the
   project reports is wrong, and the error is invisible because the normalised metrics
   still look fine.
2. A row's baseline sees only rows strictly earlier than its own timestamp minus the
   settling lag. If a later row can move an earlier row's baseline, the backtest is
   using information that did not exist at submission time and its numbers are fake.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from hn_upvotes.target.normalise import (
    Baseline,
    BaselineConfig,
    compute_trailing_baseline,
    forward,
    inverse,
)

# Spans the range that matters: 0 and 1 are the floor spike (a large share of HN posts
# score 1 or 2), 300 is a front-page post, 3000 is the tail.
SCORES = np.array([0.0, 1.0, 2.0, 5.0, 17.0, 50.0, 183.0, 300.0, 3000.0])


def test_round_trip_recovers_the_raw_score():
    baseline = (2.9, 1.3)  # illustrative centre and spread, not measured
    assert inverse(forward(SCORES, baseline), baseline) == pytest.approx(SCORES, rel=1e-12)


def test_round_trip_holds_with_the_divide_by_spread_step_off():
    # Phase 1 may decide subtracting a trailing centre is enough. Removing the divide
    # must not break invertibility.
    baseline = (2.9, 1.0)
    assert inverse(forward(SCORES, baseline), baseline) == pytest.approx(SCORES, rel=1e-12)


def test_round_trip_holds_against_a_per_row_computed_baseline():
    frame = _synthetic_frame()
    config = BaselineConfig(min_periods=1)
    baseline = compute_trailing_baseline(frame["time"], frame["score"], config)
    recovered = inverse(forward(frame["score"], baseline), baseline)
    assert recovered == pytest.approx(frame["score"].to_numpy(), rel=1e-12)


def test_zero_score_survives_the_round_trip_exactly():
    # log1p(0) is 0, so this is the one input where a naive log would blow up.
    baseline = (1.8, 1.2)
    assert inverse(forward(np.array([0.0]), baseline), baseline) == pytest.approx([0.0], abs=1e-12)


def test_later_rows_cannot_change_an_earlier_rows_baseline():
    """Compute each prefix of the frame and check the answers never move."""
    frame = _synthetic_frame()
    config = BaselineConfig(min_periods=1)
    full = compute_trailing_baseline(frame["time"], frame["score"], config)

    for i in range(1, len(frame) + 1):
        prefix = frame.iloc[:i]
        partial = compute_trailing_baseline(prefix["time"], prefix["score"], config)
        assert partial.centre == pytest.approx(full.centre[:i]), f"row {i - 1} moved"
        assert partial.spread == pytest.approx(full.spread[:i]), f"row {i - 1} moved"


def test_the_settling_lag_excludes_recent_rows():
    """A big score one day before the target row must not reach its baseline."""
    target_time = pd.Timestamp("2024-03-01")
    times = pd.Series(
        [
            target_time - timedelta(days=10),
            target_time - timedelta(days=5),
            target_time - timedelta(days=3),
            target_time - timedelta(days=1),  # inside the 48h lag, must be excluded
            target_time,
        ]
    )
    scores = pd.Series([10, 10, 10, 100_000, 7])

    # centre="mean" is set explicitly because this test is about which rows land in the
    # window, not about the default. Phase 1 measured no centre drift and turned
    # centring off by default, which would zero the statistic this test reads.
    config = BaselineConfig(
        window=timedelta(days=30), lag=timedelta(hours=48), min_periods=1, centre="mean"
    )
    baseline = compute_trailing_baseline(times, scores, config)

    # Only the three rows at 10 points qualify, so the centre is log1p(10) = 2.3979.
    # If the 100k row leaked in, the mean would be about 4.6.
    assert baseline.centre[-1] == pytest.approx(np.log1p(10.0))
    assert baseline.n_prior[-1] == 3


def test_rows_outside_the_window_are_excluded():
    target_time = pd.Timestamp("2024-06-01")
    times = pd.Series(
        [
            target_time - timedelta(days=200),  # older than the 30 day window
            target_time - timedelta(days=10),
            target_time,
        ]
    )
    scores = pd.Series([100_000, 10, 7])

    # centre="mean" for the same reason as above: the subject is the window bound.
    baseline = compute_trailing_baseline(
        times, scores, BaselineConfig(min_periods=1, centre="mean")
    )
    assert baseline.centre[-1] == pytest.approx(np.log1p(10.0))
    assert baseline.n_prior[-1] == 1


def test_rows_with_no_history_fall_back_instead_of_dividing_by_zero():
    times = pd.Series([pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")])
    scores = pd.Series([5, 5])
    config = BaselineConfig(fallback_centre=0.0, fallback_spread=1.0)

    baseline = compute_trailing_baseline(times, scores, config)
    assert baseline.n_prior[0] == 0
    assert baseline.centre[0] == 0.0
    assert baseline.spread[0] == 1.0
    # With the fallback in place the transform degenerates to plain log1p.
    assert forward(scores, baseline)[0] == pytest.approx(np.log1p(5.0))


def test_input_order_does_not_change_the_result():
    frame = _synthetic_frame()
    config = BaselineConfig(min_periods=1)
    ordered = compute_trailing_baseline(frame["time"], frame["score"], config)

    shuffled = frame.iloc[::-1].reset_index(drop=True)
    reversed_result = compute_trailing_baseline(shuffled["time"], shuffled["score"], config)

    assert reversed_result.centre == pytest.approx(ordered.centre[::-1])


def test_scale_off_leaves_the_spread_at_one():
    frame = _synthetic_frame()
    baseline = compute_trailing_baseline(
        frame["time"], frame["score"], BaselineConfig(scale=False, min_periods=1)
    )
    assert np.all(baseline.spread == 1.0)


def test_unix_seconds_are_read_as_timestamps():
    """The source dump stores `time` as unix seconds, so integers must work directly."""
    base = pd.Timestamp("2024-01-01").value // 10**9
    seconds = pd.Series([base, base + 86_400 * 10, base + 86_400 * 20])
    scores = pd.Series([10, 10, 7])

    from_seconds = compute_trailing_baseline(seconds, scores, BaselineConfig(min_periods=1))
    from_datetimes = compute_trailing_baseline(
        pd.to_datetime(seconds, unit="s"), scores, BaselineConfig(min_periods=1)
    )
    assert from_seconds.centre == pytest.approx(from_datetimes.centre)
    assert isinstance(from_seconds, Baseline)


def _synthetic_frame() -> pd.DataFrame:
    """Daily rows over 60 days, with a deliberate spike late in the series.

    The spike is what the strictly-earlier test is looking for. If it moves the baseline
    of any row before it, the window bound is wrong.
    """
    start = pd.Timestamp("2024-01-01")
    times = [start + timedelta(days=i) for i in range(60)]
    scores = [3 + (i % 7) for i in range(60)]
    scores[50] = 5000
    return pd.DataFrame({"time": times, "score": scores})
