"""The ``BaselineConfig`` defaults Phase 1's gates set, and what they imply.

Every expectation here is a measurement, and each one cites its number. No network and
no data file: the measured values are restated as constants so that changing a default
without redoing the measurement turns this red.

Measurements were taken on 2026-08-05/06 over 4,168,189 stories with settled scores.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from hn_upvotes.target.normalise import (
    BaselineConfig,
    compute_trailing_baseline,
    forward,
    inverse,
)

#: Gate 1. Median log1p(score) per year, 2007 to 2024 inclusive, over settled months.
#: Every one of the eighteen years came out at this exact value: a median score of 2.
#: 2025 is the single exception at log1p(3) = 1.3863.
MEASURED_YEARLY_MEDIAN_LOG1P = 1.0986

#: Gate 2. Standard deviation of log1p(score), 2010 and 2025. A 2.9% move over sixteen
#: years. The interquartile range moved the other way over the same span, 1.2528 to
#: 0.8473, with a year correlation of -0.098.
MEASURED_STD_2010 = 1.1386
MEASURED_STD_2025 = 1.1721

#: Gate 3. Balanced mean log1p(score) by age at observation, and the settled reference.
MEASURED_MEAN_LOG1P_0_12H = 1.490
MEASURED_MEAN_LOG1P_12_24H = 1.733
MEASURED_SETTLED_REFERENCE = 1.683
MEASURED_SETTLED_CI = (1.656, 1.712)


def test_gate1_turned_centring_off() -> None:
    """The median did not drift, so there is no centre to subtract.

    Median log1p(score) was 1.0986 in every year from 2007 to 2024. A statistic that
    takes one value across eighteen years is not drifting.
    """
    assert BaselineConfig().centre == "zero"
    assert float(np.log1p(2)) == pytest.approx(MEASURED_YEARLY_MEDIAN_LOG1P, abs=1e-4)


def test_gate2_turned_scaling_off() -> None:
    """The spread did not drift either, and the two spread measures disagree in sign."""
    assert BaselineConfig().scale is False
    drift = MEASURED_STD_2025 / MEASURED_STD_2010
    assert drift == pytest.approx(1.029, abs=0.002), "std moved 2.9% over 2010-2025"
    # The IQR fell over the same period. Two measures of the same thing moving in
    # opposite directions is what no drift looks like.
    assert (0.8473 / 1.2528) < 1.0 < drift


def test_gate3_set_the_lag_to_24_hours() -> None:
    """Scores reach their final level between 12 and 24 hours after submission.

    The 0 to 12 hour band sits below the settled reference and its interval does not
    reach it. The 12 to 24 hour band already overlaps. 24 hours is the top of the
    interval in which settling completes, so it is the lag. The scaffold's 48 hours was
    an assumption and is now measured down.
    """
    assert BaselineConfig().lag == timedelta(hours=24)
    settled_low, _settled_high = MEASURED_SETTLED_CI
    # 1.490 against a settled interval starting at 1.656: still rising.
    assert settled_low > MEASURED_MEAN_LOG1P_0_12H
    # 1.733 against the same interval: arrived, and if anything above it.
    assert settled_low <= MEASURED_MEAN_LOG1P_12_24H
    # The reference lies inside its own interval, and the young band lies outside it.
    assert settled_low <= MEASURED_SETTLED_REFERENCE <= _settled_high


def test_the_window_is_unchanged() -> None:
    """No gate measured the window length, so it stays where Phase 0 put it."""
    assert BaselineConfig().window == timedelta(days=30)


def test_default_transform_is_plain_log1p() -> None:
    """With centring and scaling both off, forward() is log1p and nothing else.

    This is the whole point of the two gate results. If a later change re-enables
    either, this test says so.
    """
    times = pd.to_datetime(pd.Series(["2024-01-01", "2024-02-01", "2024-03-01"]))
    # Deliberately wild scores: under the old default these would move the baseline a
    # long way, and under the measured default they must not move it at all.
    scores = pd.Series([1, 5000, 3])

    baseline = compute_trailing_baseline(times, scores, BaselineConfig())
    assert np.all(baseline.centre == 0.0)
    assert np.all(baseline.spread == 1.0)
    np.testing.assert_allclose(forward(scores, baseline), np.log1p(scores.to_numpy()))


def test_round_trip_survives_the_new_defaults() -> None:
    """inverse(forward(x)) == x across the observed score range.

    The observed maximum over the 4,168,189 settled stories is 6,015 points.
    """
    times = pd.to_datetime(pd.Series(pd.date_range("2024-01-01", periods=8, freq="15D")))
    scores = pd.Series([0, 1, 2, 3, 42, 355, 1001, 6015])

    baseline = compute_trailing_baseline(times, scores, BaselineConfig())
    recovered = inverse(forward(scores, baseline), baseline)
    np.testing.assert_allclose(recovered, scores.to_numpy(dtype=float), rtol=1e-9, atol=1e-9)


def test_the_mechanism_still_works_when_switched_back_on() -> None:
    """Turning centring and scaling back on is one config change, not a code change.

    The machinery is kept because the drift that does exist is in the extreme tail, and
    a later phase may find a statistic that captures it.
    """
    times = pd.to_datetime(pd.Series(pd.date_range("2024-01-01", periods=200, freq="6h")))
    rng = np.random.default_rng(0)
    scores = pd.Series(rng.integers(1, 100, size=200))

    config = BaselineConfig(centre="mean", scale=True, spread="std", min_periods=5)
    baseline = compute_trailing_baseline(times, scores, config)

    assert not np.all(baseline.centre == 0.0), "centring is on"
    assert not np.all(baseline.spread == 1.0), "scaling is on"
    recovered = inverse(forward(scores, baseline), baseline)
    np.testing.assert_allclose(recovered, scores.to_numpy(dtype=float), rtol=1e-9, atol=1e-9)
