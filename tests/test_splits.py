"""The single time-based split: no test row before a train row, no unsettled row at all.

Two things are being defended:

1. Every test row is later than every training row. If that is not true the whole result
   is meaningless, because the model saw the future.
2. Both unsettled windows are gone. Those months had their scores captured at
   submission, before anyone voted, so a row from them is a wrong label rather than a
   noisy one. The exclusion is read from ``ingest.UNSETTLED_SCORE_MONTHS``, so this test
   asks for the outcome and never restates the dates.

Synthetic rows only. No data file, no network.
"""

import pandas as pd
import pytest

from hn_upvotes.data.ingest import UNSETTLED_SCORE_MONTHS
from hn_upvotes.data.splits import (
    DEFAULT_SPLIT,
    SplitConfig,
    TemporalSplit,
    assert_no_temporal_overlap,
    split_stories,
    to_temporal_split,
)


def _monthly_frame() -> pd.DataFrame:
    """One row on the 15th of every month from 2006-01 to 2026-12."""
    months = pd.date_range("2006-01-15", "2026-12-15", freq="MS") + pd.Timedelta(days=14)
    return pd.DataFrame({"time": months, "score": range(len(months))})


def test_no_test_row_is_earlier_than_any_train_row():
    parts = split_stories(_monthly_frame())
    assert len(parts["train"]) > 0
    assert len(parts["test"]) > 0
    assert parts["train"]["time"].max() < parts["test"]["time"].min()


def test_neither_unsettled_window_survives_the_split():
    parts = split_stories(_monthly_frame())
    kept = pd.concat([parts["train"], parts["test"]])
    months = kept["time"].dt.strftime("%Y-%m")
    assert not set(months) & set(UNSETTLED_SCORE_MONTHS)


def test_the_boundary_months_are_inclusive():
    """`train_end` of 2022-11 keeps November and drops December."""
    parts = split_stories(_monthly_frame())
    train_months = set(parts["train"]["time"].dt.strftime("%Y-%m"))
    test_months = set(parts["test"]["time"].dt.strftime("%Y-%m"))
    assert "2022-11" in train_months
    assert "2022-12" not in train_months
    assert "2024-01" in test_months
    assert "2025-12" in test_months
    assert "2026-01" not in test_months


def test_the_default_boundaries_are_the_ones_that_were_measured():
    # Verified against data/stories.parquet on 2026-08-06: these four boundaries give
    # 3,568,252 train rows and 599,937 test rows, which is what the README quotes.
    assert (DEFAULT_SPLIT.train_start, DEFAULT_SPLIT.train_end) == ("2006-10", "2022-11")
    assert (DEFAULT_SPLIT.test_start, DEFAULT_SPLIT.test_end) == ("2024-01", "2025-12")


def test_a_split_that_would_train_on_the_future_is_rejected():
    overlapping = SplitConfig(
        train_start="2006-10", train_end="2024-06", test_start="2024-01", test_end="2025-12"
    )
    with pytest.raises(ValueError, match="after test starts"):
        to_temporal_split(overlapping)


def test_an_empty_interval_is_rejected():
    backwards = TemporalSplit(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2019-01-01"),
        test_start=pd.Timestamp("2024-01-01"),
        test_end=pd.Timestamp("2026-01-01"),
    )
    with pytest.raises(ValueError, match="train interval is empty"):
        assert_no_temporal_overlap(backwards)
