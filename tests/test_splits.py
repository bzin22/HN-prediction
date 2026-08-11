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
    DEFAULT_VALIDATION_START,
    SplitConfig,
    TemporalSplit,
    assert_no_temporal_overlap,
    cut_validation_tail,
    sample_validation_tail,
    split_stories,
    to_temporal_split,
    validation_tail_mask,
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


def test_no_validation_row_is_earlier_than_any_row_fitted_on():
    """The property early stopping depends on. A random slice would break it."""
    train = split_stories(_monthly_frame())["train"]
    parts = cut_validation_tail(train)
    assert len(parts["fit"]) > 0
    assert len(parts["validation"]) > 0
    assert parts["fit"]["time"].max() < parts["validation"]["time"].min()


def test_the_validation_tail_takes_the_end_of_the_training_period_and_nothing_else():
    train = split_stories(_monthly_frame())["train"]
    parts = cut_validation_tail(train)
    # No row is lost or duplicated, and the tail ends where training ends.
    assert len(parts["fit"]) + len(parts["validation"]) == len(train)
    assert parts["validation"]["time"].max() == train["time"].max()
    assert set(parts["validation"]["time"].dt.strftime("%Y-%m")) == {
        "2021-12", "2022-01", "2022-02", "2022-03", "2022-04", "2022-05",
        "2022-06", "2022-07", "2022-08", "2022-09", "2022-10", "2022-11",
    }  # fmt: skip


def test_the_validation_tail_never_reaches_the_test_period():
    parts = split_stories(_monthly_frame())
    tail = cut_validation_tail(parts["train"])["validation"]
    assert tail["time"].max() < parts["test"]["time"].min()


def test_the_default_validation_start_is_twelve_months_of_training_data():
    # Verified against data/stories.parquet on 2026-08-07: 2021-12 onward is 296,531 of
    # the 3,568,252 training rows, 8.3%, leaving 3,271,721 to fit on.
    assert DEFAULT_VALIDATION_START == "2021-12"
    boundary = pd.Timestamp(f"{DEFAULT_VALIDATION_START}-01")
    train_end = pd.Timestamp(f"{DEFAULT_SPLIT.train_end}-01") + pd.offsets.MonthBegin(1)
    assert (train_end.year - boundary.year) * 12 + train_end.month - boundary.month == 12


def test_sampling_the_tail_can_only_ever_drop_rows_never_add_them():
    """The property that keeps sampling from touching strictly-earlier.

    A sampled row was already in the tail, so it is still later than every fitted row. The
    danger would be a sampler that reached back across the boundary, and a mask that is a
    strict subset of the tail mask cannot.
    """
    train = split_stories(_monthly_frame())["train"]
    in_tail = validation_tail_mask(train["time"])
    sampled = sample_validation_tail(in_tail, sample_rows=4, seed=0)

    assert sampled.sum() == 4
    assert not (sampled & ~in_tail).any(), "sampling invented a row outside the tail"


def test_sampling_is_seeded_so_a_rerun_scores_the_same_rows():
    train = split_stories(_monthly_frame())["train"]
    in_tail = validation_tail_mask(train["time"])
    first = sample_validation_tail(in_tail, sample_rows=5, seed=0)
    assert (first == sample_validation_tail(in_tail, sample_rows=5, seed=0)).all()
    assert not (first == sample_validation_tail(in_tail, sample_rows=5, seed=1)).all()


def test_asking_for_more_rows_than_the_tail_holds_keeps_all_of_them():
    train = split_stories(_monthly_frame())["train"]
    in_tail = validation_tail_mask(train["time"])
    assert (sample_validation_tail(in_tail, sample_rows=10_000) == in_tail).all()
    assert (sample_validation_tail(in_tail, sample_rows=None) == in_tail).all()


def test_sampling_does_not_move_a_dropped_row_into_the_fit_set():
    """A row held back but not scored is scored by nothing, and fitted on by nothing."""
    train = split_stories(_monthly_frame())["train"]
    unsampled = cut_validation_tail(train)
    sampled = cut_validation_tail(train, sample_rows=3)

    assert len(sampled["validation"]) == 3
    assert len(sampled["fit"]) == len(unsampled["fit"]), "the fit set has to be untouched"
    assert sampled["fit"]["time"].max() < sampled["validation"]["time"].min()


def test_a_validation_tail_with_nothing_on_one_side_is_rejected():
    train = split_stories(_monthly_frame())["train"]
    with pytest.raises(ValueError, match="validation tail is empty"):
        cut_validation_tail(train, "2030-01")
    with pytest.raises(ValueError, match="nothing left to fit on"):
        cut_validation_tail(train, "2000-01")


def test_an_empty_interval_is_rejected():
    backwards = TemporalSplit(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2019-01-01"),
        test_start=pd.Timestamp("2024-01-01"),
        test_end=pd.Timestamp("2026-01-01"),
    )
    with pytest.raises(ValueError, match="train interval is empty"):
        assert_no_temporal_overlap(backwards)
