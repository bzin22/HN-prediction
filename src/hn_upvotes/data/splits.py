"""Temporal splitting. There are no random splits anywhere in this project.

A random split lets the model train on posts submitted after the ones it is tested on,
which is information no submission-time predictor could have. Every split here is a cut
on the time axis.

Phase 2 uses a single cut, train on 2006-10 to 2022-11 and test on 2024-01 to 2025-12.
The 13 months in between, and everything from 2026-01, are dropped because the archive
captured those scores at submission before anyone had voted. That exclusion is not
written here: it is read from ``ingest.UNSETTLED_SCORE_MONTHS`` through
``ingest.drop_unsettled_months``, so it cannot drift from its source of truth.

The boundaries are :class:`SplitConfig`, not literals in code, so moving the cut is a
config change and shows up as one in a diff.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from hn_upvotes.data.ingest import drop_unsettled_months


@dataclass(frozen=True)
class SplitConfig:
    """Where the cut falls, as inclusive ``YYYY-MM`` months.

    Months rather than instants because the thing being avoided is measured in months:
    the archive lost whole months of settled scores at a time.

    The defaults are the Phase 2 split. Measured against
    ``data/stories.parquet`` on 2026-08-06: 3,568,252 training rows, 599,937 test rows,
    and 569,815 rows dropped in the two unsettled windows (345,029 and 224,786).
    """

    train_start: str = "2006-10"
    train_end: str = "2022-11"
    test_start: str = "2024-01"
    test_end: str = "2025-12"


#: The Phase 2 split. Import this rather than constructing a config at a call site.
DEFAULT_SPLIT = SplitConfig()


@dataclass(frozen=True)
class TemporalSplit:
    """One train/test cut, as half-open ``[start, end)`` intervals.

    Train and test do not have to touch, and here they do not. The gap between them is
    the first unsettled window, and a split that closed the gap would be reading scores
    that were captured before anyone voted on them.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def to_temporal_split(config: SplitConfig = DEFAULT_SPLIT) -> TemporalSplit:
    """Turn the month boundaries into half-open timestamp intervals.

    The ``_end`` months are inclusive in the config and exclusive here, so ``train_end``
    of ``"2022-11"`` becomes ``2022-12-01``, which keeps every November row.
    """
    split = TemporalSplit(
        train_start=_month_start(config.train_start),
        train_end=_month_after(config.train_end),
        test_start=_month_start(config.test_start),
        test_end=_month_after(config.test_end),
    )
    assert_no_temporal_overlap(split)
    return split


def split_stories(
    frame: pd.DataFrame,
    config: SplitConfig = DEFAULT_SPLIT,
    time_column: str = "time",
) -> dict[str, pd.DataFrame]:
    """Drop the unsettled months, then cut into ``train`` and ``test``.

    This is the only entry point anything reading ``score`` as a label should use. The
    unsettled drop comes first and is unconditional, so the split boundaries are a
    second line of defence rather than the only one.
    """
    settled = drop_unsettled_months(frame)
    return apply_split(settled, to_temporal_split(config), time_column=time_column)


def apply_split(
    frame: pd.DataFrame,
    split: TemporalSplit,
    time_column: str = "time",
) -> dict[str, pd.DataFrame]:
    """Slice a frame into its train and test parts. Rows in the gap are dropped."""
    assert_no_temporal_overlap(split)
    times = pd.to_datetime(frame[time_column])
    in_train = (times >= split.train_start) & (times < split.train_end)
    in_test = (times >= split.test_start) & (times < split.test_end)
    return {"train": frame.loc[in_train].copy(), "test": frame.loc[in_test].copy()}


def assert_no_temporal_overlap(split: TemporalSplit) -> None:
    """Raise if the intervals are out of order or overlap.

    Cheap enough to assert at the top of every run, so a bad config fails in a second
    rather than after an hour of fitting.
    """
    if split.train_start >= split.train_end:
        raise ValueError(f"train interval is empty: {split.train_start} to {split.train_end}")
    if split.test_start >= split.test_end:
        raise ValueError(f"test interval is empty: {split.test_start} to {split.test_end}")
    if split.train_end > split.test_start:
        raise ValueError(
            f"train ends at {split.train_end}, after test starts at {split.test_start}. "
            "A test row earlier than a training row makes the whole result meaningless."
        )


def rows_before(times: np.ndarray, cutoff: pd.Timestamp) -> np.ndarray:
    """Boolean mask of rows strictly earlier than ``cutoff``.

    The single helper every expanding-window statistic goes through, so the strictness
    of the comparison is defined in one place.
    """
    return pd.to_datetime(pd.Series(times)).to_numpy() < np.datetime64(pd.Timestamp(cutoff))


def walk_forward_folds(
    times: pd.Series,
    train_window: timedelta,
    test_window: timedelta,
    step: timedelta,
) -> Iterator[TemporalSplit]:
    """Yield successive walk-forward folds over the time axis.

    **Deferred.** Phase 2 chose a single cut, on the grounds that walk-forward is worth
    its cost once one number is shown to be hiding something. The thing it would expose
    is tail drift, and the two metrics that would show it, Spearman and Precision@100,
    are already reported on the single cut.
    """
    raise NotImplementedError("deferred: Phase 2 reports a single cut. See the docstring.")


def _month_start(month: str) -> pd.Timestamp:
    return pd.Timestamp(f"{month}-01")


def _month_after(month: str) -> pd.Timestamp:
    return _month_start(month) + pd.offsets.MonthBegin(1)
