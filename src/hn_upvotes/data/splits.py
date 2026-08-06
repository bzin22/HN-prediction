"""Temporal splitting. There are no random splits anywhere in this project.

A random split lets the model train on posts submitted after the ones it is tested on,
which is information no submission-time predictor could have. Every split here is a cut
on the time axis, and the walk-forward harness rolls that cut forward so the README can
show whether a model degrades as it ages rather than quoting one number from one cut.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """One train/validation/test cut, expressed as half-open time intervals.

    Every interval is ``[start, end)``. Validation starts where training ends and test
    starts where validation ends, so no row can land in two of them.
    """

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_end: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def val_start(self) -> pd.Timestamp:
        return self.train_end

    @property
    def test_start(self) -> pd.Timestamp:
        return self.val_end


def time_ordered_split(
    times: pd.Series,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> TemporalSplit:
    """Cut a single train/validation/test split at time quantiles.

    Test is the most recent ``test_fraction`` of rows by time, validation the slice
    before it, training everything earlier. Used for quick iteration. The reported
    numbers come from :func:`walk_forward_folds`.
    """
    raise NotImplementedError


def walk_forward_folds(
    times: pd.Series,
    train_window: timedelta,
    test_window: timedelta,
    step: timedelta,
    val_window: timedelta | None = None,
) -> Iterator[TemporalSplit]:
    """Yield successive walk-forward folds over the time axis.

    Train on a window, test on the window immediately after, roll forward by ``step``,
    repeat. Drift inside any one fold is small, and the sequence of scores across folds
    shows how the model ages.
    """
    raise NotImplementedError


def apply_split(frame: pd.DataFrame, split: TemporalSplit, time_column: str = "time") -> dict:
    """Slice a frame into train, validation and test parts for one split.

    Returns a mapping with keys ``train``, ``val`` and ``test``.
    """
    raise NotImplementedError


def assert_no_temporal_overlap(split: TemporalSplit) -> None:
    """Raise if the intervals are out of order or overlap.

    A cheap invariant to assert at the top of every training run, so a bad config fails
    before it burns an hour of MPS time.
    """
    raise NotImplementedError


def rows_before(times: np.ndarray, cutoff: pd.Timestamp) -> np.ndarray:
    """Boolean mask of rows strictly earlier than ``cutoff``.

    The single helper every expanding-window statistic goes through, so the strictness
    of the comparison is defined in one place.
    """
    raise NotImplementedError
