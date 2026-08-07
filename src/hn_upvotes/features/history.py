"""Prior means over strictly earlier rows. One implementation, shared by every rung.

A key is an author name or a hostname. For a row at time ``t``, its prior is the mean of
the target over that key's rows earlier than ``t - lag``. The bound is the same one
``target.normalise.compute_trailing_baseline`` uses and it is the project's second
protected invariant: a March post's author mean reads that author's February posts and
never their April ones.

Two bounds, not one:

* **Strictly earlier.** The upper end of the window is exclusive, so a row can never
  reach itself or anything submitted at the same instant.
* **Settling lag.** Rows inside the last ``lag`` are excluded too. A post from three
  hours ago is still collecting votes, so its score is not a fact yet.

The query rows and the history rows are separate arguments on purpose. Evaluating the
test period means asking about test rows against a history that runs from 2006 through
the test period itself, and only the per-row bound keeps that honest. It is also what a
deployed predictor would have: a system running in March 2024 does know what January
2024 posts scored.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from hn_upvotes.target.normalise import to_datetime

#: Sort and search key. Key first, then time, so one lexicographic ``searchsorted``
#: answers "how many of this key's rows are older than this instant" for every row at
#: once. Structured comparison is field order, which is exactly the ordering wanted.
_KEY_TIME = np.dtype([("key", np.int64), ("time", np.int64)])


@dataclass(frozen=True)
class PriorConfig:
    """How a prior is computed.

    window
        ``None`` expands from the start of the history. A finite window makes the
        statistic recency weighted instead.
    lag
        Settling lag, matching ``target.normalise.BaselineConfig.lag``. Phase 1 measured
        scores reaching their final level between 12 and 24 hours after submission.
    min_rows
        Below this many prior rows the mean comes back as ``NaN`` and the caller picks a
        fallback. A mean of one post is not a track record.
    """

    window: timedelta | None = None
    lag: timedelta = timedelta(hours=24)
    min_rows: int = 1

    def __post_init__(self) -> None:
        if self.window is not None and self.window <= timedelta(0):
            raise ValueError(f"window must be positive or None, got {self.window}")
        if self.lag < timedelta(0):
            raise ValueError(f"lag cannot be negative, got {self.lag}")
        if self.min_rows < 1:
            raise ValueError(f"min_rows must be at least 1, got {self.min_rows}")


@dataclass(frozen=True)
class Prior:
    """Per-query-row prior, aligned to the query rows.

    ``mean`` is ``NaN`` wherever ``count`` is below ``PriorConfig.min_rows``, so a caller
    that forgets to supply a fallback gets a loud NaN rather than a quiet zero.
    """

    mean: np.ndarray
    count: np.ndarray

    def __len__(self) -> int:
        return int(self.count.shape[0])


def prior_mean(
    query_times: pd.Series | np.ndarray,
    query_keys: pd.Series | np.ndarray,
    history_times: pd.Series | np.ndarray,
    history_keys: pd.Series | np.ndarray,
    history_values: pd.Series | np.ndarray,
    config: PriorConfig | None = None,
) -> Prior:
    """Mean of ``history_values`` over each query row's own key, strictly earlier.

    Pass a constant ``query_keys`` and ``history_keys`` to get one global trailing mean
    rather than a per-key one. That is rung 1 of the baseline ladder.

    Input order does not matter and results come back aligned to the query rows.
    """
    config = config or PriorConfig()

    query_time_ns = _nanoseconds(query_times)
    history_time_ns = _nanoseconds(history_times)
    values = np.asarray(history_values, dtype=np.float64)
    if len(history_time_ns) != len(values):
        raise ValueError(
            f"history times and values differ in length: {len(history_time_ns)} vs {len(values)}"
        )

    query_code, history_code = _shared_codes(query_keys, history_keys)
    if len(query_code) != len(query_time_ns):
        raise ValueError(
            f"query times and keys differ in length: {len(query_time_ns)} vs {len(query_code)}"
        )

    n_query = len(query_time_ns)
    if len(values) == 0:
        return Prior(np.full(n_query, np.nan), np.zeros(n_query, dtype=np.int64))

    history = np.empty(len(values), dtype=_KEY_TIME)
    history["key"] = history_code
    history["time"] = history_time_ns
    order = np.argsort(history, kind="stable")
    history = history[order]
    values = values[order]

    upper = np.empty(n_query, dtype=_KEY_TIME)
    upper["key"] = query_code
    upper["time"] = query_time_ns - pd.Timedelta(config.lag).value
    # side="left" is what makes the bound strict: a history row at exactly the cutoff is
    # excluded, so a row can never contribute to its own prior.
    hi = np.searchsorted(history, upper, side="left")

    if config.window is None:
        # Expanding: start at the first row this key has, whenever that was.
        lo = np.searchsorted(history["key"], query_code, side="left")
    else:
        lower = upper.copy()
        lower["time"] = upper["time"] - pd.Timedelta(config.window).value
        lo = np.searchsorted(history, lower, side="left")

    count = (hi - lo).astype(np.int64)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    total = cumulative[hi] - cumulative[lo]
    mean = np.where(count >= config.min_rows, total / np.maximum(count, 1), np.nan)
    return Prior(mean, count)


def _nanoseconds(times: pd.Series | np.ndarray) -> np.ndarray:
    """Timestamps as int64 nanoseconds, through the one coercion the project uses."""
    return to_datetime(times).to_numpy().astype("datetime64[ns]").view(np.int64)


def _shared_codes(
    query_keys: pd.Series | np.ndarray,
    history_keys: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Factorise both key columns together, so the two sides of the search agree.

    Missing values become ``""`` rather than a distinct NaN code. In this dump an absent
    URL is already ``""`` (see ``data/ingest``), so the two spellings of "no link" have
    to land in the same bucket.
    """
    query = pd.Series(query_keys).reset_index(drop=True).fillna("").astype(str)
    history = pd.Series(history_keys).reset_index(drop=True).fillna("").astype(str)
    codes, _ = pd.factorize(pd.concat([query, history], ignore_index=True))
    return codes[: len(query)].astype(np.int64), codes[len(query) :].astype(np.int64)
