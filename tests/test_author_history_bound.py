"""An author's prior reads their strictly earlier posts and nothing else.

This is the project's second protected invariant and it is the easy mistake in the
baseline ladder: take an author's mean over the whole table, then apply it to every row
of that author. That scores beautifully and is worthless, because the March prediction
was made with April's data.

The fixture is built so a broken bound cannot pass quietly. Alice's second post is worth
999 points against a first post worth 9, so any prediction that reaches forward moves by
more than a factor of two and the assertion says which way it broke.

Synthetic rows only. No data file, no network.
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from hn_upvotes.models.baselines import AuthorMeanPredictor, TrailingMeanPredictor

# log1p of the three scores Alice posts, which is the space every rung predicts in.
FIRST = np.log1p(9.0)  # 2.3026
SECOND = np.log1p(999.0)  # 6.9078


def _frame() -> pd.DataFrame:
    """Alice posts in January, February and March. Bob posts once, enormously.

    Bob is here so a bound that leaks across authors fails as loudly as one that leaks
    across time: his 99,999 points would swamp any mean he reached.
    """
    return pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01", "2024-01-15"]),
            "by": ["alice", "alice", "alice", "bob"],
            "score": [9, 999, 3, 99_999],
            "title": ["a", "b", "c", "d"],
            "url": ["", "", "", ""],
        }
    )


def _predict(frame: pd.DataFrame) -> np.ndarray:
    targets = np.log1p(frame["score"].to_numpy(dtype=np.float64))
    model = AuthorMeanPredictor(fallback=TrailingMeanPredictor(min_rows=1), min_posts=1)
    return model.fit(frame, targets).predict(frame)


def test_a_row_sees_only_that_authors_earlier_posts():
    predictions = _predict(_frame())

    # February reads January alone: log1p(9) = 2.3026.
    # Reaching its own row would give (2.3026 + 6.9078) / 2 = 4.6052.
    # Reaching March as well would give (2.3026 + 6.9078 + 1.3863) / 3 = 3.5322.
    assert predictions[1] == pytest.approx(FIRST)

    # March reads January and February: (2.3026 + 6.9078) / 2 = 4.6052.
    assert predictions[2] == pytest.approx((FIRST + SECOND) / 2)


def test_another_authors_posts_never_reach_the_prior():
    """Bob's 99,999 points sit between Alice's January and February posts."""
    predictions = _predict(_frame())
    # If Bob leaked into Alice's March prior it would be (2.3026 + 6.9078 + 11.5129) / 3
    # = 6.9078, which is exactly SECOND, so assert the clean value instead.
    assert predictions[2] == pytest.approx((FIRST + SECOND) / 2)
    assert predictions[2] < np.log1p(99_999.0)


def test_the_settling_lag_holds_back_a_post_from_this_morning():
    """A post 12 hours old has not finished scoring, so it is not a track record yet."""
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(["2024-01-01 00:00", "2024-02-01 00:00", "2024-02-01 12:00"]),
            "by": ["alice", "alice", "alice"],
            "score": [9, 999, 3],
            "title": ["a", "b", "c"],
            "url": ["", "", ""],
        }
    )
    predictions = _predict(frame)
    # The third row is 12 hours after the second, inside the 24 hour lag, so its prior is
    # January alone: log1p(9) = 2.3026. Without the lag it would be 4.6052.
    assert predictions[2] == pytest.approx(FIRST)


def test_an_author_with_no_earlier_post_falls_back_to_the_rung_below():
    frame = _frame()
    predictions = _predict(frame)
    targets = np.log1p(frame["score"].to_numpy(dtype=np.float64))
    rung_one = TrailingMeanPredictor(min_rows=1).fit(frame, targets).predict(frame)

    # Alice's January post is the first row in the table, so it has no history of any
    # kind and rung 1 has nothing either: both come out at 0.0.
    assert predictions[0] == pytest.approx(rung_one[0])
    # Bob's only post falls back to rung 1, which by then has seen Alice's January post.
    assert predictions[3] == pytest.approx(rung_one[3])
    assert rung_one[3] == pytest.approx(FIRST)


def test_a_widening_history_never_moves_an_earlier_answer():
    """Predict on every prefix of the table. No answer may change as rows are added."""
    frame = _frame().sort_values("time").reset_index(drop=True)
    full = _predict(frame)
    for i in range(1, len(frame) + 1):
        partial = _predict(frame.iloc[:i])
        assert partial == pytest.approx(full[:i]), f"row {i - 1} moved when later rows arrived"


def test_min_posts_holds_back_a_track_record_of_one():
    frame = _frame()
    targets = np.log1p(frame["score"].to_numpy(dtype=np.float64))
    strict = AuthorMeanPredictor(fallback=TrailingMeanPredictor(min_rows=1), min_posts=2)
    predictions = strict.fit(frame, targets).predict(frame)

    # February has one earlier post, below the minimum of two, so it falls back.
    assert predictions[1] != pytest.approx(FIRST)
    # March has two and keeps its own mean.
    assert predictions[2] == pytest.approx((FIRST + SECOND) / 2)


def test_the_window_can_forget_old_posts():
    """A finite window is the recency-weighted variant. Rung 2 does not use one."""
    frame = _frame()
    targets = np.log1p(frame["score"].to_numpy(dtype=np.float64))
    model = AuthorMeanPredictor(fallback=TrailingMeanPredictor(min_rows=1), min_posts=1)
    model.config = model.config.__class__(
        window=timedelta(days=20), lag=model.config.lag, min_rows=1
    )
    predictions = model.fit(frame, targets).predict(frame)

    # March 1 with a 20 day window looks back to February 10, so January is forgotten and
    # only February counts: log1p(999) = 6.9078, against 4.6052 for the expanding version.
    assert predictions[2] == pytest.approx(SECOND)
