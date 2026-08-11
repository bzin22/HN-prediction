"""Rung 6 with early stopping: the tail it stops on is later than everything it fits on.

Two things are being defended:

1. **The validation slice is a cut on the time axis and the model never sees it while
   fitting.** Not the trees, and not the vectorisers either: a vocabulary built over the
   validation rows would let the model stop on a number it had already read. This is the
   strictly-earlier invariant, applied to the third slice.
2. **Early stopping actually stops.** A run that silently reaches its ceiling has measured
   the ceiling, and the result would read as an answer when it is not one.

Synthetic rows only. No data file, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# xgboost needs an OpenMP runtime and raises XGBoostError, not ImportError, when it cannot
# find one, so importorskip does not cover it. On macOS the runtime is missing until
# `brew install libomp`; `make tune-xgboost` points DYLD_LIBRARY_PATH at scikit-learn's own
# copy. The Linux wheel CI uses bundles it, so the embeddings job runs this file for real.
try:
    import xgboost  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"xgboost unavailable: {exc}", allow_module_level=True)

from hn_upvotes.models.baselines import (  # noqa: E402
    AuthorMeanPredictor,
    DomainMeanPredictor,
    EarlyStoppedXGBoost,
    TrailingMeanPredictor,
)

VALIDATION_START = "2022-01"

#: A word planted in the validation tail and nowhere else. If it reaches the vocabulary,
#: the vectorisers were fitted over rows the model is supposed to be judged on.
TAIL_ONLY_WORD = "quokka"


def _frame(rows: int = 600, tail_rows: int = 120, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Rows either side of the validation boundary, in shuffled row order.

    Shuffled on purpose. The cut has to come from the timestamps, not from the order the
    rows happen to arrive in.
    """
    rng = np.random.default_rng(seed)
    early = pd.date_range("2020-01-01", periods=rows - tail_rows, freq="12h")
    tail = pd.date_range(f"{VALIDATION_START}-01", periods=tail_rows, freq="12h")
    times = pd.DatetimeIndex(list(early) + list(tail))

    words = ["python", "rust", "startup", "database", "compiler"]
    titles = [f"{words[i % len(words)]} release {i}" for i in range(rows)]
    titles[-tail_rows:] = [f"{TAIL_ONLY_WORD} {t}" for t in titles[-tail_rows:]]

    frame = pd.DataFrame(
        {
            "time": times,
            "title": titles,
            "text": ["" if i % 3 else f"some body text number {i}" for i in range(rows)],
            "url": [f"https://host{i % 7}.example.com/{i}" for i in range(rows)],
            "by": [f"author{i % 23}" for i in range(rows)],
        }
    )
    # Real signal plus noise: the author carries most of it. Pure noise would make the
    # first round the best round, and then there would be no truncation to test.
    author_effect = np.array([2 + 4 * (i % 23) for i in range(rows)], dtype=np.float64)
    targets = np.log1p(author_effect + rng.integers(0, 10, size=rows))

    order = rng.permutation(rows)
    return frame.iloc[order].reset_index(drop=True), targets[order]


def _fitted(frame: pd.DataFrame, targets: np.ndarray, **kwargs) -> EarlyStoppedXGBoost:
    author = AuthorMeanPredictor(fallback=TrailingMeanPredictor(), min_posts=1)
    author.fit(frame, targets)
    domain = DomainMeanPredictor(fallback=TrailingMeanPredictor(), min_posts=1)
    domain.fit(frame, targets)
    settings = {
        "first_validation_month": VALIDATION_START,
        "max_features": 200,
        "n_estimators": 80,
        "early_stopping_rounds": 5,
        "verbose_every": 0,
        "validation_sample_rows": None,
        **kwargs,
    }
    return EarlyStoppedXGBoost(author, domain, **settings).fit(frame, targets)


def test_the_validation_tail_is_cut_by_time_not_by_row_order():
    frame, targets = _frame()
    model = _fitted(frame, targets, validation_sample_rows=None)

    boundary = pd.Timestamp(f"{VALIDATION_START}-01")
    assert model.held_back_rows == int((frame["time"] >= boundary).sum())
    assert model.validation_rows == model.held_back_rows
    assert model.fit_rows == len(frame) - model.held_back_rows
    assert model.fit_rows > 0 and model.validation_rows > 0


def test_sampling_the_scored_rows_does_not_grow_the_fit_set():
    """Rows held back but not scored are not handed back to training.

    The whole point of the bound is per-round cost. If the dropped rows fell back into the
    fit set, the sampled run and the unsampled run would be fitting different models and
    nothing could be compared between them.
    """
    frame, targets = _frame()
    unsampled = _fitted(frame, targets, validation_sample_rows=None)
    sampled = _fitted(frame, targets, validation_sample_rows=40)

    assert sampled.validation_rows == 40
    assert sampled.held_back_rows == unsampled.held_back_rows
    assert sampled.fit_rows == unsampled.fit_rows


def test_the_vectorisers_never_see_the_validation_tail():
    """A word that appears only after the boundary must not be in the vocabulary.

    The trees are the obvious leak and the vocabulary is the quiet one. Fitting TF-IDF on
    the whole training split, then stopping on part of it, tunes the stopping point using
    text the model already had.
    """
    frame, targets = _frame()
    model = _fitted(frame, targets)

    vocabulary = model.features.title.vectoriser.vocabulary_
    assert TAIL_ONLY_WORD in " ".join(frame["title"]), "the planted word has to be present"
    assert not any(TAIL_ONLY_WORD in term for term in vocabulary)


def test_early_stopping_ends_the_run_before_the_ceiling():
    """The signal here is one author effect, so it runs out of things to learn quickly.

    If this ever runs to the ceiling, the tree count reported by the real run is the
    ceiling rather than a measurement.
    """
    frame, targets = _frame()
    model = _fitted(frame, targets)

    assert model.best_iteration + 1 < model.n_estimators
    assert len(model.validation_rmse) == model.best_iteration + 1 + model.early_stopping_rounds


def test_predicting_with_fewer_trees_gives_a_different_answer():
    """The truncation the 500 tree comparison rests on. If it is a no-op, so is the row."""
    frame, targets = _frame()
    model = _fitted(frame, targets, early_stopping_rounds=40)

    full = model.predict(frame)
    truncated = model.predict(frame, trees=1)
    assert model.best_iteration + 1 > 1, "need more than one tree for this to mean anything"
    assert not np.allclose(full, truncated)


def test_a_boundary_that_leaves_one_side_empty_is_rejected():
    frame, targets = _frame()
    with pytest.raises(ValueError, match="both have to be non-empty"):
        _fitted(frame, targets, first_validation_month="2030-01")
