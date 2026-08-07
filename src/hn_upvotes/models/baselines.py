"""The baseline ladder.

Built before any neural network, so the fusion models have a real bar to clear. Six
rungs, each answering one question:

1. **Trailing mean.** The mean ``log1p(score)`` of recent posts before this one. The
   floor. Anything that fails to beat it has extracted nothing from the post.
2. **Author history.** That author's mean over their earlier posts. Does knowing who
   posted it help?
3. **Domain history.** The same, keyed on the hostname of the link. Does knowing where
   it points help?
4. **Body text alone.** The text the poster wrote under the headline, its length, and
   whether there is any, into a linear model. Is the body worth anything by itself?
5. **All signals plus Ridge.** Title, body, both history means, whether there is a link
   and the time of day, in one matrix, into a linear model. Does everything together
   help?
6. **All signals plus XGBoost.** The same matrix into gradient-boosted trees. Does a
   non-linear model find something the linear one cannot?

Rungs 1 to 4 are single-signal on purpose: each one isolates what one input is worth.
Rungs 5 and 6 see every submission-time signal, so they differ from each other only in
the model, not in the features.

Rungs 1 to 3 are numpy only. Rungs 4 and 5 need scikit-learn and rung 6 needs xgboost,
both from the ``train`` extra, and all import inside ``fit`` rather than at module level
so the first three rungs stay usable on a base install.

Every rung predicts ``log1p(score)`` directly. Phase 1 switched the trailing z-score off,
so that is the training target and there is no normalised space to convert out of.

**Rungs 1 to 3 read a history, and the history includes the test period.** That is not a
leak and it is not an accident. The bound in ``features.history.prior_mean`` is per row
and strict: a row's statistic sees only rows earlier than its own timestamp minus the 24
hour settling lag. A predictor running in March 2024 does know what January 2024 posts
scored, and pretending otherwise would understate rungs 2 and 3 rather than keep them
honest. Rungs 4 to 6 are fitted models, so their vectorisers, scalers and coefficients
see the training split and nothing else.
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol

import numpy as np
import pandas as pd

from hn_upvotes.data.splits import (
    DEFAULT_VALIDATION_SAMPLE_ROWS,
    DEFAULT_VALIDATION_START,
    sample_validation_tail,
    validation_tail_mask,
)
from hn_upvotes.features.body import BODY_FEATURE_NAMES, build_body_features, strip_html_column
from hn_upvotes.features.domain import extract_hostnames, has_url
from hn_upvotes.features.history import PriorConfig, prior_mean
from hn_upvotes.features.temporal import TEMPORAL_FEATURE_NAMES, build_temporal_features

logger = logging.getLogger(__name__)

#: Settling lag, matching ``target.normalise.BaselineConfig.lag``. Phase 1 measured
#: scores reaching their final level between 12 and 24 hours after submission.
SETTLING_LAG = timedelta(hours=24)

#: Window for rung 1, matching ``target.normalise.BaselineConfig.window``.
TRAILING_WINDOW = timedelta(days=30)

#: Rows needed in the 30 day window before its mean is trusted. Matches
#: ``BaselineConfig.min_periods``.
TRAILING_MIN_ROWS = 30


class Estimator(Protocol):
    """The interface every rung of the ladder and every fusion model presents."""

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> Estimator:
        """Fit on the training split. ``frame`` holds allowlisted columns only."""
        ...

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict ``log1p(score)``, one per row."""
        ...


@dataclass
class FallbackCount:
    """How often a rung could not answer from its own signal and fell back.

    Reported alongside the metrics because it bounds how much the rung could ever be
    worth. A rung that falls back on 60% of rows is 60% the rung below it.
    """

    rows: int = 0
    total: int = 0

    @property
    def fraction(self) -> float:
        return self.rows / self.total if self.total else 0.0


class TrailingMeanPredictor:
    """Rung 1. The mean ``log1p(score)`` of recent posts submitted before this one.

    Two levels, both knowable at prediction time. The 30 day trailing mean where there
    are at least ``TRAILING_MIN_ROWS`` posts in the window, and where there are not, the
    mean over the entire history before the row. The second level exists because the
    first test month looks back into the excluded window and finds nothing there.

    This is very nearly a constant, so it has almost no ranking. That is the point: it
    sets the error floor, and its Spearman and Precision@100 say what "no information"
    looks like on those two metrics.
    """

    def __init__(
        self,
        window: timedelta = TRAILING_WINDOW,
        lag: timedelta = SETTLING_LAG,
        min_rows: int = TRAILING_MIN_ROWS,
    ) -> None:
        self.recent = PriorConfig(window=window, lag=lag, min_rows=min_rows)
        self.all_history = PriorConfig(window=None, lag=lag, min_rows=1)
        self.rows_without_a_recent_window = FallbackCount()

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> TrailingMeanPredictor:
        """Record the history. There are no parameters to fit."""
        self._times = pd.Series(frame["time"]).reset_index(drop=True)
        self._targets = np.asarray(targets, dtype=np.float64)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        times = pd.Series(frame["time"]).reset_index(drop=True)
        # A single shared key, so "this key's earlier rows" means "all earlier rows".
        one_key = np.zeros(len(times), dtype=np.int8)
        history_key = np.zeros(len(self._times), dtype=np.int8)

        recent = prior_mean(
            times, one_key, self._times, history_key, self._targets, self.recent
        ).mean
        wide = prior_mean(
            times, one_key, self._times, history_key, self._targets, self.all_history
        ).mean

        thin = np.isnan(recent)
        self.rows_without_a_recent_window = FallbackCount(int(thin.sum()), len(times))
        # Nothing at all before the row can only happen at the very start of the data.
        return np.where(thin, np.nan_to_num(wide, nan=0.0), recent)


class HistoryMeanPredictor:
    """Rungs 2 and 3. The mean ``log1p(score)`` over earlier rows sharing a key.

    Expanding, not windowed: an author's whole back catalogue counts, because most
    authors post rarely enough that a 30 day window would be empty for almost all of
    them. Rows whose key has no history fall through to the rung below, which is rung 1.
    """

    #: Column the key is read from. Subclasses set this.
    key_column = ""

    def __init__(self, fallback: TrailingMeanPredictor, min_posts: int = 1) -> None:
        self.fallback = fallback
        self.config = PriorConfig(window=None, lag=SETTLING_LAG, min_rows=min_posts)
        self.rows_without_history = FallbackCount()

    def keys(self, frame: pd.DataFrame) -> pd.Series:
        """The grouping key for each row."""
        return pd.Series(frame[self.key_column]).reset_index(drop=True)

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> HistoryMeanPredictor:
        """Record the history, and fit the rung below to fall back to."""
        self._times = pd.Series(frame["time"]).reset_index(drop=True)
        self._keys = self.keys(frame)
        self._targets = np.asarray(targets, dtype=np.float64)
        self.fallback.fit(frame, targets)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        times = pd.Series(frame["time"]).reset_index(drop=True)
        prior = prior_mean(
            times, self.keys(frame), self._times, self._keys, self._targets, self.config
        )
        missing = np.isnan(prior.mean)
        self.rows_without_history = FallbackCount(int(missing.sum()), len(times))
        return np.where(missing, self.fallback.predict(frame), prior.mean)


class AuthorMeanPredictor(HistoryMeanPredictor):
    """Rung 2. Keyed on ``by``, which is Hacker News's own name for the account."""

    key_column = "by"


class DomainMeanPredictor(HistoryMeanPredictor):
    """Rung 3. Keyed on the hostname of the linked URL.

    Text posts have no URL and get their own bucket rather than falling back, because
    "no link" is a real and consistent category with its own scoring behaviour, not a
    missing value. They share the empty-string key, so their prior is the mean of
    earlier text posts.
    """

    key_column = "url"

    def keys(self, frame: pd.DataFrame) -> pd.Series:
        return extract_hostnames(frame[self.key_column])


class BodyTextFeatures:
    """Rung 4's matrix: the body text and nothing else.

    Three parts. TF-IDF over the stripped body, the length of that body, and a flag for
    whether there is one. The length and the flag are in because the measured effect is
    largely a length effect and 88.4% of rows have no body at all, so an all-zero row has
    to be distinguishable from a genuinely short one.

    The body is HTML as the API returns it. ``features.body.strip_html`` removes the tags
    and unescapes the entities first, so the vocabulary is words and the length is a
    length of prose.
    """

    def __init__(self, max_features: int = 50_000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.body = TfidfBlock(max_features, ngram_range)

    def fit_transform(self, frame: pd.DataFrame):  # noqa: ANN201 - scipy sparse
        plain = strip_html_column(frame["text"])
        self.scaler = _new_scaler()
        return stack(
            [self.body.fit_transform(plain)],
            self.scaler.fit_transform(build_body_features(plain).to_numpy(dtype=np.float64)),
        )

    def transform(self, frame: pd.DataFrame):  # noqa: ANN201 - scipy sparse
        plain = strip_html_column(frame["text"])
        return stack(
            [self.body.transform(plain)],
            self.scaler.transform(build_body_features(plain).to_numpy(dtype=np.float64)),
        )


class AllSignalFeatures:
    """Rungs 5 and 6: every signal a post carries at submission time, as one matrix.

    Two sparse text blocks and a dense block:

    1. TF-IDF over the title. Word and word-pair counts, each down-weighted by how many
       titles the term appears in, so "the" counts for almost nothing.
    2. TF-IDF over the body text, vectorised **separately** from the title. A headline and
       a paragraph of explanation are different registers and one shared bag of words
       would lose that.
    3. Ten dense columns: the author's trailing mean, the hostname's trailing mean, the
       body length, whether there is a body, whether there is a link, and the five time
       features from ``features.temporal``.

    The two trailing means are the *fitted rung 2 and rung 3 predictors*, called per row,
    not a mean recomputed over the whole training set. That distinction is the whole game:
    a global author mean applied to every one of that author's rows is leakage, and it is
    the easy mistake here.

    "Has a link" is kept apart from the hostname's track record on purpose. A text post's
    hostname is empty, and empty has to read as information rather than as a gap.

    The dense columns are standardised on the training rows before being stacked, because
    Ridge applies one penalty to every coefficient and an unscaled column running 0 to 9
    next to TF-IDF values running 0 to 1 would be penalised far less than the text. Trees
    do not care either way, and both rungs take the same matrix so the comparison between
    them stays like for like.
    """

    def __init__(
        self,
        author: AuthorMeanPredictor,
        domain: DomainMeanPredictor,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        self.author = author
        self.domain = domain
        self.title = TfidfBlock(max_features, ngram_range)
        self.body = TfidfBlock(max_features, ngram_range)

    @property
    def dense_column_names(self) -> tuple[str, ...]:
        return (
            "author_prior_mean",
            "domain_prior_mean",
            *BODY_FEATURE_NAMES,
            "has_url",
            *TEMPORAL_FEATURE_NAMES,
        )

    def fit_transform(self, frame: pd.DataFrame):  # noqa: ANN201 - scipy sparse
        plain = strip_html_column(frame["text"])
        self.scaler = _new_scaler()
        return stack(
            [self.title.fit_transform(titles(frame)), self.body.fit_transform(plain)],
            self.scaler.fit_transform(self._dense_block(frame, plain)),
        )

    def transform(self, frame: pd.DataFrame):  # noqa: ANN201 - scipy sparse
        plain = strip_html_column(frame["text"])
        return stack(
            [self.title.transform(titles(frame)), self.body.transform(plain)],
            self.scaler.transform(self._dense_block(frame, plain)),
        )

    def _dense_block(self, frame: pd.DataFrame, plain: pd.Series) -> np.ndarray:
        return np.column_stack(
            [
                self.author.predict(frame),
                self.domain.predict(frame),
                build_body_features(plain).to_numpy(dtype=np.float64),
                has_url(frame["url"]),
                build_temporal_features(frame["time"]).to_numpy(dtype=np.float64),
            ]
        )


class TfidfBlock:
    """One TF-IDF vectoriser over one text column. Fitted on the training split only."""

    def __init__(self, max_features: int, ngram_range: tuple[int, int]) -> None:
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit_transform(self, texts: pd.Series):  # noqa: ANN201 - scipy sparse
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vectoriser = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=True,
            dtype=np.float32,
        )
        return self.vectoriser.fit_transform(texts)

    def transform(self, texts: pd.Series):  # noqa: ANN201 - scipy sparse
        return self.vectoriser.transform(texts)


def stack(sparse_blocks: list, dense: np.ndarray):  # noqa: ANN001, ANN201 - scipy sparse
    """Put the sparse text blocks and the dense columns side by side, as one CSR matrix."""
    from scipy.sparse import csr_matrix, hstack

    return hstack([*sparse_blocks, csr_matrix(dense.astype(np.float32))], format="csr")


def _new_scaler():  # noqa: ANN202 - sklearn
    from sklearn.preprocessing import StandardScaler

    return StandardScaler()


class RidgeOnFeatures:
    """Ridge regression over whichever feature matrix it is handed.

    Nothing here is tuned. ``alpha`` is scikit-learn's default of 1.0 and the vocabulary
    size is the round number the scaffold started with. A baseline that has been tuned is
    not a baseline.
    """

    def __init__(self, features) -> None:  # noqa: ANN001 - structural
        self.features = features
        self.alpha = 1.0

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> RidgeOnFeatures:
        """Fit the vectorisers, the scaler and the regressor on the training split only."""
        from sklearn.linear_model import Ridge

        matrix = self.features.fit_transform(frame)
        # lsqr is the sparse iterative solver. The default solver would try to form a
        # dense 100,000 by 100,000 matrix.
        self.model = Ridge(alpha=self.alpha, solver="lsqr")
        self.model.fit(matrix, np.asarray(targets, dtype=np.float64))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(self.features.transform(frame)))


class BodyTextRidge(RidgeOnFeatures):
    """Rung 4. Ridge over the body text alone.

    The rung that answers whether the body is worth anything on its own. It sees no
    title, no author, no hostname and no clock.
    """

    def __init__(self, max_features: int = 50_000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        super().__init__(BodyTextFeatures(max_features, ngram_range))


class AllSignalsRidge(RidgeOnFeatures):
    """Rung 5. Ridge over every submission-time signal.

    The bar the from-scratch embeddings have to clear. A linear model over word counts is
    a strong baseline on short text, and saying so up front is more useful than
    discovering it after training three fusion architectures.
    """

    def __init__(
        self,
        author: AuthorMeanPredictor,
        domain: DomainMeanPredictor,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        super().__init__(AllSignalFeatures(author, domain, max_features, ngram_range))


class AllSignalsXGBoost(AllSignalsRidge):
    """Rung 6. The same matrix into gradient-boosted trees.

    XGBoost is a separate package with a scikit-learn compatible estimator, not part of
    scikit-learn. It takes the sparse matrix directly: no densifying, which at 3.5 million
    rows by 100,010 columns would be about 1.4 TB, and no dimensionality reduction, which
    would change what the rung is measuring.

    Rungs 5 and 6 see exactly the same features, so the gap between them is linear against
    non-linear and nothing else.

    Not tuned, for the same reason rung 5 is not. ``max_bin`` is the one setting moved off
    its default, from 256 down to 64, and that is a wall-clock decision rather than an
    accuracy one: histogram building dominates the fit at this width, and TF-IDF columns
    hold only a handful of distinct values each, so the finer bins had nothing to resolve.
    """

    def __init__(
        self,
        author: AuthorMeanPredictor,
        domain: DomainMeanPredictor,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        max_bin: int = 64,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(author, domain, max_features, ngram_range)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_bin = max_bin
        self.n_jobs = n_jobs

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> AllSignalsXGBoost:
        """Fit on the training split only."""
        from xgboost import XGBRegressor

        matrix = self.features.fit_transform(frame)
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_bin=self.max_bin,
            tree_method="hist",
            n_jobs=self.n_jobs,
            random_state=0,
        )
        self.model.fit(matrix, np.asarray(targets, dtype=np.float32))
        return self


class EarlyStoppedXGBoost(AllSignalsXGBoost):
    """Rung 6, stopped by a held-out number instead of by a fixed tree count.

    Same features, same depth, same learning rate, same ``max_bin`` as
    :class:`AllSignalsXGBoost`. One thing changes: ``n_estimators`` goes from 500 to a
    ceiling of 5,000 and training ends when the validation RMSE has not improved for
    ``early_stopping_rounds`` rounds. The ceiling is high enough that early stopping is
    what ends the run; if the run reaches it, the number measured is the ceiling and the
    result has to say so.

    The question it answers: 500 trees at depth 6 is at most 31,500 splits, against
    100,010 columns, so the untuned rung could not have looked at most of its own feature
    matrix. Either the run stops early, and 500 trees was already more than the model
    could use, or it runs long and keeps improving, and 500 was starving it.

    **The validation slice is the last months of the training period, never a random
    sample of the training period.** ``data.splits.validation_tail_mask`` is where that is
    enforced. A slice drawn at random across the whole period would let the tree count be
    chosen on posts from the same weeks the model trained on. The test period is not
    touched: it is scored once, at the end.

    What *is* sampled is which of those held-back rows get scored each round, bounded by
    ``validation_sample_rows``. Every row after the boundary stays out of the fit set
    either way, so the sampled rows are still strictly later than everything learned from,
    and the fit set is identical whether sampling is on or off.

    The vectorisers and the scaler are fitted on the earlier part only, so the validation
    slice is out of sample all the way down to the vocabulary, not just for the trees.

    **This rung calls ``xgboost.train`` directly instead of ``XGBRegressor``, and that is
    a speed decision with a measurement behind it.** The scikit-learn wrapper builds an
    eval set as a ``QuantileDMatrix``, which carries no incremental prediction cache, so
    every round re-scores the whole validation slice from scratch. Measured on the real
    matrix, 10 rounds each: 95.13 seconds a round through the wrapper against 4.52 through
    the native API with a plain ``DMatrix`` eval, on the identical 296,531 row slice. A
    5,000 tree run is six days one way and six hours the other. The rest of the ladder
    keeps the wrapper, because nothing else on it passes an eval set.
    """

    #: Boosting parameters, in the native API's names. ``learning_rate``, ``max_depth`` and
    #: ``max_bin`` come from the untuned rung unchanged, so the tree count is the only
    #: difference between the two.
    def _params(self, base_score: float) -> dict:
        return {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "max_bin": self.max_bin,
            "tree_method": "hist",
            "nthread": self.n_jobs,
            "seed": 0,
            # Set rather than left to the default, because the two APIs pick a starting
            # value differently and the untuned rung got the wrapper's, which is the mean.
            "base_score": base_score,
        }

    def __init__(
        self,
        author: AuthorMeanPredictor,
        domain: DomainMeanPredictor,
        first_validation_month: str = DEFAULT_VALIDATION_START,
        max_features: int = 50_000,
        ngram_range: tuple[int, int] = (1, 2),
        n_estimators: int = 5_000,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        max_bin: int = 64,
        n_jobs: int = -1,
        early_stopping_rounds: int = 50,
        verbose_every: int = 25,
        validation_sample_rows: int | None = DEFAULT_VALIDATION_SAMPLE_ROWS,
        validation_sample_seed: int = 0,
    ) -> None:
        super().__init__(
            author, domain, max_features, ngram_range, n_estimators, max_depth,
            learning_rate, max_bin, n_jobs,
        )  # fmt: skip
        self.first_validation_month = first_validation_month
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_every = verbose_every
        self.validation_sample_rows = validation_sample_rows
        self.validation_sample_seed = validation_sample_seed
        self.fit_rows = 0
        self.held_back_rows = 0
        self.validation_rows = 0
        self.best_iteration = 0
        self.boosting_seconds = 0.0
        self.validation_rmse: list[float] = []

    def fit(self, frame: pd.DataFrame, targets: np.ndarray) -> EarlyStoppedXGBoost:
        """Cut the validation tail off the end, fit on what is left, stop on the tail."""
        import xgboost as xgb

        every = np.asarray(targets, dtype=np.float32)
        in_tail = validation_tail_mask(frame["time"], self.first_validation_month)
        evaluated = sample_validation_tail(
            in_tail, self.validation_sample_rows, self.validation_sample_seed
        )
        self.fit_rows = int((~in_tail).sum())
        self.held_back_rows = int(in_tail.sum())
        self.validation_rows = int(evaluated.sum())
        if not self.fit_rows or not self.validation_rows:
            raise ValueError(
                f"{self.first_validation_month} leaves {self.fit_rows:,} rows to fit on and "
                f"{self.validation_rows:,} to validate on; both have to be non-empty"
            )
        logger.info(
            "fitting on %s rows, holding back %s from %s and scoring %s of them each round",
            f"{self.fit_rows:,}",
            f"{self.held_back_rows:,}",
            self.first_validation_month,
            f"{self.validation_rows:,}",
        )

        matrix = self.features.fit_transform(frame.loc[~in_tail])
        validation_matrix = self.features.transform(frame.loc[evaluated])
        fit_target, validation_target = every[~in_tail], every[evaluated]
        # The frames are several GB with body text on 3.3 million rows, and the matrices
        # are built. Holding both while boosting pushed the first attempt into swap and
        # cost about 100 seconds a round; see docs/design.md.
        del frame, targets, every
        gc.collect()

        # QuantileDMatrix for training, because it bins once and holds the bins rather
        # than the values. Plain DMatrix for the eval, because that is the one XGBoost
        # keeps a prediction cache for, and the cache is the difference between 4.5
        # seconds a round and 95.
        train_data = xgb.QuantileDMatrix(matrix, label=fit_target, max_bin=self.max_bin)
        validation_data = xgb.DMatrix(validation_matrix, label=validation_target)
        del matrix, validation_matrix
        gc.collect()

        history: dict = {}
        started = time.perf_counter()
        self.model = xgb.train(
            self._params(float(fit_target.mean())),
            train_data,
            num_boost_round=self.n_estimators,
            evals=[(validation_data, "validation")],
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=history,
            verbose_eval=self.verbose_every,
            callbacks=[_round_timer(self.verbose_every)],
        )
        self.boosting_seconds = time.perf_counter() - started
        self.best_iteration = int(self.model.best_iteration)
        self.validation_rmse = [float(v) for v in history["validation"]["rmse"]]
        logger.info(
            "%s rounds in %.0fs, %.2fs a round",
            self.trees_built,
            self.boosting_seconds,
            self.seconds_per_round,
        )
        return self

    @property
    def seconds_per_round(self) -> float:
        """Wall clock per boosting round. The number that decides whether a run is viable."""
        return self.boosting_seconds / self.trees_built if self.trees_built else 0.0

    def predict(self, frame: pd.DataFrame, trees: int | None = None) -> np.ndarray:
        """Predict with the first ``trees`` trees, or with the best round if not given.

        XGBoost already truncates at the best round when early stopping fired, but this
        says so out loud, and the ``trees`` argument is how the same fitted model gets
        scored at 500 trees for the comparison against the untuned rung.
        """
        import xgboost as xgb

        end = self.best_iteration + 1 if trees is None else trees
        data = xgb.DMatrix(self.features.transform(frame))
        return np.asarray(self.model.predict(data, iteration_range=(0, end)))

    @property
    def trees_built(self) -> int:
        """Rounds actually run, including the ones after the best that failed to improve."""
        return len(self.validation_rmse)

    def predictions_by_tree_count(
        self, frame: pd.DataFrame, counts: list[int]
    ) -> dict[int, np.ndarray]:
        """One set of predictions per tree count, off a single fitted model.

        Early stopping watches RMSE, because that is what the objective minimises, and
        this project ranks on Spearman. Those can come apart: error can flatten while
        ordering is still improving. Scoring the same model at a ladder of tree counts is
        how that shows up, and it costs one feature transform rather than one refit per
        count.

        Counts above what the run actually built are dropped rather than clamped, so a
        row in the output always means the trees existed.
        """
        import xgboost as xgb

        data = xgb.DMatrix(self.features.transform(frame))
        return {
            count: np.asarray(self.model.predict(data, iteration_range=(0, count)))
            for count in counts
            if 0 < count <= self.trees_built
        }


def _round_timer(every: int):  # noqa: ANN202 - xgboost callback, imported lazily
    """Log seconds per boosting round every ``every`` rounds.

    A run at 3 seconds a round finishes 5,000 trees in four hours and a run at 100 seconds
    a round takes six days. The difference does not show up in any output XGBoost prints,
    so it is logged here and the log is what says whether to let a run continue.

    The class is defined inside the function because ``xgboost`` is imported inside ``fit``
    and this module has to stay importable without the ``train`` extra.
    """
    from xgboost.callback import TrainingCallback

    class RoundTimer(TrainingCallback):
        def before_training(self, model):  # noqa: ANN001, ANN202
            self.started = self.marked = time.perf_counter()
            self.marked_round = 0
            return model

        def after_iteration(self, model, epoch: int, evals_log) -> bool:  # noqa: ANN001
            done = epoch + 1
            if every and done % every == 0:
                now = time.perf_counter()
                since = done - self.marked_round
                logger.info(
                    "round %s, %.2fs a round over the last %s, %.0fs so far",
                    done,
                    (now - self.marked) / since,
                    since,
                    now - self.started,
                )
                self.marked, self.marked_round = now, done
            return False

    return RoundTimer()


def titles(frame: pd.DataFrame) -> pd.Series:
    """Title text, with the empty-string sentinel the dump uses for a missing title."""
    return pd.Series(frame["title"]).reset_index(drop=True).fillna("").astype(str)
