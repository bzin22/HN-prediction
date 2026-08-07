"""Was rung 6 underfitting? One early-stopped XGBoost run, and the numbers to decide it.

``make tune-xgboost``. Rung 6 lost to rung 5 on every metric while taking eight times as
long, 1,518 seconds against 172, on an identical feature matrix. The hypothesis was that
500 trees at depth 6 is at most 31,500 splits against 100,010 columns, so the model never
got to look at most of its own features.

Early stopping settles it in one run. Training stops when the validation RMSE has not
improved for 50 rounds, with a ceiling of 5,000 trees so the ceiling is not what ends it.
Stop early and 500 trees was already more than the model could use. Run long and keep
improving and 500 was starving it.

Four fits, so the answer is not confounded by the validation slice costing training rows:

1. Ridge on the full training split. Rung 5 as reported, re-measured on this machine.
2. Ridge on the training split minus the validation tail. The same rung on the same rows
   XGBoost gets, so any gap between the two is what the held-back year costs.
3. XGBoost with early stopping, scored at its best round.
4. The same fitted XGBoost truncated to 500 trees. Same rows, same fit, one difference:
   the tree count. This is the clean answer to the question.

Plus a ladder: the same fitted model scored on the test split at 25, 50, 100, 250, 500
trees and up. Early stopping watches RMSE, which is what the objective minimises, and this
project ranks on Spearman. The ladder is what shows whether ordering was still improving
after error had stopped.

The test period, 2024-01 to 2025-12, is not touched by any of it. It is scored once.

Needs the ``data`` extra for DuckDB and the ``train`` extra for scikit-learn and xgboost.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hn_upvotes.data.splits import (
    DEFAULT_SPLIT,
    DEFAULT_VALIDATION_START,
    SplitConfig,
    split_stories,
    validation_tail_mask,
)
from hn_upvotes.models.baselines import (
    AllSignalsRidge,
    AuthorMeanPredictor,
    DomainMeanPredictor,
    EarlyStoppedXGBoost,
    TrailingMeanPredictor,
)
from hn_upvotes.training.metrics import MetricReport, report
from hn_upvotes.training.run_baselines import load_stories

logger = logging.getLogger(__name__)

RESULTS = Path("artifacts/xgboost-early-stopping.json")

#: The untuned rung 6, from the Phase 2 run on 2026-08-06. Carried here so the comparison
#: is in the output file rather than only in the README.
UNTUNED_RUNG_6 = {
    "trees": 500,
    "rmse_log1p_score": 1.152,
    "mae_log1p_score": 0.814,
    "spearman_raw_score": 0.265,
    "precision_at_100": 0.01,
    "fit_seconds": 1518.0,
}

#: Tree count of the untuned rung, scored again off the early-stopped model.
UNTUNED_TREES = 500

#: Tree counts the test split is scored at, off the one fitted model. Early stopping
#: watches validation RMSE because that is what the objective minimises, and this project
#: ranks on Spearman. If ordering keeps improving after error stops, the stopping metric
#: cost something and this ladder is where that becomes visible. Counts above what the run
#: built are dropped.
TREE_COUNT_LADDER = (25, 50, 100, 250, 500, 1_000, 1_500, 2_000, 3_000, 4_000, 5_000)


@dataclass
class FitResult:
    """One fitted model's test metrics, plus how long it took and what it saw."""

    name: str
    metrics: MetricReport
    fit_seconds: float
    training_rows: int
    trees: int | None = None
    note: str = ""
    #: True when this row reuses another row's fit rather than costing its own time.
    reuses_a_fit: bool = False
    validation_rmse: list[float] = field(default_factory=list)
    test_metrics_by_tree_count: dict[int, MetricReport] = field(default_factory=dict)
    seconds_per_round: float = 0.0
    rows_held_back: int = 0
    rows_scored_each_round: int = 0

    def as_dict(self) -> dict:
        out = {
            "model": self.name,
            "training_rows": self.training_rows,
            "trees": self.trees,
            "fit_seconds": round(self.fit_seconds, 1),
            "reuses_a_fit": self.reuses_a_fit,
            "note": self.note,
            **self.metrics.as_dict(),
        }
        if self.seconds_per_round:
            out["seconds_per_round"] = round(self.seconds_per_round, 3)
            out["rows_held_back"] = self.rows_held_back
            out["rows_scored_each_round"] = self.rows_scored_each_round
        if self.validation_rmse:
            out["validation_rmse_per_round"] = [round(v, 6) for v in self.validation_rmse]
        if self.test_metrics_by_tree_count:
            out["test_metrics_by_tree_count"] = {
                str(trees): metrics.as_dict()
                for trees, metrics in sorted(self.test_metrics_by_tree_count.items())
            }
        return out


def run(
    frame: pd.DataFrame,
    config: SplitConfig = DEFAULT_SPLIT,
    first_validation_month: str = DEFAULT_VALIDATION_START,
) -> list[FitResult]:
    """Fit the four models and score every one of them on the untouched test split."""
    parts = split_stories(frame, config)
    train, test = parts["train"], parts["test"]
    # The source table is 4.7 million rows carrying body text, and the split has already
    # copied out everything still needed. Holding it alongside the feature matrices is
    # what put the first attempt into swap.
    del frame, parts
    gc.collect()
    held_back = validation_tail_mask(train["time"], first_validation_month)
    logger.info(
        "train %s rows, of which %s held back from %s as validation, leaving %s to fit on. "
        "test %s rows, untouched",
        f"{len(train):,}",
        f"{int(held_back.sum()):,}",
        first_validation_month,
        f"{int((~held_back).sum()):,}",
        f"{len(test):,}",
    )

    train_target = np.log1p(train["score"].to_numpy(dtype=np.float64))
    test_target = np.log1p(test["score"].to_numpy(dtype=np.float64))

    # Rungs 2 and 3 as feature builders, exactly as run_baselines wires them: fitted on
    # the whole settled record, with the per-row strictly-earlier bound inside
    # features.history.prior_mean doing the work.
    history = pd.concat([train, test], ignore_index=True)
    history_target = np.concatenate([train_target, test_target])
    author = AuthorMeanPredictor(fallback=TrailingMeanPredictor(), min_posts=1)
    author.fit(history, history_target)
    domain = DomainMeanPredictor(fallback=TrailingMeanPredictor(), min_posts=1)
    domain.fit(history, history_target)
    # Both predictors kept their own copy of the times, keys and targets they need, which
    # is three columns rather than seven, so the concatenated frame can go.
    del history, history_target
    gc.collect()

    results = []

    logger.info("fitting Ridge on the full training split")
    started = time.perf_counter()
    ridge = AllSignalsRidge(author, domain).fit(train, train_target)
    full_ridge_seconds = time.perf_counter() - started
    results.append(
        FitResult(
            "5. All signals + Ridge",
            report(ridge.predict(test), test_target),
            full_ridge_seconds,
            len(train),
            note="Rung 5 as Phase 2 reported it, re-measured here.",
        )
    )
    _log(results[-1])
    # Its vectorisers hold a 100,010 term vocabulary and the fitted matrix is no longer
    # needed, so it goes before the next fit rather than after the run.
    del ridge
    gc.collect()

    logger.info("fitting Ridge on the training split minus the validation tail")
    started = time.perf_counter()
    short_ridge = AllSignalsRidge(author, domain)
    short_ridge.fit(train.loc[~held_back], train_target[~held_back])
    short_ridge_seconds = time.perf_counter() - started
    results.append(
        FitResult(
            "5b. Ridge, validation tail held back",
            report(short_ridge.predict(test), test_target),
            short_ridge_seconds,
            int((~held_back).sum()),
            note=(
                f"The same rung on the same rows XGBoost gets. The gap to rung 5 is what "
                f"holding back {first_validation_month} onward costs, on its own."
            ),
        )
    )
    _log(results[-1])
    del short_ridge
    gc.collect()

    logger.info("fitting XGBoost with early stopping. This is the long one")
    model = EarlyStoppedXGBoost(author, domain, first_validation_month=first_validation_month)
    started = time.perf_counter()
    model.fit(train, train_target)
    xgboost_seconds = time.perf_counter() - started
    stopped_early = model.best_iteration + 1 < model.n_estimators
    logger.info(
        "best round %s of a %s ceiling, %s",
        model.best_iteration + 1,
        model.n_estimators,
        "stopped early" if stopped_early else "RAN TO THE CEILING",
    )

    # One fitted model, scored on the test split at a ladder of tree counts. No refits, so
    # the rows, the vocabulary and the seed are identical down the ladder and the only
    # thing that changes is how many trees vote.
    best_trees = model.best_iteration + 1
    counts = sorted({*TREE_COUNT_LADDER, UNTUNED_TREES, best_trees})
    logger.info("scoring the test split at %s tree counts", len(counts))
    curve = {
        trees: report(predictions, test_target)
        for trees, predictions in model.predictions_by_tree_count(test, counts).items()
    }

    results.append(
        FitResult(
            "6. All signals + XGBoost, early stopped",
            curve[best_trees],
            xgboost_seconds,
            model.fit_rows,
            trees=best_trees,
            note=(
                (
                    f"Stopped early at round {best_trees} of {model.n_estimators}."
                    if stopped_early
                    else f"Ran to the {model.n_estimators} tree ceiling. The ceiling ended "
                    "it, not early stopping, so this is a floor on the useful tree count."
                )
                + f" {model.held_back_rows:,} rows held back from {first_validation_month},"
                f" {model.validation_rows:,} of them scored each round,"
                f" at {model.seconds_per_round:.2f}s a round."
            ),
            validation_rmse=model.validation_rmse,
            test_metrics_by_tree_count=curve,
            seconds_per_round=model.seconds_per_round,
            rows_held_back=model.held_back_rows,
            rows_scored_each_round=model.validation_rows,
        )
    )
    _log(results[-1])

    if UNTUNED_TREES in curve and best_trees != UNTUNED_TREES:
        results.append(
            FitResult(
                f"6b. The same model, first {UNTUNED_TREES} trees",
                curve[UNTUNED_TREES],
                xgboost_seconds,
                model.fit_rows,
                trees=UNTUNED_TREES,
                note="Truncated at prediction time, not refitted, so only the tree count differs.",
                reuses_a_fit=True,
            )
        )
        _log(results[-1])
    else:
        logger.info(
            "the run built %s trees, so there is no %s tree truncation to score against",
            model.trees_built,
            UNTUNED_TREES,
        )
    return results


def _log(result: FitResult) -> None:
    m = result.metrics
    logger.info(
        "%s: RMSE %.3f, MAE %.3f, Spearman %.3f, P@100 %.0f, %.0fs",
        result.name,
        m.rmse_log1p_score,
        m.mae_log1p_score,
        m.spearman_raw_score,
        m.precision_at_100 * 100,
        result.fit_seconds,
    )


def markdown_table(results: list[FitResult]) -> str:
    """The table that goes in the README."""
    lines = [
        "| Model | Trees | RMSE | MAE | Spearman | Fit |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        m = r.metrics
        trees = f"{r.trees:,}" if r.trees else "n/a"
        fit = "same fit" if r.reuses_a_fit else f"{r.fit_seconds:.0f}s"
        lines.append(
            f"| {r.name} | {trees} | {m.rmse_log1p_score:.3f} | {m.mae_log1p_score:.3f} "
            f"| {m.spearman_raw_score:.3f} | {fit} |"
        )
    return "\n".join(lines)


def tree_count_table(curve: dict[int, MetricReport], best_trees: int) -> str:
    """Test metrics against tree count, off the one fitted model.

    Read down the Spearman column. If it is still climbing at the round early stopping
    picked, RMSE was the wrong thing to stop on and a search is worth asking for.
    """
    lines = [
        "| Trees | RMSE | MAE | Spearman | P@100 |",
        "|---|---|---|---|---|",
    ]
    for trees, m in sorted(curve.items()):
        mark = " (stopped here)" if trees == best_trees else ""
        lines.append(
            f"| {trees:,}{mark} | {m.rmse_log1p_score:.3f} | {m.mae_log1p_score:.3f} "
            f"| {m.spearman_raw_score:.3f} | {m.precision_at_100 * 100:.0f} |"
        )
    return "\n".join(lines)


def validation_curve_summary(curve: list[float], trees: int) -> str:
    """What the validation RMSE did between the untuned tree count and the best round.

    The within-run answer to the underfitting question, with no confound at all: one fit,
    one curve, two points on it.
    """
    if len(curve) < UNTUNED_TREES:
        return f"the run ended at round {len(curve)}, before round {UNTUNED_TREES}"
    at_untuned, best = curve[UNTUNED_TREES - 1], curve[trees - 1]
    change = (best - at_untuned) / at_untuned
    return (
        f"validation RMSE was {at_untuned:.4f} at round {UNTUNED_TREES} and {best:.4f} at "
        f"round {trees}, a change of {change:+.2%}"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    results = run(load_stories())

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            {
                "untuned_rung_6": UNTUNED_RUNG_6,
                "validation_start": DEFAULT_VALIDATION_START,
                "results": [r.as_dict() for r in results],
            },
            indent=2,
        )
        + "\n"
    )

    print()
    print(markdown_table(results))
    print()
    for r in results:
        if r.note:
            print(f"{r.name}: {r.note}")
        if r.validation_rmse and r.trees:
            print(validation_curve_summary(r.validation_rmse, r.trees))
        if r.test_metrics_by_tree_count and r.trees:
            print()
            print("Test metrics against tree count, one fitted model, no refits:")
            print(tree_count_table(r.test_metrics_by_tree_count, r.trees))
    logger.info("wrote %s", RESULTS)


if __name__ == "__main__":
    main()
