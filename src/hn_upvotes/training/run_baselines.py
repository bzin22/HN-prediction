"""Run the baseline ladder on the real split and print the results table.

``make baselines``. Reads ``data/stories.parquet``, cuts the Phase 2 split, fits the six
rungs, and writes both a Markdown table for the README and a JSON file for provenance.

Needs the ``data`` extra for DuckDB and the ``train`` extra for rungs 4 to 6.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from hn_upvotes.data.splits import DEFAULT_SPLIT, SplitConfig, split_stories
from hn_upvotes.features.body import strip_html_column
from hn_upvotes.features.domain import NO_URL_KEY, extract_hostnames
from hn_upvotes.models.baselines import (
    AllSignalsRidge,
    AllSignalsXGBoost,
    AuthorMeanPredictor,
    BodyTextRidge,
    DomainMeanPredictor,
    TrailingMeanPredictor,
)
from hn_upvotes.training.metrics import MetricReport, report

logger = logging.getLogger(__name__)

STORIES = Path("data/stories.parquet")
RESULTS = Path("artifacts/baselines.json")

#: Columns the ladder reads. ``score`` is here to build the target, not as a feature;
#: ``features/schema.py`` rejects it as one. ``text`` is Hacker News's own name for the
#: body written under the headline, and it is on the allowlist.
COLUMNS = ("id", "time", "title", "text", "score", "by", "url")


@dataclass(frozen=True)
class RungResult:
    """One rung's numbers, plus the note that goes under the table."""

    name: str
    metrics: MetricReport
    fit_seconds: float
    note: str = ""


def load_stories(path: Path = STORIES) -> pd.DataFrame:
    """Read the stories table. ``by`` is a reserved word in DuckDB and must be quoted.

    ``time`` in this table is already a naive UTC ``TIMESTAMP``: ingest applied the
    ``AT TIME ZONE 'UTC'`` cast on the way in. Applying it a second time here would
    reinterpret it in the session timezone and shift every month boundary.
    """
    import duckdb

    con = duckdb.connect()
    try:
        projection = ", ".join(f'"{c}"' for c in COLUMNS)
        return con.execute(f"SELECT {projection} FROM read_parquet('{path}')").df()
    finally:
        con.close()


def run(frame: pd.DataFrame, config: SplitConfig = DEFAULT_SPLIT) -> list[RungResult]:
    """Fit every rung and score it on the test split."""
    parts = split_stories(frame, config)
    train, test = parts["train"], parts["test"]
    logger.info("train %s rows, test %s rows", f"{len(train):,}", f"{len(test):,}")

    train_target = np.log1p(train["score"].to_numpy(dtype=np.float64))
    test_target = np.log1p(test["score"].to_numpy(dtype=np.float64))

    # Rungs 1 to 3 look up a history rather than fitting parameters, so they are given
    # the whole settled record, train and test. The per-row bound in
    # features.history.prior_mean is what keeps that honest: a row reads only rows
    # earlier than its own timestamp minus the 24 hour settling lag.
    history = pd.concat([train, test], ignore_index=True)
    history_target = np.concatenate([train_target, test_target])

    results = []

    trailing = TrailingMeanPredictor()
    results.append(_score("1. Trailing mean", trailing, history, history_target, test, test_target))
    thin = trailing.rows_without_a_recent_window
    results[-1] = _with_note(
        results[-1],
        f"{thin.rows:,} test rows ({thin.fraction:.2%}) had fewer than 30 posts in their "
        "30 day window and used the mean over all earlier history instead.",
    )

    author = AuthorMeanPredictor(fallback=TrailingMeanPredictor(), min_posts=1)
    results.append(_score("2. Author history", author, history, history_target, test, test_target))
    missing = author.rows_without_history
    results[-1] = _with_note(
        results[-1],
        f"{missing.rows:,} test rows ({missing.fraction:.2%}) had an author with no earlier "
        "post and fell back to rung 1.",
    )

    domain = DomainMeanPredictor(fallback=TrailingMeanPredictor(), min_posts=1)
    results.append(_score("3. Domain history", domain, history, history_target, test, test_target))
    missing = domain.rows_without_history
    text_posts = int((extract_hostnames(test["url"]) == NO_URL_KEY).sum())
    results[-1] = _with_note(
        results[-1],
        f"{missing.rows:,} test rows ({missing.fraction:.2%}) had a hostname with no earlier "
        f"post and fell back to rung 1. {text_posts:,} test rows ({text_posts / len(test):.2%}) "
        "are text posts with no link; they share one bucket rather than falling back.",
    )

    results.append(
        _score("4. Body text + Ridge", BodyTextRidge(), train, train_target, test, test_target)
    )
    with_body = int((strip_html_column(test["text"]).str.len() > 0).sum())
    results[-1] = _with_note(
        results[-1],
        f"{with_body:,} test rows ({with_body / len(test):.1%}) carry any body text at all. "
        "The other rows differ only in their length and has-body columns, both zero, so "
        "this rung predicts one number for all of them.",
    )

    # Rungs 5 and 6 reuse the fitted rungs 2 and 3 as feature builders, so the author and
    # domain columns carry the same strictly-earlier bound rather than a mean recomputed
    # over the whole training set. The Ridge and XGBoost fits themselves see the training
    # split only.
    results.append(
        _score(
            "5. All signals + Ridge",
            AllSignalsRidge(author, domain),
            train,
            train_target,
            test,
            test_target,
        )
    )
    results.append(
        _score(
            "6. All signals + XGBoost",
            AllSignalsXGBoost(author, domain),
            train,
            train_target,
            test,
            test_target,
        )
    )
    return results


def _score(
    name: str,
    model,  # noqa: ANN001 - Estimator protocol, structural
    fit_frame: pd.DataFrame,
    fit_target: np.ndarray,
    test: pd.DataFrame,
    test_target: np.ndarray,
) -> RungResult:
    started = time.perf_counter()
    model.fit(fit_frame, fit_target)
    predictions = model.predict(test)
    elapsed = time.perf_counter() - started
    logger.info("%s fitted and scored in %.1fs", name, elapsed)
    return RungResult(name, report(predictions, test_target), elapsed)


def _with_note(result: RungResult, note: str) -> RungResult:
    return RungResult(result.name, result.metrics, result.fit_seconds, note)


def markdown_table(results: list[RungResult]) -> str:
    """The table that goes in the README."""
    lines = [
        "| Rung | RMSE (`log1p` score) | MAE (`log1p` score) | Spearman | P@100 |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        m = r.metrics
        lines.append(
            f"| {r.name} | {m.rmse_log1p_score:.4f} | {m.mae_log1p_score:.4f} "
            f"| {m.spearman_raw_score:.4f} | {m.precision_at_100:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not STORIES.exists():
        raise SystemExit(f"{STORIES} is missing. Phase 1's `make ingest` builds it.")

    results = run(load_stories())

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(
        json.dumps(
            [
                {
                    "rung": r.name,
                    "fit_seconds": round(r.fit_seconds, 1),
                    "note": r.note,
                    **r.metrics.as_dict(),
                }
                for r in results
            ],
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
    logger.info("wrote %s", RESULTS)


if __name__ == "__main__":
    main()
