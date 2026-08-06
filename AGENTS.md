# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Toolchain

Plain `venv` and `pip`, no uv. `make setup` builds `.venv` and installs `-e ".[dev]"`.
`make help` lists every target; `make check` is exactly what CI runs. There is no lock
file: `uv.lock` was removed with uv, so installs resolve fresh. Add a `requirements.txt`
from `pip freeze` if that becomes a problem.

`requires-python` is `>=3.12`. CI tests 3.12 because that is the supported floor, and
`.python-version` says 3.13 because that is the dev machine. Both ends get exercised. The
old `<3.13` cap existed because torch and gensim lagged; both cleared 3.13 by 2026-08-06.

Heavy dependencies are optional extras (`data`, `train`, `serve`), so a default install
has no torch. `dev` is a normal extra rather than a dependency group, because dependency
groups are a uv concept that pip cannot install. Modules under `embeddings/`, `models/`,
`training/` and `serving/` import their extra at module level and will not import without
it. Keep `src/hn_upvotes/__init__.py` free of submodule imports so `import hn_upvotes`
stays cheap and dependency free.

## Column names are Hacker News's, not ours

`descendants` is the comment count, `kids` is the list of reply ids, `by` is the author.
They are the upstream API's own field names, confirmed against
`https://hacker-news.firebaseio.com/v0/item/8863.json`, and the code keeps them so the
schema matches the source. When writing prose, use the plain English name and say the raw
name is Hacker News's own. Do not rename these columns casually: `by` is on the feature
allowlist, which is a protected invariant, so a rename is its own reviewed change.

## Two invariants that are not conventions

These are enforced by tests. Do not weaken either to make something pass.

1. **Feature allowlist.** `src/hn_upvotes/features/schema.py` is the only place that
   decides what a model may see. `score`, `descendants` and `kids` are post-hoc and can
   never be features. `tests/test_feature_schema.py` restates the banned names
   independently, so editing `schema.py` cannot take the test with it.
2. **Strictly earlier.** Every rolling statistic and every split is computed from rows
   strictly before the row it serves. No random splits anywhere. The reference
   implementation of the window bound is `target/normalise.compute_trailing_baseline`.

## Design decisions that are argued, not incidental

`docs/design.md` holds the reasoning; `README.md` holds the results and how to run it.
Read the design doc before changing the target transform, the tokeniser, or the embedding
hyperparameters.

**Era drift is real and it is in the tail. Phase 1 measured it. Do not write that it was
refuted.** The 99th percentile of raw score went from 38 points in 2007 to 355 in 2025,
and the 99.9th from 89 to 1,001.

The four statistics a trailing transform can use are all blind to that. Median
`log1p(score)` is `log1p(2)` in every year from 2007 to 2024, the standard deviation
moves 2.9% across 2010-2025, and the IQR moves the other way. The flat median is largely
a floor artefact: 57.6% of stories score 1 or 2, so it is pinned there whatever happens
above it. Never cite the flat median as evidence that scoring is stable.

So `BaselineConfig` defaults to `centre="zero"`, `scale=False` and the target is plain
`log1p(score)`. The reason is that switching the transform **on** would not have
corrected the tail either, not that there is nothing to correct. The machinery is kept
and is one config change to re-enable. Do not re-enable it without a statistic that
actually tracks the tail.

**Tail drift is an open risk, not a closed finding.** It is unhandled, and Phase 2's
walk-forward folds are where it has to be dealt with. Spearman and P@100 are the metrics
that will expose it.

Terminology: CBOW and Skip-gram are training objectives. The embedding is the input
weight matrix kept after the task is discarded. "CBOW embeddings" is wrong here.

## Sharp edges in the source data

All four are encoded in `data/ingest.py`; the evidence is in `docs/design.md`.

1. **Not every archived score is final.** `UNSETTLED_SCORE_MONTHS` lists 21 months
   (2022-12 to 2023-12 and 2026-01 to 2026-08, 12.0% of rows) whose `score` was captured
   at submission. Anything reading `score` as a label must go through
   `ingest.drop_unsettled_months` first. The archive is live, so this list can grow:
   re-check it by refetching a monthly sample from the live HN API.
2. `by` is a reserved word in DuckDB and must be double quoted in every query.
3. Missing values are sentinels, not `NULL`. An absent title is `''`, an absent score
   is `0`, so an `IS NOT NULL` filter silently keeps everything.
4. `time` is `TIMESTAMP_MICROS` in UTC, not unix seconds. Cast with `AT TIME ZONE 'UTC'`
   or every month boundary shifts.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
