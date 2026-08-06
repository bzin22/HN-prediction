# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Toolchain

`uv` only. Python is pinned to 3.12 in `.python-version` and `pyproject.toml` requires
`>=3.12,<3.13`. Do not run this on the 3.13 installed on the dev machine. `make help`
lists every target; `make check` is exactly what CI runs.

Heavy dependencies are optional extras (`data`, `train`, `serve`), so the default
`uv sync` has no torch. Modules under `embeddings/`, `models/`, `training/` and
`serving/` import their extra at module level and will not import without it. Keep
`src/hn_upvotes/__init__.py` free of submodule imports so `import hn_upvotes` stays
cheap and dependency free.

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

**Era drift was measured in Phase 1 and mostly is not there.** Median `log1p(score)` is
`log1p(2)` in every year from 2007 to 2024, and the standard deviation moves 2.9% across
2010-2025. So `BaselineConfig` now defaults to `centre="zero"`, `scale=False`, and the
training target is plain `log1p(score)`. The machinery is kept and is one config change
to re-enable. What does drift is the extreme tail (99th percentile of raw score 38 in
2007, 355 in 2025), which no centre or spread statistic captures. Do not describe the
transform as justified by drift, and do not re-enable it without a statistic that
measures the tail.

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
