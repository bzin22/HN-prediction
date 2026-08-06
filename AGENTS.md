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

Read `README.md` before changing the target transform or the embedding hyperparameters.
The reasoning is written down there: why the target is a trailing z-score of
`log1p(score)` rather than raw score, why `year` is not a feature, and where each
hyperparameter comes from. Era drift is asserted and unmeasured until Phase 1 runs its
gates, so do not write it up as measured fact.

Terminology: CBOW and Skip-gram are training objectives. The embedding is the input
weight matrix kept after the task is discarded. "CBOW embeddings" is wrong here.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
