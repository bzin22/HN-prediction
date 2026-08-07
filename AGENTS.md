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
groups are a uv concept that pip cannot install. Most modules under `embeddings/`,
`models/`, `training/` and `serving/` import their extra at module level and will not
import without it. `models/baselines.py` is the exception on purpose: it imports
scikit-learn and xgboost inside `fit`, so the numpy-only rungs and their tests run on a
base install and CI does not need the `train` extra. Keep `src/hn_upvotes/__init__.py`
free of submodule imports so `import hn_upvotes` stays cheap and dependency free.

xgboost needs an OpenMP runtime, which macOS does not ship, so `import xgboost` fails
with a `libomp.dylib` error out of the box. `brew install libomp` is the official fix.
`make baselines` avoids needing it by pointing `DYLD_LIBRARY_PATH` at the copy
scikit-learn's own wheel already carries inside `.venv`. Run rung 5 any other way and
that variable has to be set the same way.

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
   implementation of the window bound is `target/normalise.compute_trailing_baseline`,
   and `features/history.prior_mean` is the same bound keyed by author or hostname.
   Every rung of the baseline ladder goes through one of those two, so there is one
   place to get it wrong. The easy mistake is a mean taken over the whole training set
   and then applied to every row of that key; `tests/test_author_history_bound.py`
   exists to catch exactly that. Anything needing a validation slice takes it from
   `splits.cut_validation_tail`, the last months of the training period, never a random
   sample. Its scored rows may be subsampled for speed and that is safe, because
   `sample_validation_tail` returns a subset of the tail mask and the rows it drops do
   not rejoin the fit set.

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

**Tail drift is an open risk, not a closed finding.** It is unhandled. Phase 2 measured
it on one cut rather than on walk-forward folds, so the fold-over-fold degradation
sequence does not exist yet. Spearman and P@100 are the metrics that expose it.

**Phase 2 chose a single time-based cut, not walk-forward.** Train 2006-10 to 2022-11,
test 2024-01 to 2025-12, and the two unsettled windows dropped. The boundaries are
`data/splits.SplitConfig`, not literals. Walk-forward is deferred until one number is
shown to be hiding something; `splits.walk_forward_folds` is the seam and says so.

**Report rank, not just error, and never quote RMSE alone.** The six baselines move RMSE
by 3.6% end to end (1.191 to 1.149) and Spearman by six times (0.050 to 0.297). 49.5% of
test posts score 1 or 2, so a constant is already near half the data and absolute error
has almost nothing left to win. An RMSE-only table would read as "nothing works".

**Rung 6 was underfitting and it still loses to Ridge. Both halves are the result.** Early
stopping against a held-back year ran to 2,303 trees, 4.6x the untuned 500, stopping early
rather than hitting the 5,000 ceiling. Every metric improved the whole way, 0.2635 to
0.2799 Spearman. Ridge is still ahead at 0.2971, and at 0.2890 on the identical reduced
rows, for 157 seconds of fitting against 4,817. So underfitting explains about half the
original gap and the other half is trees being wrong for a hundred thousand sparse columns.
**Do not tune rung 6 further to close it.** Depth, not tree count, is the only knob left
worth trying, and it was deliberately not tried: a baseline tuned until it wins is not a
baseline. Reasoning in `docs/design.md`, "Was rung 6 underfitting?".

**Anything passing an eval set to XGBoost must use `xgboost.train`, not `XGBRegressor`.**
The wrapper builds eval sets as `QuantileDMatrix`, which has no incremental prediction
cache, so every round re-scores the whole slice. Measured on the real matrix, same 296,531
row slice: 95.13 seconds a round through the wrapper against 4.52 native. That is the
difference between a six day run and a six hour one. `models/baselines.EarlyStoppedXGBoost`
is the worked example, including setting `base_score` by hand because the two APIs pick a
starting value differently.

**Precision@100 does not discriminate at this scale.** Every rung scores 0, 1 or 2 hits,
and 100 random rows out of 599,937 would be expected to hit 0.017 times. Use it to show
that nothing finds the tail, not to rank models. The full numbers are in the README and
`artifacts/baselines.json`.

**Body text is a distribution shift, not just a sparse feature.** Body text was added to
the allowlist in Phase 2, so the legal feature set is now title, `by`, `url`, `time` and
`text`. It carries a trap: 6.6% of training rows have body text against 9.9% of test
rows, and link posts with a body comment are 0.9% of train against 5.0% of test. The
practice grew. Any figure quoted for body text has to say which period it is measured
over, and the widely quoted 11.6% is a 2025 number, not a full-history one (7.4%).

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
   or every month boundary shifts. **This applies to the source shards only.** Ingest
   already applied that cast, so `time` in `data/stories.parquet` is a plain naive UTC
   `TIMESTAMP`. Casting it a second time reinterprets it in the session timezone and
   silently moves rows across month boundaries: it moved 226 rows out of the Phase 2
   training segment before the counts were checked against the known totals.

## Word2vec: what Phase 3 measured, which overrides earlier assumptions

`docs/word2vec.md` is the full reasoning. These are the ones that change what you write.

1. **Train on CPU, not MPS.** Measured 105,142 against 24,383 pairs/s at 100k words by 300
   dimensions, so CPU is 4.3x faster. `select_device()` returns CPU on purpose, against the
   scaffold's original assumption. Sparse gradients do work on MPS; MPS is just slower,
   because the steps are too small to pay for kernel launches. The Python feeder is not the
   bottleneck either (963,848 tokens/s). Re-measure before assuming this holds for a bigger
   model in a later phase.
2. **`learning_rate = 0.025` is measured, not inherited from gensim.** `make lr-sweep` ran
   0.0025, 0.01, 0.025 and 0.05 on the same 5M-token text8 slice at three seeds each. 0.0025
   crawls (ends at 5.52 where the others reach 3.9) and its vectors score nothing. **0.05
   diverged on two of three seeds**, ending at 1.2e11 and 2.5e16. 0.025 is the largest rate
   that converges smoothly and it also wins on both intrinsic scores.
   `tests/test_learning_rate.py` pins it. Do not change it without rerunning the sweep.
3. **Three seeds, because one lies.** On seed 0 alone, 0.05 does not blow up and posts the
   best intrinsic scores, and 0.01 appears to beat 0.025 on WordSim-353. Both reverse over
   three seeds. Any comparison of these intrinsic scores has to clear a noise band: the
   Spearman sampling error is `1 / sqrt(n - 3)`, which is 0.055 at the 336 pairs covered, and
   the measured seed spread was larger still at 0.114.
4. **The optimiser gets `learning_rate * batch_size`, and that is now measured too.** The loss
   is a batch mean, so `0.025 * 1024 = 25.6` reaches the optimiser. `make batch-scaling` ran
   batch 1, 32 and 1024 over identical examples: 4.8183, 4.9987, 5.4211, against **11.0790 for
   batch 1024 with the multiplication removed**, which is the untrained starting loss. So the
   multiplication is the difference between learning and not learning. The residual +12.5% at
   batch 1024 is within-batch staleness, grows with the batch, and is reported not gated.
5. **CBOW needs more corpus than Skip-gram for the same convergence.** With a window of 5 it
   produces one training example per position against Skip-gram's ten, so 3.8x fewer optimiser
   steps on the same text. Any corpus sized on Skip-gram's numbers has to be checked against
   CBOW: the dry-run corpus was not, and CBOW failed its own gate at 0.392 purity the first
   time both objectives ran. `chain.DRY_RUN_CORPUS` is sized for CBOW now.
6. **The gensim comparison has two divergences, not one.** Its `sample` default is 1e-3
   against our 1e-5 and must be passed explicitly, and its subsampling formula is
   `sqrt(t/f) + t/f` where the paper's is `sqrt(t/f)`, which cannot be passed away. Both are
   in `evaluate.train_gensim`.

Two gate metrics were tried and rejected; do not reinvent them. Rank correlation between our
word-pair similarities and gensim's scores 0.012 for a *correct* implementation, because most
random pairs have no defined answer. And the planted-synonym corpus cannot gate anything:
gensim recovers 0 of 6 pairs on it while this implementation recovers 6, so it is too small to
compare against. Gate on task scores instead, task supplied by the caller.

`embeddings/negative_sampling.py` owns both matrices, the sampler and the loss; `cbow.py` and
`skipgram.py` must contain nothing but `forward`. A test enforces that, because a duplicated
sampler is how the CBOW-against-Skip-gram comparison stops meaning anything.

The overnight chain is `embeddings/chain.py`. **Both objectives by default, four stages each,
six variants**, and `make chain-dry-run` proves all eight on synthetic corpora in about 20 seconds.
Artefacts are `{objective}-{stage}.npz` and checkpoints `{objective}-{stage}-epoch{n}.npz`, so
a resume cannot cross objectives and a variant on disk cannot lie about what made it. The two
objectives share the Wikipedia subset, sized from Skip-gram's throughput because it is the
slower one; sizing each to its own ceiling would make the comparison about corpus size.
Checkpoints carry **both** matrices, since resuming from the embedding alone restarts scoring
from zero. A cut-short epoch banks under the previous epoch number and keeps its loss out of
the per-epoch list, or a resume double-counts it.

A night is **7.5 hours expected against 11.0 hours of ceilings**; `chain.overnight_budget` is
the arithmetic and `make hn-token-count` is where its 75,283,676-token input comes from. Two
costs sit outside it and are not measured: gensim's half of each gate, and the Hacker News
reader's real throughput. If it stops fitting, cut epochs on the HN stages, not an objective.

Embedding tests need the `train` extra, which the base install and the main CI job do not
carry, so they `importorskip`. The separate `train-extra` CI job installs the CPU torch wheel
and runs them; without that job they would never run anywhere. Anything else needing that
extra has to be added to that job by name, or it silently never runs.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
