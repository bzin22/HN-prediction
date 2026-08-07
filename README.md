# Hacker News upvote prediction

Predicts the score a Hacker News post will get, using only what is knowable at the
moment it is submitted: the title, the body text under it, the account posting it, the
linked host, and the timestamp.

Word embeddings are trained from scratch in PyTorch, both the CBOW and the Skip-gram
objective, with negative sampling. Three fusion architectures combine the title vector
with the other signals. Validation is a single cut on the time axis: train on 2006 to
2022, test on 2024 and 2025.

**Status: Phase 2 done, Phase 3 implemented and tuned.** Six baseline models are measured
on that split and the numbers are below. Both word2vec objectives are written and their two
hyperparameter claims are settled on text8, but no embedding variant has been trained yet.
The reasoning behind every design choice, with the plots, is in
[`docs/design.md`](docs/design.md).

## The data

4,739,207 usable stories pulled from
[`open-index/hacker-news`](https://huggingface.co/datasets/open-index/hacker-news) on
Hugging Face, measured as of 2026-08-06. 4,168,189 of those carry final scores and can
serve as labels; the rest fall in months where the archive recorded the score at
submission, before anyone had voted
([why](docs/design.md#scores-that-are-not-final)).

Nine columns, including `text`, which is Hacker News's own name for the body the poster
writes under the headline. Pulling it takes the transfer from 776 MB to 6.96 GB and the
ingest from 4 minutes to 37. Link posts and text posts are not exclusive categories: 7.4%
of stories carry body text, 6.2% have no link, and 1.8% have both a link and a body. All
three shares are rising. Over 2025 alone, 11.3% carried body text and 6.5% had both.

## Phase 1 gate results

Scoring inflation is real and it sits in the top 1% of posts. The 99th percentile went
from 38 points in 2007 to 355 in 2025, while the typical post did not move. The trailing
z-score transform the project was built around was switched off, because the statistics it
corrects, the centre and the spread, cannot see the tail
([detail](docs/design.md#what-phase-1-measured)). The training target is now plain
`log1p(score)`.

A score reaches its final value between 12 and 24 hours after submission, measured from a
live read of 13,852 posts.

Three defaults in [`target/normalise.py`](src/hn_upvotes/target/normalise.py) changed.
Every measurement behind them, with the plots, is in that design doc section and in
[`notebooks/01-eda.ipynb`](notebooks/01-eda.ipynb).

| Default | Was | Now |
|---|---|---|
| `centre` | `"mean"` | `"zero"` |
| `scale` | `True` | `False` |
| `lag` | 48 h | 24 h |

## Phase 2 results

One time-based split: 3,568,252 training rows from 2006-10 to 2022-11 and 599,937 test
rows from 2024-01 to 2025-12. The 571,018 rows between and after them are dropped because
the archive recorded those scores at submission, before anyone had voted.

Six baselines, all predicting `log1p(score)`, plus rung 6 re-run with early stopping as
6b. Rung 6b is fitted on 296,531 fewer rows, because those are the validation tail it
stops on. Spearman is rank correlation: it compares
two orderings and ignores the actual numbers, 1 if they order posts identically and 0 if
there is no relationship. Precision@100 is in
[`docs/design.md`](docs/design.md#the-reporting-format) with the full table.

| Rung | Sees | Trees | RMSE | MAE | Spearman |
|---|---|---|---|---|---|
| 1. Trailing mean | nothing | | 1.191 | 0.854 | 0.050 |
| 2. Author history | the account | | 1.210 | 0.844 | 0.176 |
| 3. Domain history | the linked host | | 1.220 | 0.852 | 0.154 |
| 4. Body text + Ridge | the body text | | 1.197 | 0.808 | 0.040 |
| 5. All signals + Ridge | everything | | **1.149** | **0.802** | **0.297** |
| 6. All signals + XGBoost, 500 trees | everything | 500 | 1.152 | 0.814 | 0.265 |
| 6b. The same, early stopped | everything | 2,303 | 1.149 | 0.809 | 0.280 |

**The models learn order, not magnitude. That is the Phase 2 result.** Rung 1 to rung 5
spreads Spearman sixfold, 0.050 to 0.297, while RMSE moves 3.6%. The floor explains the
gap: 49.5% of test posts score 1 or 2, so a constant already sits close to half the data
and there is little absolute error left to win. Ordering is the question the features
answer.

**All signals into Ridge is the strongest rung** and the bar Phase 5 has to clear. It wins
on all three columns, though rung 6b ties it on RMSE at three decimals and only loses at
four, 1.1485 against 1.1490.

**Body text alone is close to the floor.** Rung 4 has the best MAE of the single-signal
rungs, 0.808, on a Spearman of 0.040, below rung 1's 0.050: it lowers typical error and
does not order posts. Only 9.9% of test rows carry any body, and the strongest pattern in
the evidence, a link post with a long body comment, is 0.9% of training rows against 5.0%
of test rows. The habit barely existed while the model was learning. Phase 3 should not
count on the body as a second corpus without re-checking on recent data alone.

**Gradient boosting does not beat the linear model, and it was underfitting.** Both are
true. Rung 6 at 500 trees is worse than Ridge on all three columns for 25 minutes of
fitting against 3. It was also starved: 500 trees at depth 6 is at most 31,500 splits
against 100,010 features.

Re-running it with early stopping settles it. The stopping point is chosen on the last
twelve months of the training period, 2021-12 to 2022-11, cut off the end and never
trained on: 296,531 rows held back, 3,271,721 left to fit on. A time cut, not a random
sample, or the tree count would be picked using posts from the same weeks the model
trained on. The test period is untouched and scored once. It stopped at
**2,303 trees, 4.6 times the 500**, and it stopped early rather than reaching the 5,000
ceiling, so that is a measurement and not a limit. Every metric improved the whole way up.
Off the one fitted model, scored on the test period at each tree count:

| Trees | RMSE | MAE | Spearman |
|---|---|---|---|
| 500 | 1.1530 | 0.8116 | 0.2635 |
| 1,000 | 1.1509 | 0.8103 | 0.2715 |
| 2,000 | 1.1492 | 0.8089 | 0.2787 |
| 2,303 | 1.1490 | 0.8087 | 0.2799 |

**Ridge still wins, and that is the answer.** Rank correlation 0.297 against 0.280, MAE
0.802 against 0.809, RMSE tied at 1.149 to three decimals. Ridge fits in 157 seconds and
the early-stopped rung takes 4,817, so gradient boosting costs 31 times as much to lose.
The comparison is not flattered by the held-back year either: Ridge fitted on the same
reduced rows scores 0.289, still ahead of 0.280.

So underfitting was real and it explains about half the original gap, 0.0165 of 0.032 in
rank correlation. The other half is not a tuning problem. Ranking gains were down to
+0.004 over the last 800 trees when it stopped, so more trees will not close it. **No
further tuning was done.** A baseline that gets tuned until it wins has stopped being a
baseline. `docs/design.md` has the reasoning and the recommendation.

Rung 2 falls back to rung 1 on the 10.0% of test rows whose author has no earlier post,
rung 3 on the 13.8% whose host has none. `make baselines` reproduces rungs 1 to 6 into
`artifacts/baselines.json` and `make tune-xgboost` reproduces the early-stopped run into
`artifacts/xgboost-early-stopping.json`, both with all four metrics.

## Phases

| Phase | Deliverable | State |
|---|---|---|
| 0 | Repo, README, scaffold, CI | Done |
| 1 | Ingest and EDA, three gates, leak audit | Done |
| 2 | Time split, six baseline rungs, the metrics that report them | Done |
| 3 | CBOW and Skip-gram on text8, validated against gensim, then the Wikipedia subset | Implemented and tuned, no variant trained yet |
| 4 | HN fine-tuning, three-variant comparison | Not started |
| 5 | Early, late and hybrid fusion, plus ablations | Not started |
| 6 | FastAPI and Docker | Not started |
| 7 | Results write-up | Not started |

## Phase 3: the implementation, before any training

CBOW and Skip-gram are implemented from scratch and the overnight training chain is wired
up for both. **No embedding variant has been produced yet**, by design: the algorithms get
reviewed before a night is spent. text8 has been trained on for two hyperparameter
measurements and nothing else, at 5M and 120k tokens. The reasoning is in
[`docs/word2vec.md`](docs/word2vec.md).

Two measurements decided the shape of it, both from `make throughput`.

**Sparse gradients work on both CPU and MPS.** torch 2.13.0 at 100,000 words by 300
dimensions produces a genuine sparse gradient, 19.7 MB of touched rows against 120 MB
dense, and SGD steps on it.

**Run it on CPU, not the GPU.** Training pairs per second, skip-gram at the full
vocabulary:

| Device | Sparse gradients | Dense gradients |
|---|---|---|
| **CPU** | **105,142** | 34,469 |
| MPS | 24,383 | 38,335 |

CPU is 4.3x faster than MPS, and the Python feeder is not the bottleneck either: it moves
963,848 tokens/s against the 81,603 the fastest training configuration consumes. The cost
is the embedding backward, and a step touching 16,000 rows of 300 floats is too little
arithmetic to pay for MPS kernel launches. This is why gensim is CPU threads with no GPU.
`select_device()` returns CPU as a result, against the scaffold's assumption.

That throughput sizes the Wikipedia stage rather than a guess doing it. At `k = 5` and a
two-hour ceiling over 5 epochs: `138,362 x 0.96 x 7,200 / 5 = 191,271,628` tokens, about
1.13 GB.

```bash
make throughput      # the two measurements above
make lr-sweep        # four learning rates on text8, three seeds each, about 25 minutes
make batch-scaling   # the batch-size rate multiplication against a batch-1 run, minutes
make chain-dry-run   # both objectives, every stage, synthetic corpora, about 20 seconds
make chain           # the real overnight run, detached
```

### Two hyperparameter claims, now measured on text8

**The learning rate is 0.025, and that is the sweep's answer rather than gensim's.** Four
rates over an identical 5,000,000-token slice of text8, three seeds each, one epoch,
dimension 300:

| Rate | Blew up | Final loss | Worst lurch, as a share of the fall | WordSim-353 | Analogy |
|---|---:|---:|---:|---:|---:|
| 0.0025 | 0 of 3 | 5.5199 | 3.1% | -0.015 | 0.00003 |
| 0.01 | 0 of 3 | 3.9568 | 3.0% | 0.031 | 0.00000 |
| **0.025** | **0 of 3** | **3.8915** | **5.8%** | **0.079** | **0.00028** |
| 0.05 | **2 of 3** | 4.1501 | 24.9% | 0.089 | 0.00064 |

The rule is the largest rate that still converges smoothly. 0.0025 crawls: it ends at a loss
the others pass in the first 5% of the run, and its vectors score nothing on all three seeds.
0.05 destabilises, and more than predicted: it ran away to 1.2e11 on one seed and 2.5e16 on
another, both inside the first tenth of the epoch. Its loss columns above are the one seed
that survived, and even that one lurches by 1.50 in a single block, 24.9% of its total fall
against 5.8% for 0.025. So 0.025 it is, unchanged.

**Three seeds, because one lies.** On seed 0 alone, 0.05 does not blow up and posts the best
score in both intrinsic columns, and 0.01 appears to beat 0.025 on WordSim-353. Both reverse
over three seeds. Every comparison now has to clear a noise band: the sampling error on a
Spearman correlation over the 336 pairs covered is 0.055, and the measured spread across seeds
was 0.114.

**The batch-size multiplication is what makes a large batch train at all.** The optimiser
gets `learning_rate * batch_size`, which is `0.025 * 1024 = 25.6` and looks wrong. Four arms
over the same 83,331 training examples, batch size the only difference:

| Arm | Optimiser rate | Final loss | Against batch 1 |
|---|---:|---:|---:|
| batch 1 | 0.025 | 4.8183 | |
| batch 32 | 0.8 | 4.9987 | +3.7% |
| batch 1024 | 25.6 | 5.4211 | +12.5% |
| batch 1024, multiplication removed | 0.025 | 11.0790 | +129.9% |

Delete the multiplication and the batch does not move: 11.0790 is the loss a zero output
matrix forces before any training. Keep it and nothing diverges. The residual +12.5% is a cost
of batching that no rate can undo, because updates inside a batch do not see each other, and
it grows with the batch as that explanation predicts.

### The chain runs both objectives

One invocation, Skip-gram then CBOW, four stages each, **six variants**. A gate runs first for
each objective: train on text8, train gensim on the same corpus with the same settings and the
matching `sg` flag, compare, and **abort that objective if our vectors are meaningfully
worse**. Then the three variants, with `hn-only` last so a Wikipedia failure costs one variant
instead of the night. Artefacts are `{objective}-{stage}.npz`, a CBOW failure cannot cost the
Skip-gram variants, and a resume does not restart an objective that already finished.

**A night is 7.5 hours expected against 11.0 hours of ceilings**, so it fits. CBOW is the cheap
half at 2.3 hours against Skip-gram's 5.2, because it is 2.2x faster per corpus token. The HN
corpus is counted rather than guessed: 75,283,676 tokens over 5,091,739 lines. If it stops
fitting, the lever is epochs on the HN stages, not an objective. See
[`docs/word2vec.md`](docs/word2vec.md) for the stage-by-stage table and the three costs that
sit outside it.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make check
```

Heavy dependencies are optional extras, installed with `pip install -e ".[data]"` and so
on for `train` and `serve`. `make baselines` and `make tune-xgboost` need both. Rung 6
uses XGBoost, a separate
package with a scikit-learn compatible estimator that is not part of scikit-learn. It
needs an OpenMP runtime, which macOS does not ship, so `make baselines` points at the copy
scikit-learn's own wheel already carries. `brew install libomp` is the alternative.

## Repository layout

Code is `src/hn_upvotes/`, split into `data/`, `embeddings/`, `features/`, `target/`,
`models/`, `training/` and `serving/`. Implemented so far:
[`features/schema.py`](src/hn_upvotes/features/schema.py),
[`features/history.py`](src/hn_upvotes/features/history.py),
[`features/body.py`](src/hn_upvotes/features/body.py),
[`features/domain.py`](src/hn_upvotes/features/domain.py),
[`features/temporal.py`](src/hn_upvotes/features/temporal.py),
[`target/normalise.py`](src/hn_upvotes/target/normalise.py),
[`data/ingest.py`](src/hn_upvotes/data/ingest.py),
[`data/splits.py`](src/hn_upvotes/data/splits.py),
[`data/preprocess.py`](src/hn_upvotes/data/preprocess.py),
[`data/gates.py`](src/hn_upvotes/data/gates.py),
[`models/baselines.py`](src/hn_upvotes/models/baselines.py),
[`training/metrics.py`](src/hn_upvotes/training/metrics.py),
[`embeddings/negative_sampling.py`](src/hn_upvotes/embeddings/negative_sampling.py),
[`embeddings/cbow.py`](src/hn_upvotes/embeddings/cbow.py),
[`embeddings/skipgram.py`](src/hn_upvotes/embeddings/skipgram.py),
[`embeddings/train.py`](src/hn_upvotes/embeddings/train.py),
[`embeddings/corpora.py`](src/hn_upvotes/embeddings/corpora.py),
[`embeddings/checkpoint.py`](src/hn_upvotes/embeddings/checkpoint.py),
[`embeddings/throughput.py`](src/hn_upvotes/embeddings/throughput.py),
[`embeddings/learning_rate_sweep.py`](src/hn_upvotes/embeddings/learning_rate_sweep.py),
[`embeddings/batch_scaling.py`](src/hn_upvotes/embeddings/batch_scaling.py) and
[`embeddings/chain.py`](src/hn_upvotes/embeddings/chain.py). Everything else is a stub
carrying its real type-annotated signature and a docstring saying what it will do.

## Hardware

An Apple M2 with 24 GB of unified memory. No CUDA, no cloud GPU. Full English Wikipedia is
out of scope, so the embedding corpus is a subset sized from a measured throughput number,
which came out at about 191M tokens for a two-hour ceiling.

**Word2vec training runs on the CPU, not the MPS backend.** Phase 3 measured CPU at 4.3x
MPS for this workload, because the steps are too small to pay for GPU kernel launches. MPS
is not available inside a container either, so local training runs outside Docker and the
training image exists for reproducibility elsewhere. A larger model in a later phase may
well favour MPS; that is a separate measurement.

## Licence

MIT for the code. The dataset is `odc-by` and is not redistributed here.
