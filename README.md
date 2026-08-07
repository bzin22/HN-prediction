# Hacker News upvote prediction

Predicts the score a Hacker News post will get, using only what is knowable at the
moment it is submitted: the title, the body text under it, the account posting it, the
linked host, and the timestamp.

Word embeddings are trained from scratch in PyTorch, both the CBOW and the Skip-gram
objective, with negative sampling. Three fusion architectures combine the title vector
with the other signals. Validation is a single cut on the time axis: train on 2006 to
2022, test on 2024 and 2025.

**Status: Phase 2 done.** Six baseline models are measured on that split and the numbers
are below. No neural network has been trained yet. The reasoning behind every design
choice, with the plots, is in [`docs/design.md`](docs/design.md).

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

Six baselines, all predicting `log1p(score)`. Error is reported here; the two ranking
metrics are in [`docs/design.md`](docs/design.md#the-reporting-format) with the rest of
the table.

| Rung | Sees | RMSE | MAE |
|---|---|---|---|
| 1. Trailing mean | nothing | 1.191 | 0.854 |
| 2. Author history | the account | 1.210 | 0.844 |
| 3. Domain history | the linked host | 1.220 | 0.852 |
| 4. Body text + Ridge | the body text | 1.197 | 0.808 |
| 5. All signals + Ridge | everything | **1.149** | **0.802** |
| 6. All signals + XGBoost | everything | 1.152 | 0.814 |

**All signals into Ridge is the strongest rung and is the bar Phase 5 has to clear.** It
wins on both columns, 1.149 and 0.802, and it is the only rung that beats the do-nothing
floor on both at once.

**Error alone makes every rung look the same, and that is a fact about the data.** Rung 1
knows nothing and predicts one number; rung 5 sees every signal and cuts RMSE by 3.6%.
The floor explains it: 49.5% of test posts score 1 or 2, so a constant already sits close
to half the data and there is very little absolute error left to win. Read these two
columns as the error floor being low, not as the features being worthless. The ranking
metrics in the design doc are what separate the rungs, and they separate them by a lot.

**Body text alone is close to the floor.** Rung 4 has the best MAE of the single-signal
rungs, 0.808, and it gets there by lowering typical error rather than by telling posts
apart. Only 9.9% of test rows carry any body, and the strongest pattern in the evidence, a
link post with a long body comment, is 0.9% of training rows against 5.0% of test rows.
The habit barely existed while the model was learning. Phase 3 should not count on the
body as a second corpus without re-checking on recent data alone.

**Gradient boosting does not beat the linear model.** Rung 6 takes the identical matrix
and is worse on both columns, for 25 minutes of fitting against 3.

Rung 2 falls back to rung 1 on the 10.0% of test rows whose author has no earlier post,
rung 3 on the 13.8% whose host has none. `make baselines` reproduces all of it into
`artifacts/baselines.json`, which carries all four metrics.

## Phases

| Phase | Deliverable | State |
|---|---|---|
| 0 | Repo, README, scaffold, CI | Done |
| 1 | Ingest and EDA, three gates, leak audit | Done |
| 2 | Time split, six baseline rungs, the metrics that report them | Done |
| 3 | CBOW and Skip-gram on text8, validated against gensim, then the Wikipedia subset | Not started |
| 4 | HN fine-tuning, three-variant comparison | Not started |
| 5 | Early, late and hybrid fusion, plus ablations | Not started |
| 6 | FastAPI and Docker | Not started |
| 7 | Results write-up | Not started |

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make check
```

Heavy dependencies are optional extras, installed with `pip install -e ".[data]"` and so
on for `train` and `serve`. `make baselines` needs both. Rung 6 uses XGBoost, a separate
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
[`models/baselines.py`](src/hn_upvotes/models/baselines.py) and
[`training/metrics.py`](src/hn_upvotes/training/metrics.py). Everything else is a stub
carrying its real type-annotated signature and a docstring saying what it will do.

## Hardware

An Apple M2 with 24 GB of unified memory, PyTorch on the MPS backend. No CUDA, no cloud
GPU. Full English Wikipedia is out of scope, so the embedding corpus is a subset whose
size is set from a measured throughput number in Phase 3. MPS is not available inside a
container, so local training runs outside Docker and the training image exists for
reproducibility elsewhere.

## Licence

MIT for the code. The dataset is `odc-by` and is not redistributed here.
