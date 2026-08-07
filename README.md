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

4,738,004 usable stories pulled from
[`open-index/hacker-news`](https://huggingface.co/datasets/open-index/hacker-news) on
Hugging Face, measured as of 2026-08-05. 4,168,189 of those carry final scores and can
serve as labels; the rest fall in months where the archive recorded the score at
submission, before anyone had voted
([why](docs/design.md#scores-that-are-not-final)).

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
on for `train` and `serve`. `make baselines` reproduces the table above and needs both
`data` and `train`.

Rung 6 uses XGBoost, which is a separate package with a scikit-learn compatible
estimator, not part of scikit-learn. It needs an OpenMP runtime, which macOS does not
ship. `make baselines` points at the copy scikit-learn's own wheel already carries, so
nothing extra is needed; `brew install libomp` is the official alternative.

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
