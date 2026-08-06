# Hacker News upvote prediction

Predicts the score a Hacker News post will get, using only what is knowable at the
moment it is submitted: the title, the account posting it, the linked domain, and the
timestamp.

Word embeddings are trained from scratch in PyTorch, both the CBOW and the Skip-gram
objective, with negative sampling. Three fusion architectures combine the title vector
with the author, domain and time signals. Validation is walk-forward on the time axis.

**Status: Phase 1 done.** The data is ingested and the three gates have run. No model has
been trained yet. The reasoning behind every design choice, with the plots and the
reporting format, is in [`docs/design.md`](docs/design.md).

## The data

4,738,004 usable stories pulled from
[`open-index/hacker-news`](https://huggingface.co/datasets/open-index/hacker-news) on
Hugging Face, measured as of 2026-08-05. 4,168,189 of those carry final scores and can
serve as labels; the rest fall in months where the archive recorded the score at
submission, before anyone had voted
([why](docs/design.md#scores-that-are-not-final)).

## The three failure modes

Three ways to post good offline numbers and be worthless in production. Full argument for
each in [`docs/design.md`](docs/design.md).

**Target leakage: training on the answer.** `score` is the thing being predicted, so it is
the target and never an input. Using it as one is reading the answer off the back of the
card. Two more columns are just as illegal and easier to miss: `descendants`, the comment
count on the post, and `kids`, the list of ids of its direct replies. Both only exist once
people have reacted, so neither is knowable at submission. The defence is an allowlist in
[`features/schema.py`](src/hn_upvotes/features/schema.py) rather than a blocklist, so a
column nobody approved is rejected by default. The legal feature set is exactly `title`,
`by`, `url`, `time`.

**Temporal leakage: testing on the past using the future.** A random split puts posts from
2025 in the training set and posts from 2019 in the test set, so the model scores well on
2019 partly because it has already seen how 2025 turned out. No submission-time predictor
could, and the reported number is not achievable in production. The defence is that there
are no random splits anywhere: validation is walk-forward, and every rolling statistic is
computed from rows strictly earlier than the row it serves.

**Era drift: the ruler moves.** Scoring inflation shows up in the top 1% of posts, where
the 99th percentile went from 38 points in 2007 to 355 in 2025, while the typical post did
not move. The trailing z-score transform the project was built around was switched off,
because its statistics cannot see the tail
([detail](docs/design.md#what-phase-1-measured)).

## Phase 1 gate results

The centre and spread of the score distribution do not drift, so the target transform came
out and the training target is now plain `log1p(score)`. A score reaches its final value
between 12 and 24 hours after submission, measured from a live read of 13,852 posts. That
covers 4,738,004 usable stories.

Three defaults in [`target/normalise.py`](src/hn_upvotes/target/normalise.py) changed.
The measurements behind each, with the plots, are in
[`docs/design.md`](docs/design.md#what-phase-1-measured) and
[`notebooks/01-eda.ipynb`](notebooks/01-eda.ipynb).

| Default | Was | Now |
|---|---|---|
| `centre` | `"mean"` | `"zero"` |
| `scale` | `True` | `False` |
| `lag` | 48 h | 24 h |

## What Phase 2 does

Phase 2 builds the simple models that any fancier model has to beat, from predicting the
recent average up to TF-IDF with gradient boosting, and the walk-forward harness, which
trains on one time window and tests on the next so a result is quotable rather than an
artefact of where the split landed. It is also where tail drift gets confronted: Spearman
correlation and precision-at-100 both measure ranking in the region the drift lives in, so
they are the metrics that will expose it. If the neural models in later phases do not beat
the simple ones, this README will say so.

| Phase | Deliverable | State |
|---|---|---|
| 0 | Repo, README, scaffold, CI | Done |
| 1 | Ingest and EDA, three gates, leak audit | Done |
| 2 | Baseline models to beat, walk-forward harness, tail drift confronted | Not started |
| 3 | CBOW and Skip-gram on text8, validated against gensim, then the Wikipedia subset | Not started |
| 4 | HN fine-tuning, three-variant comparison | Not started |
| 5 | Early, late and hybrid fusion, plus ablations | Not started |
| 6 | FastAPI and Docker | Not started |
| 7 | Results write-up | Not started |

## Repository layout

Code is `src/hn_upvotes/`, split into `data/`, `embeddings/`, `features/`, `target/`,
`models/`, `training/` and `serving/`. Implemented so far:
[`features/schema.py`](src/hn_upvotes/features/schema.py),
[`target/normalise.py`](src/hn_upvotes/target/normalise.py),
[`data/ingest.py`](src/hn_upvotes/data/ingest.py),
[`data/preprocess.py`](src/hn_upvotes/data/preprocess.py) and
[`data/gates.py`](src/hn_upvotes/data/gates.py). Everything else is a stub carrying its
real type-annotated signature and a docstring saying what it will do.

## Hardware

An Apple M2 with 24 GB of unified memory, PyTorch on the MPS backend. No CUDA, no cloud
GPU. Full English Wikipedia is out of scope, so the embedding corpus is a subset whose
size is set from a measured throughput number in Phase 3. MPS is not available inside a
container, so local training runs outside Docker and the training image exists for
reproducibility elsewhere.

## Licence

MIT for the code. The dataset is `odc-by` and is not redistributed here.
