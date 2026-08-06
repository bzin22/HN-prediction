# Hacker News upvote prediction

Predicts the score a Hacker News post will get, using only what is knowable at the
moment it is submitted: the title, the account posting it, the linked domain, and the
timestamp.

Word embeddings are trained from scratch in PyTorch, both the CBOW and the Skip-gram
objective, with negative sampling. Three fusion architectures combine the title vector
with the author, domain and time signals. Validation is walk-forward on the time axis.

**Status: Phase 1 done.** The data is ingested and the three measurement gates have run.
No model has been trained yet, so the model tables below are empty. The reasoning behind
every design choice is in [`docs/design.md`](docs/design.md).

The headline result: **era drift is real and the standard fix aims at the wrong
statistic.** Scoring inflation is concentrated in the tail, where the 99th percentile of
raw score went from 38 points in 2007 to 355 in 2025. The trailing z-score transform this
project was built around corrects the mean and the spread, both of which moved under 3%,
so it was switched off. Handling the tail is
[an open problem for Phase 2](#the-drift-is-in-the-tail).

## The data

4,738,004 usable stories, filtered from the 49,119,480 items in
[`open-index/hacker-news`](https://huggingface.co/datasets/open-index/hacker-news) on
Hugging Face. Measured 2026-08-05. **The archive is live and updates every five minutes,
so that is a snapshot count, not a constant.**

The plan expected "roughly 4M to 6M". That was an expectation. 4,738,004 is a
measurement.

Of those, 4,168,189 have scores that can be used as labels. The other 569,815 fall in
two windows of months where the archive recorded the score at submission instead of
after the post finished scoring, which was found by refetching samples from the live HN
API. [Details and the evidence table](docs/design.md#scores-that-are-not-final).

Reading the eight needed columns transfers 776 MB, not the 11.93 GB the full sixteen
occupy. DuckDB pushes the projection into the Parquet reader. `text` alone is 6.18 GB.

## The three failure modes

Stated briefly here. The full argument for each is in
[`docs/design.md`](docs/design.md).

**Target leakage.** `score`, `descendants` and `kids` all describe what happened after a
post went live, so none can be a feature. Enforcement is an allowlist in
[`features/schema.py`](src/hn_upvotes/features/schema.py), not a blocklist: a column
nobody approved is rejected, so a new source column cannot silently promote itself.
`descendants` is the one that catches people, because it reads like metadata about the
post rather than about its reception.

**Temporal leakage.** No random splits anywhere. A random split trains on posts
submitted after the ones it is tested on. Validation is walk-forward, and every rolling
statistic is computed from rows strictly earlier than the row it serves.

**Era drift.** The claim was that the same quality of post earns more points now than it
used to, so a model trained on old data would under-predict new posts. **The drift is
real, and it is in the tail.** The 99th percentile of raw score went from 38 points in
2007 to 355 in 2025, and the 99.9th from 89 to 1,001.

The trailing transform still came out, because it cannot see that. Every statistic it
subtracts or divides by (mean, median, standard deviation, interquartile range) is blind
to the tail, so switching it on would not have corrected the drift either. It was the
wrong instrument for the drift that exists, not the right instrument for a drift that
does not. **Tail drift is unhandled and is an open risk for
[Phase 2](docs/design.md#why-not-raw-score).**

The legal feature set is exactly `title`, `by`, `url`, `time`.

## Phase 1 gate results

Each gate changed a default in
[`target/normalise.py`](src/hn_upvotes/target/normalise.py). Reproduced by
[`notebooks/01-eda.ipynb`](notebooks/01-eda.ipynb).

| Question | Answer | What changed |
|---|---|---|
| Does the median drift by year? | **No, but the median cannot answer this.** It is 1.0986 = `log1p(2)` in every year from 2007 to 2024, and 57.6% of stories score 1 or 2, so it is pinned to the floor | `centre`: `"mean"` to `"zero"` |
| Does the spread drift too? | **Not as the transform measures it.** Standard deviation moves 1.1386 to 1.1721 over 2010-2025, which is 2.9%. The IQR moves the other way, 1.2528 to 0.8473 | `scale`: `True` to `False` |
| Where is the drift, then? | **The tail.** 99th percentile of raw score: 38 points in 2007, 355 in 2025. 99.9th: 89 to 1,001 | Nothing. **Open risk for Phase 2** |
| How long does a score take to settle? | **Between 12 and 24 hours** | `lag`: 48 h to 24 h |
| Usable story rows after filtering | **4,738,004**, of which 4,168,189 carry final scores | Snapshot dated 2026-08-05 |

Both transform steps are now off, so the training target is plain `log1p(score)`. That is
not because the data is stationary. It is because the transform's statistics cannot see
the part that moves.

### Gate 1: the median is flat, and it is pinned

![Median and mean log1p(score) per year](docs/img/gate1-median-by-year.png)

The median takes exactly one value for eighteen consecutive years: a median score of 2
points, every year from 2007 to 2024. The mean wanders in a band between 1.43 and 1.70
with no trend. 2023 is missing because its archived scores are not final.

**Read the flat line carefully.** 57.6% of stories score 1 or 2 points, so more than half
the distribution sits on the floor and the median is stuck there by construction. It
would read `log1p(2)` every year no matter what happened in the upper half. Eighteen flat
years is weak evidence about scoring inflation, not strong evidence against it.

What the flat median does establish is narrower and still useful: there is no centre for
a trailing subtraction to remove.

### Gate 2: the spread is flat as the transform measures it

![Standard deviation and IQR of log1p(score) per year](docs/img/gate2-spread-by-year.png)

The standard deviation moves 2.9% across sixteen years. The interquartile range moves in
the opposite direction, with a year correlation of -0.098. Two measures of the same
quantity drifting in opposite directions means neither is tracking a real trend.

Both are pulled toward the floor by the same spike. The IQR especially: with 57.6% of
stories at 1 or 2 points, its 25th percentile is `log1p(1)` in every single year.

So gates 1 and 2 do not say the distribution is stable. They say the four statistics the
trailing transform can use are stable, which is a different and much weaker statement.

### The drift is in the tail

Raw score by percentile, over settled months:

| Percentile | 2007 | 2025 | Ratio |
|---|---|---|---|
| 50th | 2 | 3 | 1.5x |
| 75th | 6 | 6 | 1.0x |
| 90th | 13 | 26 | 2.0x |
| 95th | 19 | 89 | 4.7x |
| 99th | 38 | 355 | 9.3x |
| 99.9th | 89 | 1,001 | 11.2x |

Below the 75th percentile, nothing moves. Above the 95th, everything does. The plan's
illustration was "a great post in 2011 got 50, in 2025 it gets 300"; for great posts that
is close to right, and for a typical post it is wrong.

**This is the drift the project set out to handle, and the transform does not handle
it.** The standard deviation of `log1p(score)` moved 2.9% while the 99th percentile of
raw score moved 9.3x. An affine transform fitted to a statistic that barely moves cannot
correct a tail that moved by an order of magnitude, so turning it on would not have
helped. Switching it off costs nothing and removes machinery that would otherwise imply
the problem was solved.

The residual risk is carried forward, not closed. A model trained on early years and
tested on recent ones will mis-rank the tail, which is exactly the region a
submission-time predictor is for. Phase 2's walk-forward folds are where this has to be
dealt with, and Spearman and P@100 are the metrics that will show it.

### Gate 3: scores settle between 12 and 24 hours

![Mean log1p(score) by age at observation, with confidence intervals](docs/img/gate3-settling-bands.png)

13,852 stories had their current scores read from the live HN API in one pass at
2026-08-06 06:22 UTC. One observation instant across posts of many ages gives the
settling curve directly.

| Age at observation | Mean `log1p(score)` | 95% interval |
|---|---|---|
| 0-12 h | 1.490 | 1.377 to 1.603 |
| 12-24 h | 1.733 | 1.662 to 1.816 |
| 24-48 h | 1.656 | 1.597 to 1.719 |
| 48-72 h | 1.534 | 1.467 to 1.603 |
| 72-168 h | 1.782 | 1.732 to 1.829 |
| 14 days and older | 1.683 | 1.656 to 1.712 |

Only the 0-12 hour band sits below the settled reference with a clear gap. The 12-24 hour
band already overlaps it. Later bands wander above and below by more than their
intervals, which is day-to-day cohort variation rather than settling, so the answer is
the top of the interval where settling completes: 24 hours.

Each band is averaged over its UTC hour-of-day cells with equal weight, because with one
observation instant a post's age and its submission hour are locked together. The raw
unbalanced curve is
[`gate3-settling-curve.png`](docs/img/gate3-settling-curve.png); at this sample size the
mean is noise and the median is pinned to the floor spike, which is why the band chart
carries the conclusion.

**Caveat, stated plainly: these are different posts at different ages, not the same post
tracked over time.** The balancing removes the largest version of that; a residual cohort
effect is small but real.

The archive's own `committed_at` timestamps cannot be used for this gate, for a reason
worth reading:
[why](docs/design.md#why-gate-3-could-not-use-committed_at).

### The score distribution

![Raw and log1p score distributions](docs/img/score-distribution.png)

| Score | Stories | Share |
|---|---|---|
| 1 | 1,372,666 | 32.9% |
| 2 | 1,027,583 | 24.7% |
| 3 | 488,649 | 11.7% |
| **1 or 2** | **2,400,249** | **57.6%** |

The plan's claim is confirmed. 57.6% of stories score 1 or 2, and the spike survives the
log transform, because `log1p` is monotone and cannot spread a point mass. It moves to
0.693 and 1.099 and stays exactly as tall.

That is why percentile rank stays a secondary readout and does not become the target:
more than half of all stories would share one of two percentile ranks.

### Tokenisation

The project uses its own tokeniser, not the dump's pre-tokenised `words` column. `words`
covers `text` and never `title`: 2,474 of the 30,102 titled stories in 2026-06, or 8.2%.
It is also sorted and deduplicated, so word order is gone, and CBOW and Skip-gram both
need a context window. On the input they share they reach a micro Jaccard of 0.899.
[Full comparison](docs/design.md#tokenisation-the-words-column-against-ours).

## Results

**Nothing below has been measured.** These tables are the shape of the output, published
empty so the reporting format is fixed before any number exists. Every number will be
quoted as mean plus or minus standard deviation across five seeds, and two models
differing by less than the seed noise are reported as indistinguishable rather than
ranked. Metrics and the walk-forward protocol are in
[`docs/design.md`](docs/design.md#evaluation).

### Model comparison

| Model | RMSE (target) | MAE (target) | RMSE (`log1p` score) | Spearman | P@100 |
|---|---|---|---|---|---|
| 1. Trailing baseline | | | | | |
| 2. Author mean | | | | | |
| 3. Domain mean | | | | | |
| 4. TF-IDF + Ridge | | | | | |
| 5. TF-IDF + GBM | | | | | |
| Early fusion | | | | | |
| Late fusion | | | | | |
| Hybrid fusion | | | | | |

### Embedding variants

| Variant | Objective | RMSE (target) | Spearman | Analogy acc. | WordSim-353 ρ |
|---|---|---|---|---|---|
| wiki-only | CBOW | | | | |
| wiki-only | Skip-gram | | | | |
| hn-only | CBOW | | | | |
| hn-only | Skip-gram | | | | |
| fine-tuned | CBOW | | | | |
| fine-tuned | Skip-gram | | | | |

### Ablations

| Dropped modality | RMSE (target) | Change vs full |
|---|---|---|
| None (full model) | | |
| Title | | |
| Author | | |
| Domain | | |
| Temporal | | |

### Implementation check against gensim

| Corpus | Metric | This implementation | gensim |
|---|---|---|---|
| text8 | Analogy accuracy | | |
| text8 | WordSim-353 ρ | | |

## Getting started

Requires [`uv`](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`
and `uv` installs it if it is missing.

```bash
uv sync            # core install: numpy, pandas, pytest, ruff
make check         # exactly what CI runs: ruff check, ruff format --check, pytest
```

Heavy dependencies are optional extras, so the scaffold and the implemented modules
install and test without a 2 GB torch download.

| Extra | Contents | Needed from |
|---|---|---|
| `data` | duckdb, pyarrow, matplotlib, jupyter tooling | Phase 1 |
| `train` | torch, scikit-learn, gensim | Phase 2 |
| `serve` | fastapi, pydantic, uvicorn | Phase 6 |

### Reproducing Phase 1

```bash
uv sync --extra data
make ingest        # measures the projected transfer, pulls 237 monthly shards, writes
                   # data/stories.parquet, prints the real row count
make clean-shards  # safe to run once ingest has reported its row count
make notebook      # re-executes notebooks/01-eda.ipynb, regenerating every plot
```

`make ingest` took roughly 40 minutes on a home connection and is resumable: it skips
any shard already on disk. The monthly shards under `data/shards/` are scratch and nothing
after Phase 1 reads them. `make help` lists the rest.

Gate 3 reads the live Hacker News API, so re-running the notebook produces slightly
different numbers from the ones quoted here. The conclusion is not close to the boundary.

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

## Phases

| Phase | Deliverable | State |
|---|---|---|
| 0 | Repo, README, scaffold, CI | Done |
| 1 | Ingest and EDA, three gates, leak audit | Done |
| 2 | Baseline ladder and walk-forward harness | Not started |
| 3 | CBOW and Skip-gram on text8, validated against gensim, then the Wikipedia subset | Not started |
| 4 | HN fine-tuning, three-variant comparison | Not started |
| 5 | Early, late and hybrid fusion, plus ablations | Not started |
| 6 | FastAPI and Docker | Not started |
| 7 | Results write-up | Not started |

## Licence

MIT for the code. The dataset is `odc-by` and is not redistributed here.
