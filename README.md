# Hacker News upvote prediction

Predicts the score a Hacker News post will get, using only what is knowable at the
moment it is submitted: the title, the account posting it, the linked domain, and the
timestamp.

Word embeddings are trained from scratch in PyTorch, both the CBOW and the Skip-gram
objective, with negative sampling. Three fusion architectures combine the title vector
with the author, domain and time signals. Validation is walk-forward on the time axis.

**Status: Phase 0.** Repository scaffold, this README, and CI. No data has been
downloaded and no model has been trained. Every results table below is empty and stays
empty until the phase that fills it. The two modules that are fully implemented are
`features/schema.py` and `target/normalise.py`, because they encode the two rules the
rest of the project depends on.

## Contents

- [What the model is allowed to see](#what-the-model-is-allowed-to-see)
- [Leakage and temporal validation](#leakage-and-temporal-validation)
- [The prediction target](#the-prediction-target)
- [Embeddings](#embeddings)
- [Title vector](#title-vector)
- [Fusion architectures](#fusion-architectures)
- [Baseline ladder](#baseline-ladder)
- [Evaluation](#evaluation)
- [Results](#results)
- [Hardware](#hardware)
- [Data](#data)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Phases](#phases)
- [References](#references)

## What the model is allowed to see

| Column | Meaning | Why it is legal |
|---|---|---|
| `title` | The submitted headline | Written by the submitter before the post exists |
| `by` | The submitting account | Known at submission |
| `url` | The linked address, null for a text post | Chosen by the submitter |
| `time` | Submission timestamp, unix seconds in the dump | The submission event itself |

That is the entire allowlist. It lives in
[`src/hn_upvotes/features/schema.py`](src/hn_upvotes/features/schema.py) and every
feature builder goes through it.

## Leakage and temporal validation

This is the part worth scrutinising, so it comes before the modelling.

### Post-hoc columns

Three columns in the source data describe what happened after a post went live. None of
them can be a feature.

| Column | What it is | Why it leaks |
|---|---|---|
| `score` | The point total | The prediction target. Using it is circular |
| `descendants` | Total comment count | Comments accumulate alongside votes. A post with 200 comments almost certainly scored well, so this column is close to a copy of the answer |
| `kids` | Ids of the direct child comments | The same information as `descendants`, in list form. Easy to miss, because it looks structural rather than numeric |

`descendants` is the one that catches people. It reads like metadata about the post
rather than about its reception, and a model given it will post excellent offline
numbers that collapse in production, because at submission time it is always zero.

### The allowlist

Enforcement is an allowlist, not a blocklist. A column nobody has explicitly approved is
rejected, so adding a new column to the dataset cannot silently promote itself into a
feature.

```python
from hn_upvotes.features.schema import validate_feature_set

# Returns the validated set.
validate_feature_set(["title", "by", "url", "time"])

# Raises LeakedFeatureError.
validate_feature_set(["title", "descendants"])

# Raises UnknownFeatureError. `text` is genuinely available at submission time and is
# still rejected, because nobody has approved it. Adding it is a deliberate edit.
validate_feature_set(["title", "text"])
```

The guard is a test, not a convention.
[`tests/test_feature_schema.py`](tests/test_feature_schema.py) restates the three banned
names independently of the module it is testing, so moving a column from banned to
allowed in `schema.py` turns the suite red rather than taking the test with it.

### Temporal splits

There are no random splits in this project. A random split trains the model on posts
submitted after the ones it is tested on, which is information no submission-time
predictor could have.

Validation is walk-forward. Train on a window, test on the window immediately after,
roll forward, repeat. Drift inside any one fold is small, and the sequence of scores
across folds shows whether the model degrades as it ages. A single early/late cut gives
one number and hides that.

### Rolling statistics

The author and domain track records are expanding-window statistics over strictly
earlier posts. A post from March never sees a statistic computed with April data.

Those statistics also carry the same settling lag as the target baseline, described
below. An author's post from two hours ago has not finished scoring, so it does not yet
count towards their track record.

## The prediction target

The model trains on `log1p(score)` normalised against a trailing baseline, and reports
back in raw score terms.

### Why not raw score

Hacker News has grown. The claim is that the same quality of post earns more points now
than it did years ago, so a model trained on old data would systematically under-predict
new posts. It would judge the post correctly while the ruler moved underneath it.

**This claim is asserted, not measured.** Phase 1 plots median score per year. If the
line is flat, all of this comes out and the model predicts plain `log1p(score)`. If it
climbs, the mechanism is justified. The plot goes in this README either way, because it
is the evidence.

The obvious fix does not work. Adding `year` as a feature fails under a temporal split:
training stops years before the test period, so a tree has no branch for a year it never
saw, and every test-period post falls into the last training bucket and gets that era's
numbers.

So the question changes instead of the features. Rather than predicting how many points
a post gets, predict how it did compared to posts from around the same time.

### The transform

```
target = (log1p(score) - baseline_centre) / baseline_spread
score  = expm1(target * baseline_spread + baseline_centre)
```

Both statistics come from a trailing window, so both are available at inference time and
the inverse works in production, not only in a backtest.

An illustrative worked example, with invented baselines. These numbers demonstrate the
mechanism and are not measurements.

| Post | Raw score | `log1p` | Baseline centre | Baseline spread | Target |
|---|---|---|---|---|---|
| Strong post, earlier era | 50 | 3.93 | 1.8 | 1.2 | **1.78** |
| Its later-era equivalent | 183 | 5.21 | 2.9 | 1.3 | **1.78** |
| A stronger later-era post | 300 | 5.71 | 2.9 | 1.3 | **2.16** |

The first two rows differ by 3.7x in raw score and are identical in target. The third is
a better post and the target says so.

### Why a z-score

It is chosen for invariance, not for probability. An affine transform preserves rank
exactly, removes location drift by subtracting, and removes scale drift by dividing. No
distributional assumption is needed for that job.

A z-score only becomes a probability statement under approximate normality, which is
doubtful here. A large share of HN posts score 1 or 2, and that floor spike survives a
log transform. Treat any percentile conversion as indicative.

Percentile rank within the trailing window is the distribution-free alternative, and is
uniform on `[0,1]` by construction. It is rejected because it flattens the tail, where a
500-point and a 3000-point post both sit near the 99.9th percentile, and the tail is the
interesting region. Percentile is reported as a secondary readout instead.

Separately, squared error is maximum likelihood under Gaussian *residuals*, not a
Gaussian target. If the Phase 2 residual plot comes out heavy tailed, the loss switches
to Huber.

### The window and the settling lag

The baseline for a row uses rows in `[t - lag - window, t - lag)`. Two rules, both load
bearing.

**Strictly earlier.** The upper bound is exclusive, so a row can never contribute to its
own baseline and neither can anything submitted at the same instant or later. Using the
post's own calendar month would include posts submitted after it, which nobody could
know at submission time.

**Settling lag.** A post from three hours ago is still gaining votes. Letting it into the
baseline drags the centre down with scores that are not final.

Defaults are a 30 day window and a 48 hour lag. Both are configuration, not constants.
Phase 1 measures how long a score actually takes to settle and sets the real lag from
that measurement.

```python
from datetime import timedelta
from hn_upvotes.target.normalise import BaselineConfig, compute_trailing_baseline, forward

config = BaselineConfig(window=timedelta(days=30), lag=timedelta(hours=48))
baseline = compute_trailing_baseline(frame["time"], frame["score"], config)
targets = forward(frame["score"], baseline)
```

### Removing the mechanism

Phase 1 may decide this is more machinery than the data justifies, so it is built to come
out cleanly.

| Setting | Effect |
|---|---|
| `scale=False` | Subtract the trailing centre, do not divide. Use if only the centre drifts |
| `centre="median"`, `spread="iqr"` | Robust statistics, for the heavy tail |
| `centre="zero"`, `scale=False` | The transform reduces to plain `log1p(score)` |

If only the centre moves, subtracting a trailing median is simpler, more robust, and more
readable, because `exp(target)` then means "this post did N times the typical post".

[`tests/test_target_normalise.py`](tests/test_target_normalise.py) covers the round trip
across scores from 0 to 3000, and proves the strictly-earlier rule by recomputing the
baseline on every prefix of a synthetic frame and checking that no answer moves when
later rows are added.

## Embeddings

### Terminology

CBOW and Skip-gram are training **objectives**, not embeddings. You train a model on a
fill-in-the-blank task, throw the task away, and keep the input weight matrix, one row
per word. That matrix is the embedding. The phrase "CBOW embeddings" does not appear in
this project's writing.

Both objectives learn two matrices, centre and context. Convention keeps the first.
Averaging the two is a cheap variant and is tested.

- `cbow.py` predicts the centre word from the averaged context vectors
- `skipgram.py` predicts context words from the centre word

Skip-gram generates one training pair per context position instead of one per window, so
it sees more updates per token, trains slower, and generally does better on rare words.

### Negative sampling

A full softmax over a 100k vocabulary is not viable, so both objectives use negative
sampling. Hyperparameters follow Mikolov et al. 2013.

| Setting | Value | Source |
|---|---|---|
| Negative samples `k`, text8 and HN titles | 15 | Paper recommends 5 to 20 for small corpora |
| Negative samples `k`, Wikipedia subset | 5 | Paper recommends 2 to 5 for large corpora |
| Noise distribution | Unigram counts raised to 0.75 | The paper's tuned value, best of the distributions tried |
| Frequent-word subsampling | `t = 1e-5` | The paper's rule, `P(keep) = min(1, sqrt(t/f))` |
| Context window | Dynamic, sampled from 1 to 5 | Weights nearer context words more heavily at no extra cost |

The 0.75 power flattens the unigram distribution so rare words turn up as negatives more
often than their raw frequency would allow. `k` is tunable and these are starting points.

### Development and validation

Development runs on `text8`, 100 MB, which trains in minutes and makes the implementation
debuggable. Correctness is checked against gensim on the same corpus before anything
scales up. Matching gensim within noise on text8 is the gate for moving to the Wikipedia
subset.

Speed expectations are set honestly: a from-scratch PyTorch SGNS runs one to two orders
of magnitude slower than gensim's Cython. The subset size is chosen from a measured
tokens-per-second figure, not a guess.

Intrinsic evaluation uses the Google analogy set, WordSim-353, and nearest-neighbour spot
checks on HN vocabulary (`rust`, `yc`, `llm`). Coverage is reported alongside accuracy,
because a small vocabulary can post a flattering score on the few questions it can
answer. Intrinsic scores are a sanity check. The result is the downstream task.

### Three variants

| Variant | Initialisation | Trained on |
|---|---|---|
| wiki-only | Random | Wikipedia subset |
| hn-only | Random | HN titles |
| fine-tuned | Wikipedia vectors | HN titles at a lower learning rate |

Words appearing in HN but not in Wikipedia get random initialisation before fine-tuning.

## Title vector

Mean pooling by default. SIF (smooth inverse frequency weighting plus removal of the
first principal component, Arora et al. 2017) is the upgrade. Both sit behind one
interface in [`features/pooling.py`](src/hn_upvotes/features/pooling.py), so the fusion
models do not know which is active.

SIF is fitted on training rows only. Fitting the word frequencies or the principal
component on the full frame would leak test-period vocabulary statistics backwards.

## Fusion architectures

Four inputs: the pooled title vector, the author, the domain, and the temporal features.
Author and domain are high cardinality, so they get learned embedding tables, with values
below a minimum post count bucketed to a shared out-of-vocabulary row.

| Architecture | Structure | Learns interactions | Notes |
|---|---|---|---|
| Early | Concatenate all four, one MLP to a scalar | Yes | A weak modality can drag the shared representation |
| Late | One tower per modality to its own scalar, learned combination | No | Interpretable per modality, degrades gracefully when one is missing |
| Hybrid | Encode each modality, concatenate the encodings, joint head | Yes | What production ranking systems usually do |

Late fusion is the one that handles a text post cleanly: no URL means the domain tower
drops out and the weights renormalise.

Ablations drop each modality in turn, so the results section can state what each one is
worth rather than asserting it.

## Baseline ladder

Built before any neural network, so there is a real bar to clear. Every rung predicts in
normalised target space, the same space the fusion models train in.

1. Predict the trailing baseline, which in normalised space is just zero
2. Author historical mean, time aware
3. Domain historical mean, time aware
4. TF-IDF plus Ridge
5. TF-IDF plus gradient boosting

Rung 5 sees the same four modalities as the fusion models, without learned
representations, so it is the honest comparison. If no fusion model beats it, this README
says so. An honest negative result reads better than a suspiciously good number.

## Evaluation

Primary metrics are RMSE and MAE on the normalised target, because that is the space the
model trains in. A normalised error is hard to read alone, so three more are reported:

- **RMSE on `log1p(score)`**, after mapping predictions back through the test period's
  trailing baseline. Quotable in real score terms.
- **Spearman correlation on raw score.** Ranking quality is what a submission-time
  predictor is for, and Spearman is invariant to the whole normalisation, so it is the
  metric the target transform cannot flatter.
- **Precision@100.** Of the top 100 posts the model predicts, how many really landed
  high.

Every number is quoted as mean plus or minus standard deviation across five seeds. A
single run is not evidence. Where two models differ by less than the seed noise, they are
reported as indistinguishable rather than ranked.

## Results

**Nothing here has been measured.** These tables are the shape of the output, published
empty so the reporting format is fixed before any number exists. No figures are filled in
until the phase that produces them.

### Phase 1 gates

| Question | Method | Result |
|---|---|---|
| Does median score drift by year? | Median `log1p(score)` per year | Not yet measured |
| Does the spread drift too? | Std and IQR of `log1p(score)` per year | Not yet measured |
| How long does a score take to settle? | Score against age, on recent posts | Not yet measured |
| Usable story rows after filtering | Count after ingest | Not yet measured |

The third row sets the settling lag. The first two decide whether the target transform
survives at all, and whether it needs the divide-by-spread step.

### Model comparison

Mean plus or minus standard deviation across five seeds, aggregated over walk-forward
folds.

| Model | RMSE (normalised) | MAE (normalised) | RMSE (`log1p` score) | Spearman | P@100 |
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

Downstream, on the best fusion architecture.

| Variant | Objective | RMSE (normalised) | Spearman | Analogy acc. | WordSim-353 ρ |
|---|---|---|---|---|---|
| wiki-only | CBOW | | | | |
| wiki-only | Skip-gram | | | | |
| hn-only | CBOW | | | | |
| hn-only | Skip-gram | | | | |
| fine-tuned | CBOW | | | | |
| fine-tuned | Skip-gram | | | | |

### Ablations

| Dropped modality | RMSE (normalised) | Change vs full |
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

## Hardware

Everything runs on an Apple M2 with 24 GB of unified memory. PyTorch uses the MPS
backend. There is no CUDA and no cloud GPU.

That has consequences worth stating rather than hiding:

- Full English Wikipedia is out of scope. The embedding corpus is a subset, and its size
  is set from a measured throughput number in Phase 3.
- Corpus size is a documented tradeoff, not a shortcut. Where a result would plausibly
  change with a bigger corpus, the results section says so.
- The training Dockerfile targets CPU or CUDA. MPS needs the host's Metal stack and is
  not available inside a container, so local training runs outside Docker and the image
  exists for reproducibility elsewhere.

## Data

Source is [`open-index/hacker-news`](https://huggingface.co/datasets/open-index/hacker-news)
on Hugging Face. Monthly Parquet files, zstd compressed, licence `odc-by`. Queried with
DuckDB directly over the Parquet files. No BigQuery and no billing.

The dump holds 49.1M rows covering every item type, not just stories. Filtering to
stories with a title and a score cuts that substantially. The real usable row count goes
in the Phase 1 gates table after ingest, rather than being quoted in advance.

Two schema notes. `type` is stored as `int8`, so the story encoding is decoded during
ingest and the mapping recorded. `words` is a pre-tokenised column, and Phase 1 compares
it against this project's own tokeniser on a sample rather than trusting either blindly.

## Repository layout

```
HN-prediction/
├── README.md
├── pyproject.toml            Python pinned to 3.12
├── Makefile
├── configs/                  default.yaml, embeddings_text8.yaml, embeddings_wiki.yaml
├── docker/                   train.Dockerfile, serve.Dockerfile
├── notebooks/
├── src/hn_upvotes/
│   ├── data/                 ingest.py, preprocess.py, splits.py
│   ├── embeddings/           cbow.py, skipgram.py, train.py, evaluate.py
│   ├── features/             schema.py, pooling.py, author.py, domain.py, temporal.py
│   ├── target/               normalise.py
│   ├── models/               baselines.py, early_fusion.py, late_fusion.py, hybrid.py
│   ├── training/             loop.py, metrics.py
│   └── serving/              app.py, schemas.py
└── tests/
```

Everything outside `features/schema.py` and `target/normalise.py` is a stub in Phase 0.
Each stub carries its real type-annotated signature and a docstring saying what it will
do, so the wiring is readable without any implementation present.

## Getting started

Requires [`uv`](https://docs.astral.sh/uv/). Python 3.12 is pinned in
`.python-version` and `uv` installs it if it is missing.

```bash
uv sync                                    # core install: numpy, pandas, pytest, ruff
uv run pytest -v
uv run ruff check
uv run python -c "import hn_upvotes"
```

Heavy dependencies are optional extras, so the scaffold and the two implemented modules
install and test without a 2 GB torch download.

| Extra | Contents | Needed from |
|---|---|---|
| `data` | duckdb, pyarrow | Phase 1 |
| `train` | torch, scikit-learn, gensim | Phase 2 |
| `serve` | fastapi, pydantic, uvicorn | Phase 6 |

```bash
uv sync --extra data --extra train --extra serve   # or: make setup-all
```

`make help` lists the rest.

## Phases

| Phase | Deliverable | State |
|---|---|---|
| 0 | Repo, README, scaffold, CI | Done |
| 1 | Ingest and EDA. Three gates: does the median drift by year, does the spread drift, how long does a score take to settle | Not started |
| 2 | Baseline ladder and walk-forward harness | Not started |
| 3 | CBOW and Skip-gram on text8, validated against gensim, then the Wikipedia subset | Not started |
| 4 | HN fine-tuning, three-variant comparison | Not started |
| 5 | Early, late and hybrid fusion, plus ablations | Not started |
| 6 | FastAPI and Docker | Not started |
| 7 | Results write-up | Not started |

## References

- Mikolov, Sutskever, Chen, Corrado, Dean (2013). *Distributed Representations of Words
  and Phrases and their Compositionality.* NeurIPS.
  [arXiv:1310.4546](https://arxiv.org/abs/1310.4546). Source of the negative sampling
  count, the 0.75 noise power, and the `t = 1e-5` subsampling rule.
- Mikolov, Chen, Corrado, Dean (2013). *Efficient Estimation of Word Representations in
  Vector Space.* [arXiv:1301.3781](https://arxiv.org/abs/1301.3781). The CBOW and
  Skip-gram objectives, and the analogy evaluation set.
- Arora, Liang, Ma (2017). *A Simple but Tough-to-Beat Baseline for Sentence Embeddings.*
  ICLR. [OpenReview](https://openreview.net/forum?id=SyK00v5xx). The SIF pooling used in
  `features/pooling.py`.
- Finkelstein et al. (2002). *Placing Search in Context: The Concept Revisited.* ACM TOIS.
  The WordSim-353 similarity ratings.

## Licence

MIT for the code. The dataset is `odc-by` and is not redistributed here.
