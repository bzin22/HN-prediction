# Word2vec from scratch: the implementation

Why the code is shaped the way it is. [`design.md`](design.md) holds the project's wider
reasoning and the embedding hyperparameter table; this note covers the implementation itself
and the two measurements that could have invalidated it.

Nothing here has been trained on a real corpus yet. Every number below comes from a synthetic
corpus or a microbenchmark, and each says which.

## Contents

- [Terminology](#terminology)
- [Two matrices, one of which survives](#two-matrices-one-of-which-survives)
- [Why negative sampling](#why-negative-sampling)
- [The two objectives differ by one line](#the-two-objectives-differ-by-one-line)
- [Why plain SGD](#why-plain-sgd)
- [The learning rate is per example, and the batch scales it](#the-learning-rate-is-per-example-and-the-batch-scales-it)
- [Subsampling, and two ways gensim differs](#subsampling-and-two-ways-gensim-differs)
- [Risk 1: sparse gradients](#risk-1-sparse-gradients)
- [Risk 2: the GPU or the Python](#risk-2-the-gpu-or-the-python)
- [What the measurements changed](#what-the-measurements-changed)
- [The overnight chain](#the-overnight-chain)
- [Sizing the Wikipedia stage](#sizing-the-wikipedia-stage)
- [The gate](#the-gate)
- [What is not proved yet](#what-is-not-proved-yet)

## Terminology

CBOW and Skip-gram are training **objectives**. You train a model on a fill-in-the-blank
task, throw the task away, and keep the input weight matrix, one row per word. That matrix is
the embedding. "CBOW embeddings" is wrong and does not appear in this project.

## Two matrices, one of which survives

Both objectives learn two matrices, each vocabulary by dimension:

| Matrix | Role | Fate |
|---|---|---|
| **Input** | one row per word | **this is the embedding**, and what everything downstream consumes |
| **Output** | same size, used only for scoring | discarded when training ends |

At 100,000 words by 300 dimensions each matrix is 120 MB of float32, so 240 MB for the pair.
Plain SGD carries no optimiser state, so that is the whole memory cost. SparseAdam would have
added 480 MB and Adagrad 240 MB.

The two matrices, the negative sampler and the loss live in one module,
[`negative_sampling.py`](../src/hn_upvotes/embeddings/negative_sampling.py). `cbow.py` and
`skipgram.py` hold only a `forward`. That is not tidiness. The project's stated experiment is
CBOW against Skip-gram, and a sampler duplicated across two files is how the two drift apart
and the comparison stops measuring the objective.
`test_the_two_objectives_share_one_sampler_and_one_loss` fails if either class grows a second
method.

## Why negative sampling

A full softmax over 100,000 words asks "which of these 100,000 words is it", and answering
costs a dot product against every row. That denominator is the entire cost of the model.

Negative sampling replaces it with 16 yes/no questions: is this the real word, and are these
15 drawn words not it. So the cost per example is 16 dot products rather than 100,000, and it
stops depending on vocabulary size at all.

The negatives are drawn from the unigram distribution raised to 0.75, which is the paper's
tuned value. It flattens the distribution: where one word is 100 times as frequent as another,
`100 ** 0.75 = 31.6`, so it is drawn about 32 times as often rather than 100. Rare words
appear as negatives more than their raw frequency allows.

Worked, on the four-word fixture in the tests. Counts 10,000 / 1,000 / 100 / 10 give weights
1000.000 / 177.828 / 31.623 / 5.623, total 1215.074, so the shares are:

| Word | Raw frequency | 0.75 power | Uniform |
|---|---|---|---|
| 10,000 | 0.90009 | **0.82299** | 0.25 |
| 1,000 | 0.09001 | **0.14635** | 0.25 |
| 100 | 0.00900 | **0.02602** | 0.25 |
| 10 | 0.00090 | **0.00463** | 0.25 |

Three visibly different columns, which is why the test can tell a correct sampler from the two
wrong ones rather than only checking that the numbers sum to 1.

The loss is binary cross-entropy **with logits** on the dot products, so the log and the
sigmoid stay fused and a large negative logit cannot underflow before the log sees it. The
per-example loss is `-log sigmoid(positive) - sum_j log sigmoid(-negative_j)`.

One consequence is a free test. The output matrix starts at zeros, matching gensim, so every
logit starts at exactly 0, every sigmoid at 0.5, and the first loss is `(1 + k) * log 2`. At
`k = 15` that is `16 * 0.693147 = 11.090`. A factor-of-two slip or a dropped negative term
moves that number and nothing else in the suite would notice.

## The two objectives differ by one line

CBOW averages the context vectors and scores the average against the centre word. Skip-gram
scores the centre word against each context word separately. In code that is:

```python
# cbow.py
hidden = masked_average(self.input_matrix(context_ids), context_mask)
# skipgram.py
hidden = self.input_matrix(centre_ids)
```

Everything after that line is shared.

The consequence is throughput, not just shape. With a window of 5, CBOW produces one training
example per position and Skip-gram up to ten. Measured on the synthetic calibration corpus at
`t = 1e-5`: **1.22 training examples per corpus token for Skip-gram against 0.33 for CBOW**, a
factor of 3.7. So Skip-gram is the slower one and is generally better on rare words.

The masked average is the one thing that is only CBOW's, and it is the sharp edge. The dynamic
window makes contexts ragged, and padding uses word id 0, which is the unknown token and **a
real row of the matrix**. Getting the mask wrong does not crash. It quietly averages the
unknown vector into every short context and divides by the wrong count. Two real vectors `a`
and `b` in a width-10 row must give `(a + b) / 2`; averaging the padded row gives
`(a + b + 8 * pad) / 10`, which is a different vector even when `pad` is zero, because the
divisor is 10 instead of 2.

## Why plain SGD

Both objectives, no momentum, nothing adaptive, learning rate decaying linearly. Two reasons.

The gate for this implementation is matching gensim on text8, and gensim uses SGD decaying
linearly to `min_alpha`. Using a different optimiser means a gap could be a bug or could be
the optimiser, and that is two things to debug at once.

Adam and AdamW are ruled out at the API level rather than by preference: PyTorch raises
`Adam does not support sparse gradients, please consider SparseAdam instead`. Embedding
training touches about 16 rows of 100,000 per step, so sparse gradients are the whole point.

**The optimiser grid is deferred, not cancelled.** If it is run, run it as a full 2x2, CBOW
and Skip-gram against SGD and Adagrad. Pairing each objective with a different optimiser
confounds the CBOW-against-Skip-gram comparison: if Skip-gram wins you cannot say whether the
objective or the optimiser did it. Note for whoever runs it, Adagrad's accumulated gradient
only grows, so its rate decays monotonically and can stall on a long run. It may win on text8
and lose on Wikipedia.

## The learning rate is per example, and the batch scales it

This is the one piece of arithmetic in the harness that looks wrong and is not.

`forward` returns the **mean** loss over the batch. gensim updates one example at a time. So a
word appearing once in a batch of 1024 moves by `rate / 1024` times its gradient, where
gensim's same word moves by `alpha` times its gradient. Left alone, our steps would be a
thousand times too small.

`SGNSConfig.learning_rate` is therefore the **per-example** rate, gensim's `alpha` and its
default of 0.025, and the optimiser is given `learning_rate * batch_size`:

```
0.025 * 1024 = 25.6
```

The number looks alarming and is not. The quantity reaching a single row of the matrix is
still 0.025 times a gradient. This is the linear scaling rule, and
[`optimiser_learning_rate`](../src/hn_upvotes/embeddings/train.py) is the one place it happens.

**The scaffold's default was 2.5e-3, ten times lower and on the scale of an Adam rate rather
than an SGD one.** It is now 0.025, decaying linearly to 0.0001, both of which are gensim's
defaults. That is a change from what the scaffold shipped and it is deliberate: the decision
recorded for this phase was to mirror gensim.

What batching genuinely does change is that updates inside one batch do not see each other,
where gensim's sequential ones do. That is inherent to batching and is a reason to expect a
small gap against gensim, not a reason to leave the step size wrong.

## Subsampling, and two ways gensim differs

`P(keep) = min(1, sqrt(t / f))` at the paper's `t = 1e-5`.

Worked on the test fixture: counts 500,000 / 5,000 / 500 / 50, so 505,550 tokens. Each keep
probability is ten times the one before, because the frequency drops 100x and the rule takes a
square root:

| Count | Frequency | `sqrt(t/f)` | Tokens kept |
|---|---|---|---|
| 500,000 | 0.989022 | 0.003180 | 1589.9 |
| 5,000 | 0.009890 | 0.031798 | 159.0 |
| 500 | 0.000989 | 0.100553 | 50.3 |
| 50 | 0.000099 | 0.317978 | 15.9 |

Total 1815.1 of 505,550, so **0.359%**. That share is tiny because the fixture is deliberately
one dominant word. Real text spreads its mass, and the measured retention on the HN corpus at
the same threshold is 31.3%.

**Two things the gensim comparison has to account for, and only the first was known going in.**

1. **gensim's `sample` default is `1e-3`, not `1e-5`.** It must be set explicitly or the gate
   compares a model keeping 82% of its tokens against one keeping 31% and blames the gap on
   us. `train_gensim` passes it, and a test asserts it.
2. **gensim's subsampling formula is not the paper's.** gensim uses `sqrt(t/f) + t/f` where
   the paper uses `sqrt(t/f)`. The two agree to a rounding error on very frequent words and
   diverge in the middle: at `f = 4e-5` and `t = 1e-5` the paper keeps `sqrt(0.25) = 0.50` and
   gensim keeps `0.50 + 0.25 = 0.75`. This cannot be passed away as a parameter. It is a known
   residual difference in the gate, and it favours gensim slightly by giving it more tokens.

`t` is a bigger lever here than in the paper, because 69% of a corpus 26x smaller is discarded
and the paper itself calls the formula "chosen heuristically". Worth a sweep on text8 with the
dimension ablation. Deferred.

Out-of-vocabulary tokens are **dropped from the training stream** rather than mapped to the
unknown token, which is what gensim does: the window then closes over the gap so the words
either side become neighbours. The unknown row stays in the matrix so serving has something to
return for a word it has never seen, but it is never trained and keeps its initialisation.
Pooling should skip it rather than average it in.

## Risk 1: sparse gradients

**Answer: they work, on both CPU and MPS.** torch 2.13.0, `nn.Embedding(sparse=True)` at
100,000 by 300. `backward` produces a genuine sparse gradient with only the touched rows
materialised, and `torch.optim.SGD` steps on it. The dense-gradient disaster case is off the
table.

The saving is real but smaller than the row count suggests. A batch of 1024 with `k = 15`
touches 16,384 rows, so the sparse gradient holds 19.7 MB of values against 120 MB dense: 6x,
not 1000x. At a smaller batch the ratio improves.

Reproduce with `make throughput`.

## Risk 2: the GPU or the Python

The hypothesis was that the Python feeder would be the bottleneck, since word2vec's cost is
usually windowing, subsampling and drawing negatives rather than the matrix maths. **Measured,
it is neither. The bottleneck is the embedding backward, and MPS is worse at it than CPU.**

Skip-gram, dimension 300, `k = 15`, batch 1024, 2M-token synthetic Zipfian corpus, in-vocabulary
corpus tokens per second:

| | Skip-gram | CBOW |
|---|---|---|
| Feeder alone, no model | **963,848** | **851,495** |
| CPU, sparse gradients | 81,603 | 181,487 |
| CPU, dense gradients | 46,325 | 135,986 |
| MPS, sparse gradients | 24,252 | 69,072 |
| MPS, dense gradients | 36,130 | 110,614 |

The feeder can move tokens 12x faster than the fastest training configuration consumes them,
so it is not the constraint. Per-stage profiling puts `backward` at 45% of a CPU step and 61%
to 92% of an MPS step, rising with batch size.

At the full 100,000-word vocabulary, in training pairs per second:

| Device | Sparse | Dense |
|---|---|---|
| **CPU** | **105,142** | 34,469 |
| MPS | 24,383 | 38,335 |

Two things fall out. **CPU with sparse gradients is 4.3x faster than MPS with sparse
gradients**, and 2.7x faster than the best MPS option. And MPS is the one place where sparse
loses to dense, because its sparse backward is slow enough that its own dense path wins.

The reason is that a step touches about 16,000 rows of 300 floats, which is far too little
arithmetic to pay for the kernel launches. Bigger batches do not rescue it: MPS improves from
27,150 to 42,436 pairs/s between batch 1024 and 16384 while CPU sits at 105,743 to 120,962.
This is the same reason gensim is CPU threads with no GPU at all.

## What the measurements changed

**`select_device()` now returns CPU.** The scaffold assumed MPS and the docstring said "pick
MPS when available"; the measurement says CPU is 4.3x faster, so defaulting to MPS would have
been a bug with a comment defending it. MPS stays reachable through `SGNSConfig.device="mps"`,
because a larger dimension or batch could move the balance and that should be re-measured
rather than assumed.

The other change is the learning rate, [above](#the-learning-rate-is-per-example-and-the-batch-scales-it).

## The overnight chain

[`chain.py`](../src/hn_upvotes/embeddings/chain.py) runs four stages in one process, in order.
`make chain-dry-run` walks all four on synthetic corpora in about 7 seconds.

| Stage | Produces | Depends on |
|---|---|---|
| 0. `gate` | nothing kept | text8 and gensim |
| 1. `wiki-only` | the Wikipedia variant | nothing |
| 2. `fine-tuned` | the fine-tuned variant | stage 1 |
| 3. `hn-only` | the HN variant | nothing |

`hn-only` runs **last on purpose**. It needs nothing from the stages before it, so a Wikipedia
failure costs one variant rather than the night. Stage 2 loads stage 1's vectors and carries
on over HN titles and bodies at a tenth of the learning rate, so it adjusts Wikipedia's
structure rather than overwriting it with a much smaller corpus. Words HN has and Wikipedia
does not start random.

Unattended operation needs six things beyond training, and each is tested:

- **A checkpoint after every epoch**, `{stage}-epoch{n}.npz`. A checkpoint holds **both**
  matrices, not just the embedding: resume from the input matrix alone and the model relearns
  every score from zero, throwing away most of the epoch it was resuming to save.
- **`--resume`** picks the highest epoch number for each stage, chosen by the name rather than
  the modification time so a restored file cannot look newest.
- **Failure isolation.** A stage that raises is recorded with its traceback and the next stage
  still runs. The gate is the deliberate exception.
- **A wall-clock budget per stage.** It stops between batches, saves, and returns with
  `cut_short` set. A cut-short epoch banks its progress under the *previous* epoch number so a
  resume repeats that epoch rather than skipping the part it never did, and its loss is kept
  out of the per-epoch list so the count cannot drift. That drift was a real bug found here: a
  resumed stage reported 4 epochs completed out of 3.
- **One JSON manifest**, rewritten atomically after every state change, holding the effective
  settings, each stage's status, throughput, per-epoch losses, and any failure with its
  traceback. A night is readable from one file.
- **Detachment.** `--detach` relaunches in a new session so closing the terminal hangs up the
  shell and not the run.

A run that was cut short reports `partial`, never `completed`. Reporting otherwise is exactly
what the flag exists to prevent.

## Sizing the Wikipedia stage

The ceiling is two hours and the subset is computed to fit it, rather than picked and then
timed. From the measured 138,362 tokens/s at `k = 5`, times the 0.96 large-vocabulary factor,
over a 7,200 second ceiling, divided by 5 epochs because every epoch rereads the whole subset:

```
138,362 x 0.96 x 7,200 / 5 = 191,271,628 tokens
```

At 5.9 bytes per token, measured on text8, that is **about 1.13 GB**, comfortably above the
300 MB floor below which the variant would be barely larger than text8 and the stage would
have lost its point. If a future measurement puts it under that floor the chain stops and
reports the number with the levers rather than shrinking the corpus quietly.

One derived estimate did not survive measurement. `k = 5` costs 6 dot products per pair against
`k = 15`'s 16, which predicts 2.7x the throughput. **Measured, it is 1.80x**: 138,362 against
77,019 tokens/s. The difference is fixed per-batch overhead that does not scale with `k`.

## The gate

Stage 0 trains our implementation and gensim on the same corpus with the same settings and
compares them. A failure aborts the whole chain. That is the single most valuable thing in the
design: without it a bug costs a night and shows up in the morning, and with it the chain
stops in minutes and the machine sits idle instead, which is far cheaper.

It **fails closed**. A gate that could not run has not passed.

The gate compares **task scores**, with the task supplied by the caller so the same code serves
the real run and the dry run: the Google analogy set and WordSim-353 on text8, and a synthetic
corpus's own planted answer in the dry run. Our score must be within 20% of gensim's, and our
nearest-neighbour lists must overlap gensim's by at least 0.15.

Two things were learned building it, both worth recording because both look like reasonable
designs and neither works.

**Comparing our word-pair similarities against gensim's by rank correlation is not a gate.**
On the synthetic topic corpus a correct implementation scores 0.012 on that metric while
scoring 1.00 on the actual task. The reason is that 97.5% of randomly sampled pairs are
cross-topic, where the corpus says nothing about how they should rank, so the metric is
dominated by pairs whose true answer is undefined. It is recorded as rejected in
`evaluate.py` so it is not tried again.

**The planted-synonym corpus is the wrong corpus to gate on.** gensim recovers 0 of its 6
pairs while this implementation recovers all 6, because 36 word types over 7,200 tokens is too
little for gensim's vectors to pull apart. Comparing against a reference that has not
converged tells you nothing. The dry run therefore gates on a 1,000-type, 192,000-token topic
corpus where **both** implementations reach a topic purity of 1.00.

The abort path is tested by deliberately breaking the implementation: replacing the matrix with
noise gives topic purity 0.045 against gensim's 1.000, a 95.5% shortfall, and neighbour overlap
0.021 against the 0.15 floor. The chain aborts, the three later stages are skipped, and no
Wikipedia artefact is written.

Incidentally, gensim is not the 10x to 100x faster reference the plan assumed. On the dry run's
corpus it measured 102,198 tokens/s against our 96,474. That is one small synthetic corpus and
not a claim about text8, where its Cython should pull ahead.

## What is not proved yet

Stated plainly, because the rest of this note is measured and these are not.

- **No real corpus has been trained on.** No text8, no Wikipedia, no HN. Every number here is
  from a synthetic corpus or a microbenchmark. That is the brief: the algorithms get reviewed
  before an epoch is spent.
- **The Wikipedia acquisition path has never run.** `prepare_wikipedia_subset` reads the
  Hugging Face dump with duckdb over `hf://`, the same way `data/ingest.py` reads the HN dump,
  but exercising it means downloading a corpus. If the query or the dataset path is wrong,
  stage 1 fails there first, and that failure is contained: stage 3 still produces `hn-only`.
- **`download_text8` has never run**, for the same reason.
- **The gate thresholds are calibrated on synthetic corpora**, where a correct implementation
  agrees with gensim far more closely than it will on text8. Re-derive them at the first real
  gate run, and expect them to loosen.
- **The HN corpus reader has not been run over the full 365 MB table.** It is exercised in the
  dry run only through a synthetic stand-in.
