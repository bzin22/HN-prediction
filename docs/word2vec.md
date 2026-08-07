# Word2vec from scratch: the implementation

Why the code is shaped the way it is. [`design.md`](design.md) holds the project's wider
reasoning and the embedding hyperparameter table; this note covers the implementation itself
and the two measurements that could have invalidated it.

**text8 has now been trained on, for two hyperparameter measurements and nothing else.** The
learning-rate sweep and the batch-scaling check are 5M and 120k tokens of it. Wikipedia and
Hacker News have not been trained on, and no variant has been produced. Every other number
below comes from a synthetic corpus or a microbenchmark, and each says which.

## Contents

- [Terminology](#terminology)
- [Two matrices, one of which survives](#two-matrices-one-of-which-survives)
- [Why negative sampling](#why-negative-sampling)
- [The two objectives differ by one line](#the-two-objectives-differ-by-one-line)
- [Why plain SGD](#why-plain-sgd)
- [Settling the learning rate on text8](#settling-the-learning-rate-on-text8)
- [The learning rate is per example, and the batch scales it](#the-learning-rate-is-per-example-and-the-batch-scales-it)
- [Subsampling, and two ways gensim differs](#subsampling-and-two-ways-gensim-differs)
- [Risk 1: sparse gradients](#risk-1-sparse-gradients)
- [Risk 2: the GPU or the Python](#risk-2-the-gpu-or-the-python)
- [What the measurements changed](#what-the-measurements-changed)
- [The overnight chain](#the-overnight-chain)
- [Both objectives, six variants](#both-objectives-six-variants)
- [Sizing the Wikipedia stage](#sizing-the-wikipedia-stage)
- [What a night costs](#what-a-night-costs)
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

## Settling the learning rate on text8

The rate was moved from the scaffold's 0.0025 to gensim's 0.025 on the strength of an
argument. `make lr-sweep` measures it instead.

Four rates over an identical 5,000,000-token slice of text8, **three seeds each**, one epoch,
dimension 300, `k = 15`, batch 1024. One vocabulary of 36,281 word types built once and shared,
so the corpus, the vocabulary and the point at which the schedule reaches `min_learning_rate`
are the same in all twelve runs and the only difference is the rate.

| Rate | Blew up | Final loss | Worst lurch | As a share of the fall | WordSim-353 | Analogy |
|---|---:|---:|---:|---:|---:|---:|
| 0.0025 | 0 of 3 | 5.5199 | 0.171 | 3.1% | -0.015 | 0.00003 |
| 0.01 | 0 of 3 | 3.9568 | 0.215 | 3.0% | 0.031 | 0.00000 |
| **0.025** | **0 of 3** | **3.8915** | 0.408 | 5.8% | **0.079** | **0.00028** |
| 0.05 | **2 of 3** | 4.1501 | 1.498 | 24.9% | 0.089 | 0.00064 |

The loss columns for 0.05 are its **one surviving seed**, because averaging a 2.5e16 into a
mean produces a number that is not about anything. The two intrinsic scores are over all three
seeds, since a blown-up run still leaves a matrix and dropping its score would flatter the
rate.

**The answer is 0.025, and it is unchanged.** The argument for it happened to be right; the
measurement is now the reason for it.

### The curves

The loss is recorded every 50 batches, so a run is 153 points. Seed 0, sampled at 0%, 10%,
20%, 30%, 40%, 60%, 80% and 100% through the epoch:

| Rate | Loss curve, seed 0 |
|---|---|
| 0.0025 | 11.09 → 11.00 → 9.18 → 7.78 → 6.92 → 6.02 → 5.69 → **5.54** |
| 0.01 | 11.09 → 6.01 → 4.70 → 4.31 → 4.14 → 3.95 → 3.95 → **4.00** |
| 0.025 | 10.87 → 4.72 → 4.10 → 4.00 → 3.88 → 3.74 → 3.80 → **3.93** |
| 0.05 | 10.16 → 5.20 → 4.96 → 4.71 → 4.45 → 4.08 → 3.97 → **4.15** |

Every curve starts near 11.09, which is the `(1 + k) * log 2` a zero output matrix forces
before any step. 0.0025 has still not reached 9 by the time the others are under 5.

The 0.05 row is the one seed that survived, and it is smoothed by this sampling. At full
resolution its first fifth reads 10.16, 5.44, **6.29**, 5.05, **5.47**, 4.45: it goes back up
by more than a full nat twice. Its two other seeds do not recover:

| Rate 0.05 | Loss curve |
|---|---|
| seed 0 | 10.16 → 5.20 → 4.96 → 4.71 → 4.45 → 4.08 → 3.97 → **4.15** |
| seed 1 | 10.22 → 7.2e6 → 5.1e12 → 4.8e12 → 2.6e12 → 8.6e11 → 3.4e11 → **1.2e11** |
| seed 2 | 10.18 → 1.6e9 → 4.0e17 → 5.2e17 → 3.2e17 → 8.0e16 → 3.4e16 → **2.5e16** |

Both are gone inside the first tenth of the epoch. The decay to `min_learning_rate` then pulls
the loss back down by orders of magnitude over the rest of the run, which is why the final
number is smaller than the peak and why "did the loss fall" is not on its own a test.

Both ends behaved as predicted, one of them more dramatically than expected.

**0.0025 crawls.** It descends perfectly smoothly and gets nowhere: after all 5M tokens it is
at 5.52, a loss the other rates pass inside the first 5% of their run. Its vectors score
-0.015 on WordSim-353, which is nothing, on all three seeds.

**0.05 diverges.** Not lurches, diverges: seed 1 ended at 1.2e11 and seed 2 at 2.5e16, against
a starting loss of 11.09. Seed 0 was the survivor, and even it lurched by 1.50 in one block,
24.9% of its total fall against 5.8% for 0.025.

### The seeds are the point

A single-seed sweep drew seed 0 first, and on seed 0 rate 0.05 does not blow up. It posts the
best WordSim-353 score of the four and the best analogy accuracy, and the whole thing turns on
reading a lurch in a loss curve. **Three seeds turned a judgement call into a result.**

The intrinsic scores needed the seeds even more than the loss did. On one seed the four rates
scored -0.018, 0.091, 0.058 and 0.149, which reads as a clean ranking putting 0.01 above
0.025. Over three seeds, 0.01 scores 0.091, -0.023 and 0.023. Its apparent lead was one draw.

So every comparison is checked against a noise band before it decides anything, and the band
is the larger of two numbers. The sampling error on a Spearman correlation over `n` pairs is
about `1 / sqrt(n - 3)`, and WordSim-353 covers 336 of its 353 pairs at this vocabulary, so
that is **0.055**. The other is the measured spread across the seeds, which for 0.01 is 0.114,
more than twice the formula's estimate. Without that check the rule throws 0.025 out for being
"36% short" of a number that was noise.

Analogy accuracy is the one that discriminates cleanly despite being tiny. It counts correct
answers out of 13,095 in-vocabulary questions, so 0.00028 against 0.00000 is 3.7 correct
against 0, and a difference in counts that small is still a real difference in a way a
correlation coefficient is not. It agrees with WordSim-353 and with the loss: 0.025 is the best
of the stable rates on all three.

### What the rule is

Two tests on the rate alone, then one comparison, then take the largest survivor. A larger
rate that converges gets further in the same wall clock, and this project's ceiling is wall
clock.

1. **It converged.** No seed blew up, and the mean loss fell by at least 5% of the 11.09 that
   a zero output matrix forces at the start. 0.05 fails the first; 0.0025 clears both, barely.
2. **It converged smoothly.** No single block of 50 batches rose by more than 10% of the total
   fall. The stable rates sit at 3.0% to 5.8% and the surviving 0.05 seed at 24.9%, so the
   threshold is 1.7x clear of one end and 2.5x clear of the other. It was 0.25 on the first
   pass, chosen blind, and 0.05 cleared it by 0.0009.
3. **Its vectors are not meaningfully worse than the best stable rate's**, where "meaningfully"
   means both over 20% and wider than the noise band. 0.0025 fails this: a gap of 0.094
   against a band of 0.055.

The bar in the third test is set by the best **stable** rate, not the best rate overall. 0.05
scored the highest WordSim-353 of the four and was already out; letting it set the bar would
have disqualified every rate that did not diverge.

## The learning rate is per example, and the batch scales it

This is the one piece of arithmetic in the harness that looks wrong and is not. **It was
argued first and has now been measured**, with `make batch-scaling`.

`forward` returns the **mean** loss over the batch. gensim updates one example at a time. So a
word appearing once in a batch of 1024 moves by `rate / 1024` times its gradient, where
gensim's same word moves by `alpha` times its gradient. Left alone, our steps would be a
thousand times too small.

`SGNSConfig.learning_rate` is therefore the **per-example** rate, gensim's `alpha` and its
default of 0.025, and the optimiser is given `learning_rate * batch_size`:

```
0.025 * 1024 = 25.6
```

### Measured, on 120,000 tokens of text8

Four arms, one corpus, one seed, **83,331 training examples per epoch identical for every
arm** and two epochs, so the arms differ by batch size and nothing else. Dimension 64,
`k = 15`, per-example rate 0.025 throughout. The fourth arm is the control: the same batch of
1024 with `optimiser_learning_rate` replaced by the identity, which is what the code would do
if the multiplication were deleted.

| Arm | Optimiser rate | Epoch losses | Steps | Against batch 1 | Neighbour overlap |
|---|---:|---|---:|---:|---:|
| batch 1 | 0.025 | 7.8857, **4.8183** | 166,662 | | |
| batch 32 | 0.8 | 8.7407, **4.9987** | 5,210 | +3.7% | 0.501 |
| batch 1024 | 25.6 | 9.3678, **5.4211** | 164 | +12.5% | 0.108 |
| batch 1024, unscaled | 0.025 | 11.0790, **11.0790** | 164 | +129.9% | 0.132 |

**The multiplication is right, and it is the difference between learning and not learning.**
Without it, the same batch does not move at all: 11.0790 is the `(1 + k) * log 2 = 11.0904`
that a zero output matrix forces before a single step. With it, the loss more than halves.
And at 25.6 nothing diverges, which was the other way the claim could have failed.

The test is deliberately a ratio and not a tolerance. The scaled arm's residual gap of 0.603
is **10.4x smaller** than the control's 6.261, and no threshold anywhere in that range turns
the answer around. The first pass at this did put a 10% tolerance on the batch-1 gap, blind,
and then measured 12.5%, which is a threshold deciding the answer rather than reporting it.

### The residual is real and now has a size

The scaling is right in kind and it is not exact. The gap against batch 1 is **+3.7% at batch
32 and +12.5% at batch 1024**, growing with the batch, which is the shape one effect predicts:
updates inside a batch do not see each other, where a batch-1 run's do. No learning rate can
undo that. It is a cost of batching, so it is reported and not gated.

Two consequences worth carrying forward. Part of whatever gap the gate sees against gensim is
this, not a bug, because gensim updates sequentially. And **batch 1024 is still the right
default**: batch 32 gives up 8.8 points of that gap but takes 6 seconds against 2 for the same
83,331 examples, and this project's ceiling is wall clock.

The neighbour overlap column says the same thing less kindly. Two runs that reach a similar
loss do not reach the same matrix: batch 32 agrees with batch 1 on half of each frequent
word's ten nearest neighbours and batch 1024 on a tenth. At 120,000 tokens both matrices are
undertrained, so this is a caveat on reading loss as identity rather than a measurement of
either.

**The scaffold's default was 2.5e-3, ten times lower and on the scale of an Adam rate rather
than an SGD one.** It is 0.025 now, and [the sweep](#settling-the-learning-rate-on-text8) is
why rather than the argument for mirroring gensim.

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

[`chain.py`](../src/hn_upvotes/embeddings/chain.py) runs two objectives in one process, four
stages each, in order. `make chain-dry-run` walks all eight on synthetic corpora in about 20 seconds.

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

## Both objectives, six variants

The CBOW against Skip-gram comparison is the project's stated experiment, so the chain runs
both and `ChainConfig.objectives` defaults to both. One invocation, Skip-gram then CBOW, four
stages each, **six variants**.

Three things had to hold, and each is a test.

**Names.** Every artefact is `{objective}-{stage}.npz` and every checkpoint is
`{objective}-{stage}-epoch{n}.npz`. The objective is in the file name and not only in the
manifest, because a variant gets loaded by name later and a `wiki-only.npz` that could be
either objective is a variant nobody can use in the comparison. The manifest carries
`objective` and `stage` as separate fields on every record, so it groups either way without
parsing a name back apart.

**Isolation, at two levels.** A CBOW gate that aborts costs the three CBOW variants and
nothing else; Skip-gram's three are already on disk and nothing rolls them back. The reverse
holds too, and is its own test, because the objective that happens to run first is not
special. Inside an objective, the per-stage isolation is unchanged.

**Resume.** `newest_checkpoint` matches on the qualified stage name, so a finished Skip-gram
Wikipedia stage cannot be picked up as a starting point for CBOW's. A resume after a complete
run reports `already_complete` for all six variants and trains nothing.

Two things are deliberately shared, and sharing them is what makes the comparison a
comparison:

* **The same Wikipedia subset, sized from Skip-gram's throughput.** Skip-gram is the slower
  objective, so the subset fits its two-hour ceiling and CBOW finishes the same corpus early.
  Sizing each objective to fill its own two hours would hand CBOW 2.2x the corpus and the
  comparison would be measuring corpus size.
* **Everything in `SGNSConfig` except the objective**, including the seed. A test asserts the
  two stage configs are equal once the objective field is blanked.

Fine-tuning warm-starts from **its own** objective's Wikipedia vectors, which is why the
Wikipedia outcome is passed into stage 2 rather than looked up by stage name.

### Running both immediately found something

The dry-run corpus was sized when only Skip-gram ran, and CBOW failed its gate on it the
first time: topic purity **0.392 against gensim's 1.000**, a 60.8% shortfall.

Not a bug in CBOW. With a window of 5, Skip-gram produces up to ten training examples per
position and CBOW exactly one, so on the same corpus CBOW takes **3.8x fewer optimiser steps**:
25,134 against 95,347 per epoch on that corpus. It was undertrained, and raising the epochs to
10 took it to 1.000. Batch size changes nothing here, which is itself a check on the batch
scaling below: at batch 512, 256 and 128 CBOW finished at the same 3.747 to three decimals,
because the per-example rate holds the trajectory fixed in examples rather than in steps.

The fix is the corpus, not the objective. `DRY_RUN_CORPUS` is now 10 topics of 15 words over
40,000 tokens, the cheapest shape measured where both objectives reach 1.000 against gensim's
1.000 in 3 epochs. Chance is 1/10 and a deliberately broken matrix scores 0.104, so the gate
keeps its margin.

**The general point survives the dry run: CBOW converges later in corpus terms than Skip-gram
does.** On text8 at 5 epochs it gets 27,000 optimiser steps against Skip-gram's 101,000, which
is enough, but any future corpus sized on Skip-gram's numbers has to be checked against CBOW
before it is trusted.

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

## What a night costs

Two objectives is not two nights. **7.5 hours expected, 11.0 hours of ceilings**, so it fits
a night with room, and no stage is expected to hit its own ceiling.

The HN corpus is counted rather than guessed: `make hn-token-count` reports **75,283,676
tokens over 5,091,739 lines**, 14.8 tokens a line. The bodies are half of it. 4,739,207 titles
at about 8 tokens is 38M, so the 352,532 lines of body text carry the rest on 7% of the rows.

`overnight_budget` in [`chain.py`](../src/hn_upvotes/embeddings/chain.py) is the arithmetic,
at 5 epochs, with the ceiling shown next to it:

| Objective | Stage | Corpus tokens | Expected | Ceiling |
|---|---|---:|---:|---:|
| Skip-gram | gate | 17,005,207 | 19.2 min | 30 min |
| Skip-gram | wiki-only | 191,271,628 | 120.0 min | 120 min |
| Skip-gram | fine-tuned | 75,283,676 | 84.8 min | 90 min |
| Skip-gram | hn-only | 75,283,676 | 84.8 min | 90 min |
| CBOW | gate | 17,005,207 | 8.6 min | 30 min |
| CBOW | wiki-only | 191,271,628 | 54.0 min | 120 min |
| CBOW | fine-tuned | 75,283,676 | 38.2 min | 90 min |
| CBOW | hn-only | 75,283,676 | 38.2 min | 90 min |
| | **total** | | **7.46 h** | **11.0 h** |

CBOW is the cheap half at 2.3 hours against Skip-gram's 5.2, because it is 2.224x faster per
corpus token (181,487 against 81,603 measured at dimension 300, `k = 15`). Skip-gram's
Wikipedia stage lands exactly on its two hours because that is the number the subset was sized
from, and CBOW then reads the same subset in 54 minutes.

Worked, for the Skip-gram Wikipedia stage: `191,271,628 tokens x 5 epochs / (138,362 x 0.96
tokens/s) = 7,199 s`. For CBOW, the same corpus at 2.224x the rate: `7,199 / 2.224 = 3,237 s`.

**Three costs are outside that table and one of them could matter.**

1. **gensim's half of each gate.** The gate trains gensim on text8 as well as us, and only our
   side is under the budget. Unmeasured on text8. The one measurement in hand is from the dry
   run's synthetic corpus, where gensim managed 102,198 tokens/s against our 96,474, and its
   Cython should do better on text8 than that. At 100,000 tokens/s it is 14 minutes per
   objective, 28 for both.
2. **The vocabulary pass.** Every stage counts its corpus once before training, and the
   training loop then rereads it once per epoch. For the four HN stages that is a duckdb read
   plus HTML stripping plus tokenisation, six times each, and the 77,019 tokens/s figure was
   measured on an in-memory synthetic corpus rather than on that reader. **This is the number
   most likely to be wrong**, and it is wrong in the direction of the night taking longer.
3. **The Wikipedia download.** 1.13 GB over `hf://`, once, before Skip-gram's stage 1. CBOW
   reuses the file.

If the total does not fit, **the thing to cut is epochs on the HN stages, not an objective**.
Dropping the four HN stages from 5 epochs to 3 saves 98 minutes and keeps all six variants,
where dropping CBOW saves 2.3 hours and cancels the experiment the phase exists to run. The
second lever is the Wikipedia ceiling: it is a ceiling and the subset is sized from it, so
`--wikipedia-hours 1.5` shrinks the corpus and saves 45 minutes across both objectives.

## The gate

Stage 0 trains our implementation and gensim on the same corpus with the same settings and
compares them. A failure aborts **that objective**. That is the single most valuable thing in
the design: without it a bug costs a night and shows up in the morning, and with it the chain
stops in minutes and the machine sits idle instead, which is far cheaper.

**It runs once per objective**, because it is validating that objective's implementation.
CBOW and Skip-gram differ by one line of `forward`, and that one line is the masked average,
which is the sharpest edge in the whole implementation. A Skip-gram gate says nothing about
it. gensim's `sg` flag is set from the objective under test, so each is compared against its
own reference rather than against Skip-gram's.

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

- **No variant has been produced.** Wikipedia and Hacker News have not been trained on, and
  none of the six artefacts exists. text8 has been trained on only for the two hyperparameter
  measurements, at 5M and 120k tokens, never a full run and never through the chain.
- **The gate has never run on text8.** Both objectives pass it on the synthetic dry-run
  corpus, which is not the same claim.
- **The Wikipedia acquisition path has never run.** `prepare_wikipedia_subset` reads the
  Hugging Face dump with duckdb over `hf://`, the same way `data/ingest.py` reads the HN dump,
  but exercising it means downloading 1.13 GB. If the query or the dataset path is wrong,
  stage 1 fails there first, and that failure is contained: stage 3 still produces `hn-only`,
  and the other objective is untouched.
- **The gate thresholds are calibrated on synthetic corpora**, where a correct implementation
  agrees with gensim far more closely than it will on text8. Re-derive them at the first real
  gate run, and expect them to loosen.
- **The intrinsic scores in the sweep are floor-level and are used only to rank.** WordSim-353
  at 0.079 and analogy accuracy at 0.00028 are what 5M tokens and one epoch buy; the paper
  trains on 20x that. They separate the rates and they are not a claim about the vectors.
- **`gensim`'s half of the gate has not been timed on text8**, so the overnight budget has an
  unmeasured 28 minutes in it. So does the Hacker News reader's throughput, which is the
  larger of the two unknowns: the 77,019 tokens/s the budget uses was measured on an
  in-memory synthetic corpus, not on duckdb plus HTML stripping plus tokenisation.
- **`CBOW_SPEEDUP` was measured at `k = 15`, not at the `k = 5` the Wikipedia stage uses.** It
  is used only to estimate how long CBOW's stages take, never to size a corpus.
