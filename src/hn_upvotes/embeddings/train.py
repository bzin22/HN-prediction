"""Training harness for both word2vec objectives.

Hyperparameter defaults come from Mikolov et al. 2013, "Distributed Representations of
Words and Phrases and their Compositionality". The paper recommends 5 to 20 negative
samples for small corpora and 2 to 5 for large ones, and reports the unigram
distribution raised to the 0.75 power as the best of the noise distributions it tried.
Subsampling of frequent words uses the paper's ``t = 1e-5`` rule.

The two matrices, the sampler and the loss are in
:mod:`hn_upvotes.embeddings.negative_sampling`. This module holds the corpus side: the
subsampling, the windowing, the batching, the optimiser and the learning-rate schedule.

Needs the ``train`` extra. See ``docs/word2vec.md`` for the reasoning.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from hn_upvotes.data.preprocess import Vocabulary, build_vocabulary, stream_corpus
from hn_upvotes.embeddings.cbow import CBOWObjective
from hn_upvotes.embeddings.checkpoint import Checkpoint, config_to_json
from hn_upvotes.embeddings.negative_sampling import (
    NegativeSampler,
    NegativeSamplingObjective,
    build_noise_distribution,
    sample_negatives,
    subsample_probabilities,
)
from hn_upvotes.embeddings.skipgram import SkipGramObjective

# Re-exported so the numeric primitives have one definition, in the module that owns the
# sampler, while still being importable from the harness that the rest of the project
# talks to.
__all__ = [
    "CBOWBatch",
    "SGNSConfig",
    "SkipGramBatch",
    "TrainedEmbeddings",
    "build_noise_distribution",
    "build_objective",
    "build_windows",
    "linear_learning_rate",
    "sample_negatives",
    "select_device",
    "subsample_probabilities",
    "subsample_tokens",
    "train_embeddings",
]

#: Batches between wall-clock checks in the training loop. At roughly 10 ms a batch this
#: overshoots a deadline by half a second at most, which is nothing against a two-hour
#: budget, and keeps the clock read out of the inner loop.
_DEADLINE_CHECK_STEPS = 50


@dataclass(frozen=True)
class SGNSConfig:
    """Word2vec training settings.

    negative_samples
        ``k``. 15 for text8 and for HN titles, 5 for the Wikipedia subset. Both are in
        the paper's recommended ranges for those corpus sizes, and both are starting
        points to tune, not fixed values.
    noise_power
        Exponent on the unigram distribution the negatives are drawn from. 0.75 is the
        paper's tuned value: it flattens the distribution so rare words appear as
        negatives more often than their raw frequency would allow.
    subsample_threshold
        ``t`` in the frequent-word subsampling rule. A word is discarded with
        probability ``1 - sqrt(t / f)``.
    dynamic_window
        Sample the window size uniformly from 1 to ``window`` per centre word, which
        weights nearer context words more heavily at no extra cost.
    learning_rate
        The rate applied to **one training example**, which is gensim's ``alpha`` and its
        default of 0.025. It decays linearly to ``min_learning_rate`` over the whole run,
        which is gensim's schedule and its ``min_alpha`` default. Because
        :meth:`NegativeSamplingObjective.negative_sampling_loss` reduces the batch with a
        mean, the optimiser is given ``learning_rate * batch_size``. See
        :func:`optimiser_learning_rate`.
    sparse_gradients
        Keep the embedding gradients sparse. Measured to work on this machine's MPS
        backend, so there is no reason to turn it off outside a bug hunt.
    device
        Force a device by name. ``None`` calls :func:`select_device`, which returns CPU
        because CPU measured 4.3x faster than MPS here.
    """

    objective: Literal["cbow", "skipgram"] = "skipgram"
    dimension: int = 300
    window: int = 5
    negative_samples: int = 15
    noise_power: float = 0.75
    subsample_threshold: float = 1e-5
    dynamic_window: bool = True
    min_count: int = 5
    vocabulary_cap: int = 100_000
    epochs: int = 5
    batch_size: int = 1024
    learning_rate: float = 0.025
    min_learning_rate: float = 1e-4
    sparse_gradients: bool = True
    device: str | None = None
    seed: int = 0


@dataclass(frozen=True)
class CBOWBatch:
    """One CBOW batch. Contexts are padded to ``2 * window`` and masked.

    ``context_ids`` is (batch, 2 * window), ``centre_ids`` is (batch,), ``context_mask``
    is (batch, 2 * window) of bool. Padding uses word id 0 and is only safe because of
    the mask: id 0 is the unknown token, a real row of the matrix.
    """

    context_ids: np.ndarray
    centre_ids: np.ndarray
    context_mask: np.ndarray

    def __len__(self) -> int:
        return int(self.centre_ids.shape[0])


@dataclass(frozen=True)
class SkipGramBatch:
    """One Skip-gram batch: paired positives, both (batch,). No padding, so no mask."""

    centre_ids: np.ndarray
    context_ids: np.ndarray

    def __len__(self) -> int:
        return int(self.centre_ids.shape[0])


@dataclass(frozen=True)
class TrainedEmbeddings:
    """The artefact that leaves this module: a matrix and the vocabulary to index it.

    ``epoch_losses`` is the mean batch loss per epoch, carried along because a run whose
    loss did not fall is not a result and the matrix alone does not say so.

    ``tokens_per_second`` counts **in-vocabulary** corpus tokens, so words below
    ``min_count`` are not in it. On a Zipfian corpus of 2M tokens at ``min_count=5`` that
    is 6.4% fewer than the raw token count. On a resumed run it covers only the tokens this
    call trained on, not the ones the checkpoint already had.

    ``epoch_losses`` holds one entry per *completed* epoch. When a budget stops the run
    part-way through an epoch, that epoch's mean loss goes in ``partial_epoch_loss`` instead,
    so the list length and ``epochs_completed`` cannot drift apart across a resume.

    ``cut_short`` is true when a wall-clock budget stopped the run before it finished its
    epochs. The matrix is still usable and is still worth keeping; it just did not get the
    training it was configured for, and nothing downstream should read it as if it did.

    ``loss_trace`` is the mean loss over each block of ``loss_trace_every`` batches, empty
    unless that argument was passed. One number per epoch cannot tell a rate that converges
    from a rate that diverges half way through and recovers its average, which is why the
    learning-rate sweep asks for this instead.
    """

    matrix: np.ndarray
    vocabulary: Vocabulary
    config: SGNSConfig
    tokens_per_second: float
    epoch_losses: tuple[float, ...] = ()
    epochs_completed: int = 0
    cut_short: bool = False
    tokens_trained: int = 0
    partial_epoch_loss: float = float("nan")
    loss_trace: tuple[float, ...] = ()

    def save(self, path: Path) -> None:
        """Write matrix and vocabulary together, so they cannot drift apart."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            matrix=self.matrix,
            words=np.asarray(self.vocabulary.index_to_word, dtype=object),
            counts=self.vocabulary.counts,
            unknown_index=np.asarray(self.vocabulary.unknown_index),
            config=np.asarray(json.dumps(asdict(self.config))),
            tokens_per_second=np.asarray(self.tokens_per_second),
            epoch_losses=np.asarray(self.epoch_losses, dtype=np.float64),
            epochs_completed=np.asarray(self.epochs_completed),
            cut_short=np.asarray(self.cut_short),
            tokens_trained=np.asarray(self.tokens_trained),
            partial_epoch_loss=np.asarray(self.partial_epoch_loss),
            loss_trace=np.asarray(self.loss_trace, dtype=np.float64),
        )

    @classmethod
    def load(cls, path: Path) -> TrainedEmbeddings:
        """Read embeddings written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=True) as payload:
            words = [str(w) for w in payload["words"]]
            vocabulary = Vocabulary(
                word_to_index={word: i for i, word in enumerate(words)},
                index_to_word=words,
                counts=payload["counts"],
                unknown_index=int(payload["unknown_index"]),
            )
            return cls(
                matrix=payload["matrix"],
                vocabulary=vocabulary,
                config=SGNSConfig(**json.loads(str(payload["config"]))),
                tokens_per_second=float(payload["tokens_per_second"]),
                epoch_losses=tuple(float(x) for x in payload["epoch_losses"]),
                epochs_completed=int(payload["epochs_completed"]),
                cut_short=bool(payload["cut_short"]),
                tokens_trained=int(payload["tokens_trained"]),
                partial_epoch_loss=float(payload["partial_epoch_loss"]),
                # Absent from artefacts written before the loss trace existed.
                loss_trace=(
                    tuple(float(x) for x in payload["loss_trace"])
                    if "loss_trace" in payload.files
                    else ()
                ),
            )


def select_device() -> torch.device:
    """Return CPU. **Measured, not assumed.**

    No CUDA branch. The hardware for this project is an Apple M2 with 24 GB of unified
    memory, and CI runs on CPU.

    The scaffold assumed MPS would win and it does not. Measured on this machine with
    ``make throughput``, skip-gram at 100,000 words by 300 dimensions, ``k=15``, batch
    1024, in training pairs per second:

    ==========  ==============  ==============
    device      sparse grads    dense grads
    ==========  ==============  ==============
    cpu         **105,142**     34,469
    mps         24,383          38,335
    ==========  ==============  ==============

    CPU with sparse gradients is 4.3x faster than MPS with sparse gradients, and 2.7x
    faster than the best MPS option. The reason is that a step touches about 16,000 rows of
    300 floats, which is far too little arithmetic to pay for the kernel launches, and MPS
    builds the sparse gradient slowly enough that its own dense path beats it. This is the
    same reason gensim is CPU threads with no GPU at all.

    MPS is still reachable through ``SGNSConfig.device="mps"``, because a larger dimension
    or batch could move the balance and that should be re-measured rather than assumed.
    """
    return torch.device("cpu")


def build_objective(config: SGNSConfig, vocabulary_size: int) -> NegativeSamplingObjective:
    """Construct the objective ``config`` names. The two share everything but ``forward``."""
    objective_class = CBOWObjective if config.objective == "cbow" else SkipGramObjective
    return objective_class(
        vocabulary_size=vocabulary_size,
        dimension=config.dimension,
        sparse_gradients=config.sparse_gradients,
        seed=config.seed,
    )


def linear_learning_rate(config: SGNSConfig, progress: float) -> float:
    """The per-example rate at ``progress`` through the run, in [0, 1].

    Linear from ``learning_rate`` to ``min_learning_rate``, which is gensim's schedule.
    At the defaults that is 0.025 falling to 0.0001.
    """
    progress = min(max(progress, 0.0), 1.0)
    return config.learning_rate + progress * (config.min_learning_rate - config.learning_rate)


def optimiser_learning_rate(per_example_rate: float, batch_size: int) -> float:
    """Scale a per-example rate for a loss that was reduced with a mean.

    The loss divides by the batch size, so a row appearing once in a batch of 1024 moves
    by ``rate / 1024`` times its gradient. gensim updates one example at a time and moves
    that row by ``alpha`` times its gradient. Multiplying back by the batch size makes the
    two the same size: ``0.025 * 1024 = 25.6``.

    The number looks alarming and is not. It is the linear scaling rule, and the quantity
    that reaches a single row of the matrix is still 0.025 times a gradient. What batching
    does change is that the updates inside one batch do not see each other, where gensim's
    sequential ones do. That is inherent to batching and is a reason to expect a small gap
    against gensim, not a reason to leave the step size a thousand times too small.
    """
    return per_example_rate * batch_size


def subsample_tokens(
    token_ids: np.ndarray,
    keep_probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Drop frequent words, keeping each occurrence with its per-word probability.

    Applied per occurrence, not per word type, so "the" survives in a few places rather
    than being deleted from the vocabulary.
    """
    token_ids = np.asarray(token_ids)
    if token_ids.size == 0:
        return token_ids
    keep = rng.random(token_ids.shape[0]) < keep_probabilities[token_ids]
    return token_ids[keep]


def build_windows(
    token_ids: np.ndarray,
    window: int,
    dynamic: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Context ids and mask for every position, both (n, 2 * window).

    One row per centre word, holding the ids of the words around it. A position is masked
    off when it falls outside the sequence or outside the sampled window, so both
    objectives read the same windowing: CBOW averages a row under its mask, Skip-gram
    flattens the masked entries into pairs.

    With ``dynamic`` the reach is drawn uniformly from 1 to ``window`` per centre word.
    That is the paper's trick for weighting nearer words more heavily: a word one position
    away is inside every one of the 5 possible windows, a word 5 away is inside 1 of them,
    so it is seen a fifth as often at no extra cost.
    """
    token_ids = np.asarray(token_ids)
    n = token_ids.shape[0]
    offsets = np.concatenate([np.arange(-window, 0), np.arange(1, window + 1)])
    if n == 0:
        empty = np.zeros((0, offsets.shape[0]), dtype=token_ids.dtype)
        return empty, np.zeros((0, offsets.shape[0]), dtype=bool)

    neighbour = np.arange(n)[:, None] + offsets[None, :]
    inside_sequence = (neighbour >= 0) & (neighbour < n)
    reach = rng.integers(1, window + 1, size=n)[:, None] if dynamic else window
    mask = inside_sequence & (np.abs(offsets)[None, :] <= reach)

    # Clip before gathering so the out-of-range indices are legal; the mask discards them.
    context_ids = np.where(mask, token_ids[np.clip(neighbour, 0, n - 1)], 0)
    return context_ids, mask


class BatchFeeder:
    """Turns tokenised lines into fixed-shape batches for one objective.

    Buffers across line boundaries, because an HN title is about 8 tokens and a batch is
    1024.

    ``tokens_read`` counts what reached the feeder, which is corpus tokens after
    :func:`encode_corpus` has dropped the out-of-vocabulary ones. That is the number the
    throughput figure is reported against. ``tokens_kept`` is what survived subsampling,
    and their ratio is the retention the subsampling threshold bought.
    """

    def __init__(self, config: SGNSConfig, keep_probabilities: np.ndarray, seed: int = 0) -> None:
        self.config = config
        self.keep_probabilities = keep_probabilities
        self.rng = np.random.default_rng(seed)
        self.tokens_read = 0
        self.tokens_kept = 0

    def batches(self, token_id_lists: Iterable[np.ndarray]) -> Iterator[CBOWBatch | SkipGramBatch]:
        """Yield batches until the corpus runs out, including a short final one."""
        parts: list[tuple[np.ndarray, ...]] = []
        buffered = 0
        for token_ids in token_id_lists:
            self.tokens_read += int(np.asarray(token_ids).shape[0])
            kept = subsample_tokens(token_ids, self.keep_probabilities, self.rng)
            self.tokens_kept += int(kept.shape[0])
            if kept.shape[0] < 2:
                continue
            part = self._examples(kept)
            if part[0].shape[0] == 0:
                continue
            parts.append(part)
            buffered += part[0].shape[0]
            while buffered >= self.config.batch_size:
                joined = tuple(np.concatenate(columns) for columns in zip(*parts, strict=True))
                cut = self.config.batch_size
                yield self._batch(tuple(column[:cut] for column in joined))
                parts = [tuple(column[cut:] for column in joined)]
                buffered -= cut
        if buffered:
            joined = tuple(np.concatenate(columns) for columns in zip(*parts, strict=True))
            yield self._batch(joined)

    def _examples(self, token_ids: np.ndarray) -> tuple[np.ndarray, ...]:
        context_ids, mask = build_windows(
            token_ids, self.config.window, self.config.dynamic_window, self.rng
        )
        if self.config.objective == "cbow":
            # One example per position. Rows with no context at all teach nothing.
            has_context = mask.any(axis=1)
            return context_ids[has_context], token_ids[has_context], mask[has_context]
        # One pair per unmasked context position.
        centre = np.repeat(token_ids, mask.shape[1]).reshape(mask.shape)
        return centre[mask], context_ids[mask]

    def _batch(self, columns: tuple[np.ndarray, ...]) -> CBOWBatch | SkipGramBatch:
        if self.config.objective == "cbow":
            return CBOWBatch(context_ids=columns[0], centre_ids=columns[1], context_mask=columns[2])
        return SkipGramBatch(centre_ids=columns[0], context_ids=columns[1])


def encode_corpus(
    token_lists: Iterable[list[str]],
    vocabulary: Vocabulary,
) -> Iterator[np.ndarray]:
    """Map tokens to ids and drop the out-of-vocabulary ones.

    Dropped rather than mapped to the unknown token, which is what gensim does: the
    window then closes over the gap, so two words either side of a dropped rare word
    become neighbours. Keeping a placeholder would put a high-frequency filler between
    them instead.

    The unknown row stays in the matrix so serving has something to return for a word it
    has never seen. It is never trained, so it keeps its initialisation and pooling should
    skip it rather than average it in.
    """
    unknown = vocabulary.unknown_index
    for tokens in token_lists:
        ids = np.asarray(vocabulary.encode(tokens), dtype=np.int64)
        yield ids[ids != unknown]


def _warm_start(
    model: NegativeSamplingObjective,
    vocabulary: Vocabulary,
    initial: TrainedEmbeddings,
) -> int:
    """Copy rows from ``initial`` into the input matrix by word, and return how many.

    Words in the new vocabulary that ``initial`` never saw keep their random
    initialisation. This is how the fine-tuned variant is built.
    """
    source = initial.vocabulary.word_to_index
    pairs = [(i, source[word]) for i, word in enumerate(vocabulary.index_to_word) if word in source]
    if not pairs:
        return 0
    target_rows, source_rows = (np.asarray(column) for column in zip(*pairs, strict=True))
    with torch.no_grad():
        model.input_matrix.weight[torch.as_tensor(target_rows)] = torch.as_tensor(
            initial.matrix[source_rows], dtype=model.input_matrix.weight.dtype
        )
    return len(pairs)


def train_embeddings(
    corpus_path: Path,
    config: SGNSConfig,
    output_path: Path | None = None,
    initial: TrainedEmbeddings | None = None,
) -> TrainedEmbeddings:
    """Train one objective over one corpus and return the embedding matrix.

    ``initial`` warm-starts from an existing matrix, which is how the fine-tuned variant
    is built: Wikipedia vectors, then HN titles at a lower learning rate. Words present
    in HN but not in Wikipedia are randomly initialised before fine-tuning.

    Reports measured tokens per second, because the Wikipedia subset size is chosen from
    that number rather than guessed. A from-scratch PyTorch SGNS runs one to two orders
    of magnitude slower than gensim's Cython, and the plan accounts for that.

    The corpus is streamed once per epoch rather than held in memory, so ``corpus_path``
    is read ``config.epochs + 1`` times: once to count the vocabulary, then once per epoch.
    """
    corpus_path = Path(corpus_path)
    vocabulary = build_vocabulary(
        stream_corpus(corpus_path), min_count=config.min_count, max_size=config.vocabulary_cap
    )
    return train_embeddings_from_lines(
        lambda: stream_corpus(corpus_path),
        vocabulary,
        config,
        output_path=output_path,
        initial=initial,
    )


def train_embeddings_from_lines(
    reopen_corpus: object,
    vocabulary: Vocabulary,
    config: SGNSConfig,
    output_path: Path | None = None,
    initial: TrainedEmbeddings | None = None,
    progress_every: int = 0,
    checkpoint_directory: Path | None = None,
    stage: str = "train",
    resume: Checkpoint | None = None,
    deadline: float | None = None,
    loss_trace_every: int = 0,
) -> TrainedEmbeddings:
    """The training loop, over a callable that reopens the corpus for each epoch.

    Split out from :func:`train_embeddings` so a test can drive it from an in-memory
    corpus without writing a file, and so the throughput measurement can reuse it.
    ``reopen_corpus`` is called once per epoch and must return a fresh iterator of token
    lists.

    The last four arguments are what unattended overnight operation needs:

    ``checkpoint_directory``
        Write a :class:`~hn_upvotes.embeddings.checkpoint.Checkpoint` after every epoch,
        named for ``stage``. Both matrices go in, because resuming from the input matrix
        alone would restart scoring from zero.
    ``resume``
        Carry on from a checkpoint instead of from a random initialisation. Skips the
        epochs it already did.
    ``deadline``
        A ``time.monotonic()`` value. When it passes the loop stops between batches, saves
        what it has, and returns with ``cut_short=True`` set on the artefact. It does not
        pretend the run finished.

    ``loss_trace_every`` records the mean loss over each block of that many batches, which is
    the within-epoch loss curve the learning-rate sweep compares. It costs one device
    synchronisation per block, so leave it at 0 in a real run.
    """
    device = torch.device(config.device) if config.device else select_device()
    model = build_objective(config, len(vocabulary)).to_device(device)
    if initial is not None:
        _warm_start(model, vocabulary, initial)

    sampler = NegativeSampler(vocabulary.counts, config.noise_power, device, config.seed)
    keep_probabilities = subsample_probabilities(vocabulary.counts, config.subsample_threshold)
    optimiser = torch.optim.SGD(
        model.parameters(), lr=optimiser_learning_rate(config.learning_rate, config.batch_size)
    )

    # Tokens the schedule expects to see, so the rate reaches min_learning_rate at the end.
    total_tokens = max(int(vocabulary.counts.sum()) * config.epochs, 1)
    feeder = BatchFeeder(config, keep_probabilities, seed=config.seed)
    step = 0
    epoch_losses: list[float] = []
    loss_trace: list[float] = []
    block_loss_total = torch.zeros((), device=device)
    block_steps = 0
    partial_epoch_loss = float("nan")
    first_epoch = 0

    if resume is not None:
        _load_checkpoint_into(model, resume)
        first_epoch = resume.epochs_done
        epoch_losses = list(resume.epoch_losses)
        # Carried so the learning-rate schedule resumes at the right point rather than
        # jumping back to the starting rate.
        feeder.tokens_read = resume.tokens_read

    started = time.perf_counter()
    cut_short = False

    for epoch in range(first_epoch, config.epochs):
        # Accumulated as a tensor on the device. Calling float() on the loss every step
        # would block on the MPS queue every step and turn the throughput number into a
        # measurement of the synchronisation instead of the training.
        epoch_loss_total = torch.zeros((), device=device)
        epoch_steps = 0
        for batch in feeder.batches(encode_corpus(reopen_corpus(), vocabulary)):
            rate = linear_learning_rate(config, feeder.tokens_read / total_tokens)
            for group in optimiser.param_groups:
                group["lr"] = optimiser_learning_rate(rate, len(batch))
            loss = _batch_loss(model, batch, sampler, config, device)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            step += 1
            epoch_loss_total += loss.detach()
            epoch_steps += 1
            if loss_trace_every:
                block_loss_total += loss.detach()
                block_steps += 1
                if block_steps == loss_trace_every:
                    loss_trace.append(float(block_loss_total) / block_steps)
                    block_loss_total = torch.zeros((), device=device)
                    block_steps = 0
            if progress_every and step % progress_every == 0:
                print(
                    f"epoch {epoch + 1}/{config.epochs} step {step} "
                    f"loss {float(loss.detach()):.4f} rate {rate:.5f}",
                    flush=True,
                )
            # Checked every _DEADLINE_CHECK_STEPS batches rather than every batch, so the
            # clock read is not itself part of the inner loop's cost.
            if (
                deadline is not None
                and step % _DEADLINE_CHECK_STEPS == 0
                and time.monotonic() >= deadline
            ):
                cut_short = True
                break
        mean_loss = float(epoch_loss_total) / epoch_steps if epoch_steps else float("nan")
        if cut_short:
            # The partial epoch's loss is reported but kept out of epoch_losses, which stays
            # one entry per *completed* epoch. Appending it would leave the list one longer
            # than epochs_done, and a resume would then count that partial epoch again and
            # report more epochs completed than were configured.
            partial_epoch_loss = mean_loss
        elif epoch_steps:
            epoch_losses.append(mean_loss)
        if checkpoint_directory is not None:
            # An epoch cut short banks its progress under the previous epoch number, so a
            # resume repeats that epoch rather than skipping the part it never did.
            _checkpoint(
                model,
                vocabulary,
                config,
                stage,
                epochs_done=epoch if cut_short else epoch + 1,
                epoch_losses=epoch_losses,
                tokens_read=feeder.tokens_read,
                directory=Path(checkpoint_directory),
            )
        if cut_short:
            break

    if block_steps:
        # The tail block is shorter than the rest. Keeping it means the curve reaches the end
        # of the run, which is the half a diverging rate shows up in.
        loss_trace.append(float(block_loss_total) / block_steps)

    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - started
    tokens_trained = feeder.tokens_read - (resume.tokens_read if resume else 0)
    # A stage resumed from a checkpoint that was already finished trains nothing. Reporting
    # 0 tokens/s for that reads like a failure, so it reports "not measured" instead.
    tokens_per_second = (
        tokens_trained / elapsed if tokens_trained > 0 and elapsed > 0 else float("nan")
    )

    trained = TrainedEmbeddings(
        matrix=model.input_embeddings().to("cpu").numpy(),
        vocabulary=vocabulary,
        config=config,
        tokens_per_second=tokens_per_second,
        epoch_losses=tuple(epoch_losses),
        epochs_completed=len(epoch_losses),
        cut_short=cut_short,
        tokens_trained=tokens_trained,
        partial_epoch_loss=partial_epoch_loss,
        loss_trace=tuple(loss_trace),
    )
    if output_path is not None:
        trained.save(Path(output_path))
    return trained


def _load_checkpoint_into(model: NegativeSamplingObjective, resume: Checkpoint) -> None:
    """Restore both matrices from a checkpoint."""
    with torch.no_grad():
        model.input_matrix.weight.copy_(torch.as_tensor(resume.input_matrix))
        model.output_matrix.weight.copy_(torch.as_tensor(resume.output_matrix))


def _checkpoint(
    model: NegativeSamplingObjective,
    vocabulary: Vocabulary,
    config: SGNSConfig,
    stage: str,
    epochs_done: int,
    epoch_losses: list[float],
    tokens_read: int,
    directory: Path,
) -> Path:
    """Write both matrices and the position in the run."""
    with torch.no_grad():
        return Checkpoint(
            stage=stage,
            epochs_done=epochs_done,
            input_matrix=model.input_matrix.weight.detach().to("cpu").numpy(),
            output_matrix=model.output_matrix.weight.detach().to("cpu").numpy(),
            vocabulary=vocabulary,
            config_json=config_to_json(config),
            epoch_losses=tuple(epoch_losses),
            tokens_read=tokens_read,
        ).save(directory)


def _batch_loss(
    model: NegativeSamplingObjective,
    batch: CBOWBatch | SkipGramBatch,
    sampler: NegativeSampler,
    config: SGNSConfig,
    device: torch.device,
) -> Tensor:
    """Move one batch to the device, draw its negatives, and return the loss."""
    negatives = sampler.draw(len(batch), config.negative_samples)
    if isinstance(batch, CBOWBatch):
        return model(
            torch.as_tensor(batch.context_ids, dtype=torch.long).to(device),
            torch.as_tensor(batch.centre_ids, dtype=torch.long).to(device),
            negatives,
            torch.as_tensor(batch.context_mask).to(device),
        )
    return model(
        torch.as_tensor(batch.centre_ids, dtype=torch.long).to(device),
        torch.as_tensor(batch.context_ids, dtype=torch.long).to(device),
        negatives,
    )


def main(argv: list[str] | None = None) -> None:
    """Train one objective over one corpus from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("corpus", type=Path, help="plain-text corpus, one or more lines")
    parser.add_argument("output", type=Path, help="where to write the .npz artefact")
    parser.add_argument("--objective", choices=("cbow", "skipgram"), default="skipgram")
    parser.add_argument("--dimension", type=int, default=SGNSConfig.dimension)
    parser.add_argument("--epochs", type=int, default=SGNSConfig.epochs)
    parser.add_argument("--negative-samples", type=int, default=SGNSConfig.negative_samples)
    parser.add_argument("--batch-size", type=int, default=SGNSConfig.batch_size)
    parser.add_argument("--device", default=None, help="cpu or mps; default picks MPS if present")
    parser.add_argument("--progress-every", type=int, default=200, help="steps between log lines")
    args = parser.parse_args(argv)

    config = replace(
        SGNSConfig(),
        objective=args.objective,
        dimension=args.dimension,
        epochs=args.epochs,
        negative_samples=args.negative_samples,
        batch_size=args.batch_size,
        device=args.device,
    )
    corpus_path = Path(args.corpus)
    vocabulary = build_vocabulary(
        stream_corpus(corpus_path), min_count=config.min_count, max_size=config.vocabulary_cap
    )
    print(f"vocabulary {len(vocabulary)} words, {int(vocabulary.counts.sum())} tokens", flush=True)
    trained = train_embeddings_from_lines(
        lambda: stream_corpus(corpus_path),
        vocabulary,
        config,
        output_path=Path(args.output),
        progress_every=args.progress_every,
    )
    print(f"wrote {args.output} at {trained.tokens_per_second:,.0f} corpus tokens/s")


if __name__ == "__main__":
    main()
