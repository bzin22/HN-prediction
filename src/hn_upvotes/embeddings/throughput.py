"""Measures the two things that could invalidate the word2vec approach rather than tune it.

1. **Do sparse gradients work on this machine's MPS backend?** If they do not, the fallback
   is a 120 MB dense gradient per matrix computed every step to update about 16 rows, which
   is unusable.
2. **Is the bottleneck the GPU or the Python feeding it?** Word2vec's cost is usually
   windowing, subsampling and drawing negatives, not the matrix maths. Part of why gensim is
   fast is that it is CPU threads with no GPU at all.

Run with ``make throughput``. The corpus is synthetic Zipfian text generated in memory, so
this downloads nothing and can run before any real corpus exists. Numbers land in
``docs/word2vec.md``.

Needs the ``train`` extra.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np
import torch

from hn_upvotes.data.preprocess import Vocabulary, build_vocabulary
from hn_upvotes.embeddings.train import (
    BatchFeeder,
    SGNSConfig,
    encode_corpus,
    subsample_probabilities,
    train_embeddings_from_lines,
)

#: Zipf exponent for the synthetic corpus. English word frequencies sit near 1.0, and the
#: shape matters here because it drives how much work subsampling and the noise
#: distribution do.
ZIPF_EXPONENT = 1.0


def synthetic_corpus(
    vocabulary_size: int = 100_000,
    tokens: int = 2_000_000,
    tokens_per_line: int = 20,
    seed: int = 0,
) -> list[list[str]]:
    """Zipfian nonsense with a realistic frequency profile, as lines of tokens.

    Not text. It has no distributional structure to learn, which is fine: this measures how
    fast the machinery moves tokens, and the arithmetic per token does not depend on whether
    the tokens mean anything.
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, vocabulary_size + 1, dtype=np.float64)
    weights = ranks**-ZIPF_EXPONENT
    weights /= weights.sum()
    drawn = rng.choice(vocabulary_size, size=tokens, p=weights)
    words = np.array([f"w{i}" for i in range(vocabulary_size)], dtype=object)
    stream = words[drawn]
    return [list(stream[i : i + tokens_per_line]) for i in range(0, tokens, tokens_per_line)]


def measure_sparse_gradients(device: torch.device, vocabulary_size: int, dimension: int) -> str:
    """Report whether a sparse embedding gradient survives backward and an SGD step here."""
    matrix = torch.nn.Embedding(vocabulary_size, dimension, sparse=True).to(device)
    ids = torch.randint(0, vocabulary_size, (16_384,), device=device)
    try:
        matrix(ids).pow(2).sum().backward()
    except Exception as error:  # noqa: BLE001 - the point is to report, not to handle
        return f"backward failed: {type(error).__name__}: {error}"
    gradient = matrix.weight.grad
    if not gradient.is_sparse:
        return "backward produced a DENSE gradient"
    dense_megabytes = vocabulary_size * dimension * 4 / 1e6
    sparse_megabytes = gradient._values().numel() * 4 / 1e6
    optimiser = torch.optim.SGD(matrix.parameters(), lr=0.1)
    before = matrix.weight.detach().clone()
    try:
        optimiser.step()
    except Exception as error:  # noqa: BLE001
        return f"sparse gradient ok, SGD step failed: {type(error).__name__}: {error}"
    moved = int(((matrix.weight.detach() - before).abs().sum(dim=1) > 0).sum())
    return (
        f"sparse ok, {gradient._nnz()} rows touched, {moved} rows moved by SGD, "
        f"gradient {sparse_megabytes:.1f} MB against {dense_megabytes:.0f} MB dense"
    )


def measure_feeder(
    lines: list[list[str]], vocabulary: Vocabulary, config: SGNSConfig
) -> tuple[float, float, float]:
    """The Python side alone: no model, no optimiser.

    Returns in-vocabulary tokens per second, the share of tokens subsampling kept, and the
    number of training examples produced per token read.
    """
    keep = subsample_probabilities(vocabulary.counts, config.subsample_threshold)
    feeder = BatchFeeder(config, keep, seed=config.seed)
    started = time.perf_counter()
    examples = sum(len(batch) for batch in feeder.batches(encode_corpus(iter(lines), vocabulary)))
    elapsed = time.perf_counter() - started
    return (
        feeder.tokens_read / elapsed,
        feeder.tokens_kept / feeder.tokens_read,
        examples / feeder.tokens_read,
    )


def measure_training(
    lines: list[list[str]], vocabulary: Vocabulary, config: SGNSConfig
) -> tuple[float, float]:
    """In-vocabulary tokens per second for a full epoch, and the mean loss it reached."""
    trained = train_embeddings_from_lines(
        lambda: iter(lines), vocabulary, replace(config, epochs=1)
    )
    return trained.tokens_per_second, trained.epoch_losses[0]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokens", type=int, default=2_000_000)
    parser.add_argument("--vocabulary-size", type=int, default=100_000)
    parser.add_argument("--dimension", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--negative-samples", type=int, default=15)
    args = parser.parse_args(argv)

    devices = [torch.device("cpu")]
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    print(
        f"torch {torch.__version__}, devices {[d.type for d in devices]}, "
        f"{torch.get_num_threads()} CPU threads\n"
    )
    print("== Risk 1: does the sparse gradient path work here")
    for device in devices:
        print(
            f"  {device.type}: "
            f"{measure_sparse_gradients(device, args.vocabulary_size, args.dimension)}"
        )

    print(f"\n== Risk 2: is it the GPU or the Python, over {args.tokens:,} synthetic tokens")
    lines = synthetic_corpus(args.vocabulary_size, args.tokens)
    vocabulary = build_vocabulary(iter(lines), min_count=5, max_size=args.vocabulary_size)
    print(
        f"  corpus {sum(len(line) for line in lines):,} tokens, "
        f"vocabulary {len(vocabulary):,} words, dimension {args.dimension}, "
        f"k={args.negative_samples}, batch {args.batch_size}"
    )

    for objective in ("skipgram", "cbow"):
        base = replace(
            SGNSConfig(),
            objective=objective,
            dimension=args.dimension,
            batch_size=args.batch_size,
            negative_samples=args.negative_samples,
            vocabulary_cap=args.vocabulary_size,
        )
        feeder_rate, retention, examples_per_token = measure_feeder(lines, vocabulary, base)
        print(
            f"\n  {objective}: subsampling kept {retention:.1%} of tokens, "
            f"{examples_per_token:.2f} training examples per token"
        )
        print(f"    feeder only            {feeder_rate:>10,.0f} tokens/s")
        for device in devices:
            for sparse in (True, False):
                config = replace(base, device=device.type, sparse_gradients=sparse)
                rate, loss = measure_training(lines, vocabulary, config)
                share = 100.0 * rate / feeder_rate if feeder_rate else float("nan")
                label = "sparse" if sparse else "dense "
                print(
                    f"    {device.type:<3} {label} gradients {rate:>10,.0f} tokens/s "
                    f"({share:>2.0f}% of the feeder's ceiling, loss {loss:.3f})"
                )


if __name__ == "__main__":
    main()
