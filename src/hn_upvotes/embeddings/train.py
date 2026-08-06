"""Training harness for both word2vec objectives.

Hyperparameter defaults come from Mikolov et al. 2013, "Distributed Representations of
Words and Phrases and their Compositionality". The paper recommends 5 to 20 negative
samples for small corpora and 2 to 5 for large ones, and reports the unigram
distribution raised to the 0.75 power as the best of the noise distributions it tried.
Subsampling of frequent words uses the paper's ``t = 1e-5`` rule.

Needs the ``train`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from hn_upvotes.data.preprocess import Vocabulary


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
    """

    objective: Literal["cbow", "skipgram"] = "skipgram"
    dimension: int = 300
    window: int = 5
    negative_samples: int = 15
    noise_power: float = 0.75
    subsample_threshold: float = 1e-5
    dynamic_window: bool = True
    min_count: int = 5
    epochs: int = 5
    batch_size: int = 1024
    learning_rate: float = 2.5e-3
    seed: int = 0


@dataclass(frozen=True)
class TrainedEmbeddings:
    """The artefact that leaves this module: a matrix and the vocabulary to index it."""

    matrix: np.ndarray
    vocabulary: Vocabulary
    config: SGNSConfig
    tokens_per_second: float

    def save(self, path: Path) -> None:
        """Write matrix and vocabulary together, so they cannot drift apart."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> TrainedEmbeddings:
        """Read embeddings written by :meth:`save`."""
        raise NotImplementedError


def build_noise_distribution(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    """Unigram distribution raised to ``power``, normalised.

    Sampled from directly rather than through the paper's table trick. Modern
    ``torch.multinomial`` is fast enough that the table is not worth the extra state.
    """
    raise NotImplementedError


def subsample_probabilities(counts: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """Per-word keep probability under the frequent-word subsampling rule.

    ``P(keep) = min(1, sqrt(t / f))`` where ``f`` is the word's corpus frequency. Drops
    most occurrences of "the" while leaving rare words untouched.
    """
    raise NotImplementedError


def select_device() -> torch.device:
    """Pick MPS when available, otherwise CPU.

    No CUDA branch. The hardware for this project is an Apple M2 with 24 GB of unified
    memory, and CI runs on CPU.
    """
    raise NotImplementedError


def sample_negatives(noise: Tensor, batch_size: int, k: int, generator: torch.Generator) -> Tensor:
    """Draw ``k`` negative word ids per positive, shape (batch, k)."""
    raise NotImplementedError


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
    """
    raise NotImplementedError
