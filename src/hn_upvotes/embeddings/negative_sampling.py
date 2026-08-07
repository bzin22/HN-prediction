"""The machinery CBOW and Skip-gram share: two matrices, the sampler, and the loss.

Both objectives are the same model. Two matrices, each vocabulary by dimension:

* the **input matrix**, one row per word, which is the embedding and the artefact
  everything downstream consumes,
* the **output matrix**, same size, used only for scoring and discarded when training
  ends.

The only difference between the two objectives is one masked-average step, which lives in
:mod:`hn_upvotes.embeddings.cbow`. Everything else is here, in one place, because a
sampler duplicated across two files is how the two drift apart and the comparison between
them stops meaning anything.

Needs the ``train`` extra. See ``docs/word2vec.md`` for the reasoning.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

#: Bound on the uniform initialisation of the input matrix, as a multiple of ``1 /
#: dimension``. gensim uses ``uniform(-0.5 / dimension, 0.5 / dimension)`` for the input
#: matrix and zeros for the output matrix, verified in its ``word2vec.py``. The gate for
#: this implementation is matching gensim, so the initialisation matches too.
_INPUT_INIT_SCALE = 0.5


def build_noise_distribution(counts: np.ndarray, power: float = 0.75) -> np.ndarray:
    """Unigram distribution raised to ``power``, normalised.

    Sampled from directly rather than through the paper's table trick. Modern
    ``torch.multinomial`` is fast enough that the table is not worth the extra state.

    The 0.75 power flattens the distribution. On a corpus where one word is 100 times as
    frequent as another, ``100 ** 0.75 = 31.6``, so the frequent word is drawn as a
    negative about 32 times as often rather than 100 times. Rare words get seen as
    negatives more than their raw frequency allows, which is what the paper tuned it for.
    """
    weights = np.asarray(counts, dtype=np.float64) ** power
    total = weights.sum()
    if total <= 0:
        raise ValueError("noise distribution is all zeros: counts sum to 0")
    return weights / total


def subsample_probabilities(counts: np.ndarray, threshold: float = 1e-5) -> np.ndarray:
    """Per-word keep probability under the frequent-word subsampling rule.

    ``P(keep) = min(1, sqrt(t / f))`` where ``f`` is the word's corpus frequency. Drops
    most occurrences of "the" while leaving rare words untouched.

    This is the paper's formula. **gensim uses a different one**: ``sqrt(t / f) + t / f``,
    which is strictly more permissive. The two agree to within a rounding error on very
    frequent words and diverge in the middle of the distribution. At ``f = 4e-5`` and
    ``t = 1e-5`` the paper keeps ``sqrt(0.25) = 0.50`` and gensim keeps
    ``0.50 + 0.25 = 0.75``. ``docs/word2vec.md`` records this as a second thing the
    gensim comparison has to account for, alongside gensim's ``sample`` default.
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("cannot compute subsampling probabilities: counts sum to 0")
    frequency = counts / total
    with np.errstate(divide="ignore", invalid="ignore"):
        keep = np.sqrt(threshold / frequency)
    # A count of zero cannot occur in the corpus, so it can never be sampled anyway.
    keep[~np.isfinite(keep)] = 1.0
    return np.minimum(keep, 1.0)


def sample_negatives(noise: Tensor, batch_size: int, k: int, generator: torch.Generator) -> Tensor:
    """Draw ``k`` negative word ids per positive, shape (batch, k).

    Drawn with replacement, so a word can appear twice among one positive's negatives.
    The paper does the same and the duplicate simply doubles that word's push away.
    """
    draws = torch.multinomial(noise, batch_size * k, replacement=True, generator=generator)
    return draws.view(batch_size, k)


class NegativeSampler:
    """Draws negative word ids from the unigram counts raised to ``noise_power``.

    Holds the noise distribution on the training device so the draw does not cross the
    host boundary every step, and owns its own generator so a run is reproducible from
    the seed alone.
    """

    def __init__(
        self,
        counts: np.ndarray,
        noise_power: float = 0.75,
        device: torch.device | None = None,
        seed: int = 0,
    ) -> None:
        device = device or torch.device("cpu")
        self.device = device
        self.noise = torch.as_tensor(
            build_noise_distribution(counts, noise_power), dtype=torch.float32, device=device
        )
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def draw(self, batch_size: int, k: int) -> Tensor:
        """Return (batch, k) negative ids."""
        return sample_negatives(self.noise, batch_size, k, self.generator)


class NegativeSamplingObjective(nn.Module):
    """Two matrices and the negative-sampling loss over them.

    Not used directly. :class:`~hn_upvotes.embeddings.cbow.CBOWObjective` and
    :class:`~hn_upvotes.embeddings.skipgram.SkipGramObjective` subclass it and supply only
    a ``forward``, which decides what vector gets scored against what word.

    A full softmax over a 100,000-word vocabulary asks "which of 100,000 words is it",
    which costs a dot product against every row. Negative sampling asks 16 yes/no
    questions instead: is this the real word, and are these 15 drawn words not it. With
    ``k = 15`` that is 16 dot products per example rather than 100,000, so the cost stops
    depending on vocabulary size.
    """

    def __init__(
        self,
        vocabulary_size: int,
        dimension: int = 300,
        sparse_gradients: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension
        self.input_matrix = nn.Embedding(vocabulary_size, dimension, sparse=sparse_gradients)
        self.output_matrix = nn.Embedding(vocabulary_size, dimension, sparse=sparse_gradients)
        self.reset_parameters(seed)

    def reset_parameters(self, seed: int = 0) -> None:
        """Initialise as gensim does: small uniform input matrix, zero output matrix.

        The zero output matrix has a useful consequence for testing. Every logit starts at
        exactly 0, so the first loss is ``(1 + k) * log 2`` whatever the corpus is. At
        ``k = 15`` that is ``16 * 0.6931 = 11.09``.
        """
        generator = torch.Generator().manual_seed(seed)
        bound = _INPUT_INIT_SCALE / self.dimension
        with torch.no_grad():
            self.input_matrix.weight.uniform_(-bound, bound, generator=generator)
            self.output_matrix.weight.zero_()

    def negative_sampling_loss(
        self,
        hidden: Tensor,
        positive_ids: Tensor,
        negative_ids: Tensor,
    ) -> Tensor:
        """Score ``hidden`` against one true word and ``k`` drawn words, and return a loss.

        ``hidden`` is (batch, dimension) and is whatever the objective decided to score:
        the averaged context for CBOW, the centre word's own vector for Skip-gram.
        ``positive_ids`` is (batch,), ``negative_ids`` is (batch, k). Both index the output
        matrix.

        Per example the loss is ``-log sigmoid(positive) - sum_j log sigmoid(-negative_j)``,
        which is the paper's objective. Computed as binary cross-entropy **with logits**, so
        the log and the sigmoid stay fused and a large negative logit does not underflow
        before the log sees it. The batch reduction is the mean, so the reported number is
        comparable across batch sizes.
        """
        positive_vectors = self.output_matrix(positive_ids)
        negative_vectors = self.output_matrix(negative_ids)

        positive_logits = (hidden * positive_vectors).sum(-1, keepdim=True)
        negative_logits = torch.bmm(negative_vectors, hidden.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([positive_logits, negative_logits], dim=1)
        targets = torch.zeros_like(logits)
        targets[:, 0] = 1.0
        per_term = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # A negative drawn from the noise distribution can land on the true word, which
        # would ask the model to push a word away from itself. gensim drops that draw
        # rather than resampling, so this drops it too: that example just gets k-1
        # negatives. At k=15 and a word at 3% of the corpus it costs 0.45 draws per
        # example, and for everything rarer it is negligible.
        keep = torch.ones_like(per_term)
        keep[:, 1:] = (negative_ids != positive_ids.unsqueeze(1)).to(per_term.dtype)

        return (per_term * keep).sum(1).mean()

    def input_embeddings(self, average_with_context: bool = False) -> Tensor:
        """Return the embedding matrix, (vocabulary, dimension).

        This is the artefact the rest of the project consumes. With
        ``average_with_context`` the input and output matrices are averaged instead, which
        is a cheap variant worth measuring rather than a different model.
        """
        with torch.no_grad():
            if average_with_context:
                return (self.input_matrix.weight + self.output_matrix.weight).div(2.0)
            return self.input_matrix.weight.detach().clone()

    def to_device(self, device: torch.device) -> Self:
        """Move to a device. MPS on this machine, CPU in CI."""
        self.to(device)
        return self
