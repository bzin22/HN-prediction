"""CBOW with negative sampling.

CBOW is a training *objective*, not an embedding. The model averages the vectors of the
context words and scores that average against the centre word. Once training finishes
the task is thrown away and the input weight matrix is kept, one row per word. That
matrix is the embedding.

Two matrices are learned, centre and context. Convention keeps the input (centre) one.
Averaging the two is a cheap variant and ``input_embeddings`` takes a flag for it.

Everything except the forward pass lives in
:mod:`hn_upvotes.embeddings.negative_sampling`, which this shares with Skip-gram. The one
thing that is only CBOW's is :func:`masked_average`.

Needs the ``train`` extra.
"""

from __future__ import annotations

from torch import Tensor

from hn_upvotes.embeddings.negative_sampling import NegativeSamplingObjective


def masked_average(vectors: Tensor, mask: Tensor) -> Tensor:
    """Average (batch, positions, dimension) over the positions the mask marks as real.

    The dynamic window makes contexts ragged, so the padded positions have to be excluded
    rather than averaged in. Padding uses word id 0, which is the unknown token and a real
    row of the matrix, so getting this wrong would not crash: it would quietly average the
    unknown vector into every short context and divide by the wrong count.

    Worked example. A context of 2 real words out of a padded width of 10, with vectors
    ``a`` and ``b``: this returns ``(a + b) / 2``. Averaging the padded row instead returns
    ``(a + b + 8 * unknown) / 10``, which is a different vector even when ``unknown`` is
    zero, because the divisor is 10 rather than 2.

    A row with no real positions at all averages to zeros rather than dividing by zero.
    """
    weights = mask.to(vectors.dtype).unsqueeze(-1)
    total = (vectors * weights).sum(dim=1)
    count = weights.sum(dim=1).clamp(min=1.0)
    return total / count


class CBOWObjective(NegativeSamplingObjective):
    """Predict the centre word from the mean of its context word vectors.

    Trained with negative sampling rather than a full softmax: a 100k vocabulary makes
    the softmax denominator the whole cost of the model.

    One training example per position, against Skip-gram's up to ``2 * window``. On
    text8's roughly 17 million tokens that is 17 million examples per epoch against up to
    170 million, so CBOW is around ten times faster and worse on rare words.
    """

    def forward(
        self,
        context_ids: Tensor,
        centre_ids: Tensor,
        negative_ids: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        """Return the mean negative-sampling loss over the batch.

        Shapes: ``context_ids`` is (batch, context), ``centre_ids`` is (batch,),
        ``negative_ids`` is (batch, k). ``context_mask`` marks real context positions,
        which matters because the dynamic window makes contexts ragged.

        Two lines. The first is the whole difference between CBOW and Skip-gram.
        """
        context_vectors = self.input_matrix(context_ids)
        if context_mask is None:
            hidden = context_vectors.mean(dim=1)
        else:
            hidden = masked_average(context_vectors, context_mask)
        return self.negative_sampling_loss(hidden, centre_ids, negative_ids)
