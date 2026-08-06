"""CBOW with negative sampling.

CBOW is a training *objective*, not an embedding. The model averages the vectors of the
context words and scores that average against the centre word. Once training finishes
the task is thrown away and the input weight matrix is kept, one row per word. That
matrix is the embedding.

Two matrices are learned, centre and context. Convention keeps the input (centre) one.
Averaging the two is a cheap variant and ``input_embeddings`` takes a flag for it.

Needs the ``train`` extra.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class CBOWObjective(nn.Module):
    """Predict the centre word from the mean of its context word vectors.

    Trained with negative sampling rather than a full softmax: a 100k vocabulary makes
    the softmax denominator the whole cost of the model.
    """

    def __init__(self, vocabulary_size: int, dimension: int = 300) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension

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
        """
        raise NotImplementedError

    def input_embeddings(self, average_with_context: bool = False) -> Tensor:
        """Return the embedding matrix, (vocabulary, dimension).

        This is the artefact the rest of the project consumes. With
        ``average_with_context`` the centre and context matrices are averaged instead.
        """
        raise NotImplementedError

    def to_device(self, device: torch.device) -> CBOWObjective:
        """Move to a device. MPS on this machine, CPU in CI."""
        raise NotImplementedError
