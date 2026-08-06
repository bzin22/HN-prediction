"""Skip-gram with negative sampling.

The mirror of CBOW: predict each context word from the centre word. Same caveat about
terminology. Skip-gram is the training objective, and the embedding is the input weight
matrix that survives after the task is discarded.

Skip-gram generates one training pair per context position instead of one per window, so
it sees more updates per token, trains slower, and generally does better on rare words.
Comparing the two on the downstream task is one of the experiments this project runs.

Needs the ``train`` extra.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SkipGramObjective(nn.Module):
    """Predict context words from the centre word, with negative sampling."""

    def __init__(self, vocabulary_size: int, dimension: int = 300) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.dimension = dimension

    def forward(
        self,
        centre_ids: Tensor,
        context_ids: Tensor,
        negative_ids: Tensor,
    ) -> Tensor:
        """Return the mean negative-sampling loss over the batch.

        Shapes: ``centre_ids`` and ``context_ids`` are (batch,) paired positives,
        ``negative_ids`` is (batch, k).
        """
        raise NotImplementedError

    def input_embeddings(self, average_with_context: bool = False) -> Tensor:
        """Return the embedding matrix, (vocabulary, dimension)."""
        raise NotImplementedError

    def to_device(self, device: torch.device) -> SkipGramObjective:
        """Move to a device. MPS on this machine, CPU in CI."""
        raise NotImplementedError
