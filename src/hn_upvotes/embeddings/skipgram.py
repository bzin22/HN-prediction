"""Skip-gram with negative sampling.

The mirror of CBOW: predict each context word from the centre word. Same caveat about
terminology. Skip-gram is the training objective, and the embedding is the input weight
matrix that survives after the task is discarded.

Skip-gram generates one training pair per context position instead of one per window, so
it sees more updates per token, trains slower, and generally does better on rare words.
Comparing the two on the downstream task is one of the experiments this project runs.

Everything except the forward pass lives in
:mod:`hn_upvotes.embeddings.negative_sampling`, which this shares with CBOW.

Needs the ``train`` extra.
"""

from __future__ import annotations

from torch import Tensor

from hn_upvotes.embeddings.negative_sampling import NegativeSamplingObjective


class SkipGramObjective(NegativeSamplingObjective):
    """Predict context words from the centre word, with negative sampling."""

    def forward(
        self,
        centre_ids: Tensor,
        context_ids: Tensor,
        negative_ids: Tensor,
    ) -> Tensor:
        """Return the mean negative-sampling loss over the batch.

        Shapes: ``centre_ids`` and ``context_ids`` are (batch,) paired positives,
        ``negative_ids`` is (batch, k).

        No averaging step, so no mask. The window has already been flattened into one
        pair per context position by the feeder, and a padded position produces no pair at
        all rather than a masked one.
        """
        hidden = self.input_matrix(centre_ids)
        return self.negative_sampling_loss(hidden, context_ids, negative_ids)
