"""Late fusion: one tower per modality to its own scalar, then a learned combination.

Each modality produces its own prediction and the model learns how to weight them. It
cannot learn cross-modal interaction in the body, which is the point of comparing it
with early fusion. In exchange it is interpretable, because every tower's scalar is a
readable per-modality opinion, and it degrades gracefully when a modality is missing:
a text post with no URL just drops the domain term.

Needs the ``train`` extra.
"""

from __future__ import annotations

from torch import Tensor, nn

from hn_upvotes.models.early_fusion import FusionDimensions


class LateFusion(nn.Module):
    """Per-modality towers to scalars, combined by a learned weighting."""

    def __init__(
        self,
        dimensions: FusionDimensions | None = None,
        tower_hidden: tuple[int, ...] = (64,),
        dropout: float = 0.2,
        learned_gate: bool = True,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions or FusionDimensions()
        self.tower_hidden = tower_hidden
        self.dropout = dropout
        self.learned_gate = learned_gate

    def forward(
        self,
        title_vectors: Tensor,
        author_ids: Tensor,
        domain_ids: Tensor,
        temporal_features: Tensor,
        modality_mask: Tensor | None = None,
    ) -> Tensor:
        """Return one predicted normalised target per row, shape (batch,).

        ``modality_mask`` is (batch, 4) and marks which modalities are present. Masked
        towers are dropped from the combination and the weights renormalise, which is
        the graceful-degradation property.
        """
        raise NotImplementedError

    def modality_scalars(
        self,
        title_vectors: Tensor,
        author_ids: Tensor,
        domain_ids: Tensor,
        temporal_features: Tensor,
    ) -> dict[str, Tensor]:
        """Return each tower's scalar separately, for interpretation.

        This is what late fusion buys and the other two architectures cannot offer.
        """
        raise NotImplementedError
