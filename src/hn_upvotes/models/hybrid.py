"""Hybrid fusion: encode each modality, concatenate the encodings, then a joint head.

Sits between the other two. Each modality gets its own encoder, so a wide raw input
cannot swamp a narrow one, but the head still sees all of them together and can learn
interactions. This is what production recommender and ranking systems usually do, which
is the reason it is in the comparison.

Needs the ``train`` extra.
"""

from __future__ import annotations

from torch import Tensor, nn

from hn_upvotes.models.early_fusion import FusionDimensions


class HybridFusion(nn.Module):
    """Per-modality encoders into a shared width, concatenated, then a joint head."""

    def __init__(
        self,
        dimensions: FusionDimensions | None = None,
        encoded_dimension: int = 64,
        head_hidden: tuple[int, ...] = (128,),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions or FusionDimensions()
        self.encoded_dimension = encoded_dimension
        self.head_hidden = head_hidden
        self.dropout = dropout

    def forward(
        self,
        title_vectors: Tensor,
        author_ids: Tensor,
        domain_ids: Tensor,
        temporal_features: Tensor,
    ) -> Tensor:
        """Return one predicted normalised target per row, shape (batch,)."""
        raise NotImplementedError

    def encode(
        self,
        title_vectors: Tensor,
        author_ids: Tensor,
        domain_ids: Tensor,
        temporal_features: Tensor,
    ) -> dict[str, Tensor]:
        """Return each modality's encoded representation, each (batch, encoded_dimension).

        Same width for every modality, so the concatenation does not implicitly weight
        one of them by being wider.
        """
        raise NotImplementedError

    def ablate(self, modality: str) -> HybridFusion:
        """Return a copy with one modality's encoder output zeroed."""
        raise NotImplementedError
