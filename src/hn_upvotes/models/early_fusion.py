"""Early fusion: concatenate every modality, then one MLP.

Four inputs: the pooled title vector, an author embedding, a domain embedding, and the
temporal features. Early fusion glues them into one vector at the input and lets a
single network find whatever interactions exist. It can learn that a given author
usually posts about a given topic, which the late fusion tower cannot.

The cost is that a weak modality can drag the whole representation, and there is no
per-modality output to inspect.

Needs the ``train`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class FusionDimensions:
    """Input widths, shared by all three fusion architectures."""

    title_dimension: int = 300
    author_vocabulary: int = 50_000
    author_dimension: int = 32
    domain_vocabulary: int = 20_000
    domain_dimension: int = 32
    temporal_dimension: int = 5


class EarlyFusion(nn.Module):
    """Concatenate title, author, domain and temporal inputs, then an MLP to a scalar."""

    def __init__(
        self,
        dimensions: FusionDimensions | None = None,
        hidden_sizes: tuple[int, ...] = (256, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions or FusionDimensions()
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout

    def forward(
        self,
        title_vectors: Tensor,
        author_ids: Tensor,
        domain_ids: Tensor,
        temporal_features: Tensor,
    ) -> Tensor:
        """Return one predicted normalised target per row, shape (batch,).

        Every input is (batch, ...) and rows are aligned. ``author_ids`` and
        ``domain_ids`` are integer indices into the learned tables, with rare values
        already bucketed to the out-of-vocabulary row by the encoders.
        """
        raise NotImplementedError

    def ablate(self, modality: str) -> EarlyFusion:
        """Return a copy with one modality zeroed, for the ablation table.

        ``modality`` is one of ``title``, ``author``, ``domain``, ``temporal``.
        """
        raise NotImplementedError
