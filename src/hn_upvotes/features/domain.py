"""Domain features: the registrable domain of the linked URL, plus its track record.

Same shape as the author features. The interesting difference is extraction: the useful
unit is the registrable domain, so ``news.ycombinator.com`` and ``www.nytimes.com``
become ``ycombinator.com`` and ``nytimes.com``, while ``github.com/user/repo`` stays at
``github.com``. Text posts have no URL and get their own bucket rather than being
dropped, because "no link" is itself a signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

#: Row reserved for domains below the minimum post count, and for unseen domains.
OOV_INDEX = 0

#: Row for text posts, which have no URL at all.
NO_URL_INDEX = 1


@dataclass(frozen=True)
class DomainStatsConfig:
    """Settings for the expanding domain statistics. Mirrors ``AuthorStatsConfig``."""

    window: timedelta | None = None
    lag: timedelta = timedelta(hours=48)
    min_posts: int = 5


def extract_registrable_domain(url: str | None) -> str | None:
    """Return the registrable domain of a URL, or ``None`` for a text post.

    Strips the scheme, any ``www.`` prefix, the port and the path. Multi-part public
    suffixes such as ``.co.uk`` need the public suffix list rather than a "last two
    labels" rule, which would collapse every UK site to ``co.uk``.
    """
    raise NotImplementedError


def expanding_domain_stats(
    times: pd.Series,
    domains: pd.Series,
    targets: pd.Series,
    config: DomainStatsConfig | None = None,
) -> pd.DataFrame:
    """Per-row domain track record, computed from strictly earlier posts only.

    Returns ``domain_prior_count``, ``domain_prior_mean`` and ``domain_prior_std``.
    """
    raise NotImplementedError


class DomainEncoder:
    """Maps domains to embedding table rows, fitted on the training split."""

    def __init__(self, min_posts: int = 20) -> None:
        self.min_posts = min_posts
        self.domain_to_index: dict[str, int] = {}

    def fit(self, domains: pd.Series) -> DomainEncoder:
        """Assign a row to every domain with at least ``min_posts`` training posts."""
        raise NotImplementedError

    def transform(self, domains: pd.Series) -> np.ndarray:
        """Map domains to row indices, with buckets for rare, unseen and absent."""
        raise NotImplementedError

    @property
    def vocabulary_size(self) -> int:
        """Number of embedding rows, including the two reserved rows."""
        raise NotImplementedError
