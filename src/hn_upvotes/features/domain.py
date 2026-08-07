"""Domain features: the hostname of the linked URL, plus its track record.

Same shape as the author features. The interesting difference is extraction. The key is
the hostname with any ``www.`` prefix removed, so ``https://www.nytimes.com/a/b`` and
``http://nytimes.com/c`` both give ``nytimes.com``, and ``news.ycombinator.com`` stays
whole.

It is the hostname, not the registrable domain, and the name says so. Collapsing
``news.ycombinator.com`` to ``ycombinator.com`` needs the public suffix list, because
the obvious "last two labels" rule turns every ``.co.uk`` site into ``co.uk``. That is a
dependency the baselines do not need: they want a stable key with a track record, and
the hostname is one. It also keeps ``blog.example.com`` and ``shop.example.com`` apart,
which for scoring is probably right.

Text posts have no URL and get their own bucket rather than being dropped, because "no
link" is itself a signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

#: Key for a post with no link at all. The dump already spells a missing URL this way
#: (missing values are sentinels, not NULL), so both spellings land in one bucket.
NO_URL_KEY = ""

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


def extract_hostname(url: str | None) -> str:
    """Return the lowercase hostname of a URL, or :data:`NO_URL_KEY` for a text post.

    Strips the scheme, any credentials, the port, the path and any ``www.`` prefix.
    Anything unparseable comes back as :data:`NO_URL_KEY` rather than raising, so one
    malformed row in five million cannot stop a run.
    """
    if not url or not isinstance(url, str):
        return NO_URL_KEY
    try:
        host = urlsplit(url if "//" in url else f"//{url}").hostname
    except ValueError:
        return NO_URL_KEY
    if not host:
        return NO_URL_KEY
    return host.removeprefix("www.")


def extract_hostnames(urls: pd.Series) -> pd.Series:
    """:func:`extract_hostname` over a column. Kept separate so there is one rule."""
    return pd.Series(urls).reset_index(drop=True).map(extract_hostname).astype(str)


def has_url(urls: pd.Series) -> np.ndarray:
    """1.0 where the post links somewhere, 0.0 where it is a text post.

    Kept separate from the hostname track record on purpose. A text post's hostname is
    the empty string, and without this flag a model cannot tell "no link" apart from "a
    link to a host with no history". The first is a fact about the post, the second is an
    absence of evidence.
    """
    return (extract_hostnames(urls) != NO_URL_KEY).to_numpy(dtype=np.float64)


def expanding_domain_stats(
    times: pd.Series,
    domains: pd.Series,
    targets: pd.Series,
    config: DomainStatsConfig | None = None,
) -> pd.DataFrame:
    """Per-row domain track record, as feature columns, from strictly earlier posts only.

    Returns ``domain_prior_count``, ``domain_prior_mean`` and ``domain_prior_std``.

    Not needed yet. The statistic itself is implemented and tested in
    ``features.history.prior_mean``, which rung 3 of the baseline ladder calls directly.
    This wrapper is the feature-column form the fusion models will want.
    """
    raise NotImplementedError("the statistic is implemented in features.history.prior_mean")


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
