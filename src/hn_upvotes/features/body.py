"""Body text features: the text under the headline, its length, and whether it exists.

``text`` is Hacker News's own name for the body the poster writes under the title. Most
posts do not have one: 11.6% of stories carry body text, measured on the full table.

Three things come out of the column, and the second and third matter as much as the
first:

* **The words**, vectorised separately from the title. A headline and a paragraph of
  explanation are different registers, and merging them into one bag of words throws
  that away.
* **The length**, as its own number. The measured effect is largely a length effect: a
  link submission carrying over 1,000 characters of body reaches the top 5% of scores
  8.1% of the time against 5.1% for a bare link. A sparse bag of words recovers that
  slowly if at all, so it is handed over directly.
* **Whether there is any body at all**, as a flag. 88.4% of rows have none. Without the
  flag an empty body is an all-zero row, which is also what a body of two common words
  looks like after vectorising, and the two are not the same thing.

The body arrives as HTML, because that is what the API returns: paragraph breaks are
``<p>``, links are ``<a href="...">``, and quotes and ampersands are escaped entities.
:func:`strip_html` removes the tags and unescapes the entities before anything is counted,
so the vocabulary is words rather than markup and a length is a length of prose.
"""

from __future__ import annotations

import html
import re

import numpy as np
import pandas as pd

#: Feature columns this module produces, in a fixed order the models rely on.
BODY_FEATURE_NAMES: tuple[str, ...] = ("body_length", "has_body_text")

#: HTML tags, including the ``<p>`` that stands in for a paragraph break.
_TAG = re.compile(r"<[^>]*>")

#: Runs of whitespace, left behind once the tags are gone.
_WHITESPACE = re.compile(r"\s+")


def strip_html(text: str | None) -> str:
    """Turn one API body into plain text.

    Tags become spaces rather than nothing, so ``one<p>two`` gives ``one two`` and not
    ``onetwo``. Entities are unescaped afterwards, so an escaped ``&lt;`` in the original
    prose survives as a literal ``<`` instead of being read as the start of a tag.
    """
    if not text or not isinstance(text, str):
        return ""
    return _WHITESPACE.sub(" ", html.unescape(_TAG.sub(" ", text))).strip()


def strip_html_column(texts: pd.Series) -> pd.Series:
    """:func:`strip_html` over a column. Kept separate so there is one rule."""
    return pd.Series(texts).reset_index(drop=True).map(strip_html).astype(str)


def build_body_features(plain_text: pd.Series) -> pd.DataFrame:
    """Length and presence, in ``BODY_FEATURE_NAMES`` order, from already-stripped text.

    Length is ``log1p`` of the character count, for the same reason the target is: raw
    lengths run from 0 to tens of thousands and the interesting differences are at the
    short end.
    """
    lengths = pd.Series(plain_text).reset_index(drop=True).str.len().to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "body_length": np.log1p(lengths),
            "has_body_text": (lengths > 0).astype(np.float64),
        },
        columns=list(BODY_FEATURE_NAMES),
    )
