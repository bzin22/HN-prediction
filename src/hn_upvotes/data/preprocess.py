"""Cleaning, tokenisation, and the vocabulary shared by the embedding objectives."""

from __future__ import annotations

import html
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Token pattern. Keeps intra-word apostrophes and hyphens ("don't", "self-hosted") and
#: keeps digits attached to letters ("gpt-4", "s3", "3d"), because those are content
#: words on Hacker News rather than noise. Everything else splits.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")

#: Collapses any run of whitespace, including the newlines that survive an unescape.
_WHITESPACE_RE = re.compile(r"\s+")

#: The out-of-vocabulary token. Not a word that occurs in the corpus.
UNKNOWN_TOKEN = "<unk>"


@dataclass(frozen=True)
class Vocabulary:
    """Word to index mapping plus the counts the negative sampler needs.

    ``counts`` is aligned with ``index_to_word`` and holds raw corpus frequencies, not
    probabilities. The 0.75 power and the subsampling threshold are applied by
    ``embeddings.train``, so the same vocabulary can serve different sampler settings.
    """

    word_to_index: dict[str, int]
    index_to_word: list[str]
    counts: np.ndarray
    unknown_index: int

    def __len__(self) -> int:
        return len(self.index_to_word)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        """Map tokens to indices, sending out-of-vocabulary tokens to ``unknown_index``."""
        lookup = self.word_to_index
        unknown = self.unknown_index
        return [lookup.get(token, unknown) for token in tokens]

    def decode(self, indices: Iterable[int]) -> list[str]:
        """Map indices back to words."""
        table = self.index_to_word
        return [table[int(i)] for i in indices]

    def save(self, path: Path) -> None:
        """Write the vocabulary so the serving image can load it without the corpus."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "index_to_word": self.index_to_word,
                    "counts": [int(c) for c in self.counts],
                    "unknown_index": int(self.unknown_index),
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> Vocabulary:
        """Read a vocabulary written by :meth:`save`."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        index_to_word = list(payload["index_to_word"])
        return cls(
            word_to_index={word: i for i, word in enumerate(index_to_word)},
            index_to_word=index_to_word,
            counts=np.asarray(payload["counts"], dtype=np.int64),
            unknown_index=int(payload["unknown_index"]),
        )


def filter_stories(frame: pd.DataFrame, min_title_tokens: int = 1) -> pd.DataFrame:
    """Drop rows that cannot be scored or cannot be featurised.

    Removes deleted and dead items, null titles, null scores, and titles shorter than
    ``min_title_tokens``. Returns a copy, sorted by time.

    Missing values in the source dump are sentinels rather than SQL ``NULL``: an absent
    title is ``''`` and an absent score is ``0``. Both spellings are dropped, so this
    behaves the same on a raw shard and on a frame that has been through pandas.
    ``deleted`` and ``dead`` are optional, because the ingested table no longer carries
    them: they have already done their job.
    """
    keep = pd.Series(True, index=frame.index)

    for flag in ("deleted", "dead"):
        if flag in frame.columns:
            keep &= pd.to_numeric(frame[flag], errors="coerce").fillna(0) == 0

    title = frame["title"].where(frame["title"].notna(), "").astype(str).str.strip()
    keep &= title != ""

    score = pd.to_numeric(frame["score"], errors="coerce")
    keep &= score.notna() & (score > 0)

    if min_title_tokens > 1:
        token_counts = title.map(lambda t: len(tokenise(normalise_title(t))))
        keep &= token_counts >= min_title_tokens

    kept = frame.loc[keep].copy()
    if "time" in kept.columns:
        kept = kept.sort_values("time", kind="stable")
    return kept.reset_index(drop=True)


def normalise_title(title: str) -> str:
    """Lowercase and strip a title before tokenisation.

    Handles the HTML entities the API leaves in place (``&#x27;`` and friends) and
    collapses whitespace. Deliberately keeps punctuation for the tokeniser to decide on.

    Unescaped twice. The dump contains double-escaped entities such as ``&amp;#x27;``,
    and one pass leaves those as a literal ``&#x27;``.
    """
    if title is None or (isinstance(title, float) and pd.isna(title)):
        return ""
    text = html.unescape(html.unescape(str(title)))
    return _WHITESPACE_RE.sub(" ", text).strip().lower()


def tokenise(title: str) -> list[str]:
    """Split a normalised title into tokens.

    The dump ships a pre-tokenised ``words`` column, which Phase 1 compared against this
    function rather than trusting either blindly. The project uses this tokeniser.
    Measured on ``2026-06``, the reasons are:

    * ``words`` tokenises ``text``, not ``title``. It is populated for 2,474 of the
      30,102 titled stories that month (8.2%), and those are the 2,446 Ask HN posts that
      have a body plus 28 edge cases. A link story with a title and a URL has no
      ``words`` at all, and link stories are 92% of the corpus. The title is the one
      field this project needs tokenised.
    * ``words`` is sorted alphabetically and deduplicated: a set, not a sequence. CBOW
      and Skip-gram are both defined over a context window, which needs word order.

    On the input the two do share (``text``, 5,000 rows sampled), they agree to a micro
    Jaccard of 0.899 and match exactly on 32.9% of rows. The disagreements are three
    classes, none of them a reason to prefer ``words`` for titles:

    * Contractions, 42.1%. ``words`` splits ``don't`` into ``don`` and ``t``; this keeps
      it whole.
    * HTML markup, 34.0%. ``words`` strips tags before tokenising; this does not, so it
      emits ``href``, ``rel`` and ``nofollow`` from a comment body. This is a real
      weakness of this tokeniser on ``text``, and it does not arise on titles: titles
      carry entities, not tags, and :func:`normalise_title` unescapes them.
    * Hyphens, 18.2%. ``words`` splits ``ad-free`` into ``ad`` and ``free``; this keeps
      it whole. On Hacker News that matters: ``gpt-4`` and ``self-hosted`` are single
      content words.

    See ``docs/design.md`` for the full comparison.

    Input is expected to have been through :func:`normalise_title`. Passing a raw title
    still works, it just leaves the HTML entities in.
    """
    return _TOKEN_RE.findall(title.lower())


def stream_corpus(path: Path) -> Iterator[list[str]]:
    """Yield tokenised lines from a plain-text corpus such as ``text8``.

    Streaming rather than loading, because the Wikipedia subset does not need to sit in
    memory alongside a training run on 24 GB.

    ``text8`` is one 100 MB line, so lines are read in fixed-size chunks and split on a
    whitespace boundary. A chunk never splits a word in half.
    """
    path = Path(path)
    chunk_size = 1 << 20
    remainder = ""
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            buffered = remainder + chunk
            # Hold back the trailing fragment: it may be the first half of a word.
            split = buffered.rfind(" ")
            if split == -1:
                remainder = buffered
                continue
            remainder = buffered[split + 1 :]
            tokens = tokenise(buffered[:split])
            if tokens:
                yield tokens
    tokens = tokenise(remainder)
    if tokens:
        yield tokens


def build_vocabulary(
    token_lists: Iterable[list[str]],
    min_count: int = 5,
    max_size: int | None = 100_000,
) -> Vocabulary:
    """Count tokens and build the vocabulary.

    Words below ``min_count`` fold into the unknown token. ``max_size`` caps the table
    because a full softmax is already off the table and the embedding matrix dominates
    memory.

    The unknown token is index 0 and its count is the total frequency of everything that
    folded into it, so the negative sampler weights it correctly rather than treating it
    as a rare word.
    """
    counter: Counter[str] = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    # Sort by count descending, then alphabetically, so the table is deterministic.
    ranked = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    kept = [(w, c) for w, c in ranked if c >= min_count]
    if max_size is not None:
        # max_size includes the unknown token, which occupies index 0.
        kept = kept[: max(max_size - 1, 0)]

    kept_words = {w for w, _ in kept}
    unknown_count = sum(c for w, c in counter.items() if w not in kept_words)

    index_to_word = [UNKNOWN_TOKEN] + [w for w, _ in kept]
    counts = np.asarray([unknown_count] + [c for _, c in kept], dtype=np.int64)
    return Vocabulary(
        word_to_index={word: i for i, word in enumerate(index_to_word)},
        index_to_word=index_to_word,
        counts=counts,
        unknown_index=0,
    )
