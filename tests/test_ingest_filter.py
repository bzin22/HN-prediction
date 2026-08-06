"""Ingest filtering and the ``int8`` type decode.

No network and no data file. The fixtures restate the source dump's encoding as it was
measured in Phase 1, so a change to that encoding turns these red rather than passing
silently.
"""

from __future__ import annotations

import pandas as pd
import pytest

from hn_upvotes.data.ingest import (
    ITEM_TYPE_BY_CODE,
    PROJECTED_COLUMNS,
    STORY_TYPE_CODE,
    decode_item_type,
    find_story_type_code,
)
from hn_upvotes.data.preprocess import filter_stories, normalise_title, tokenise
from hn_upvotes.features.schema import (
    LeakedFeatureError,
    leaked_columns_present,
    validate_feature_set,
)


@pytest.fixture
def raw_items() -> pd.DataFrame:
    """A synthetic shard in the source dump's own encoding.

    Missing values are sentinels, not NULL: an absent title is ``''`` and an absent
    score is ``0``. Measured on ``data/2025/2025-06.parquet``, where ``count(title)``
    returns the full row count on a table of 345,905 comments.
    """
    return pd.DataFrame(
        [
            # Four keepers.
            ("keep: ordinary story", 1, 0, 0, "Rust 1.90 released", 42, 7),
            ("keep: score of exactly 1", 1, 0, 0, "My side project", 1, 0),
            ("keep: text post, no url", 1, 0, 0, "Ask HN: how do you test?", 15, 22),
            ("keep: high score", 1, 0, 0, "Show HN: a thing", 2439, 310),
            # Dropped: wrong type. Polls and jobs carry a title and a score too.
            ("drop: comment", 2, 0, 0, "", 0, 0),
            ("drop: poll", 3, 0, 0, "Poll: which editor?", 5, 9),
            ("drop: pollopt", 4, 0, 0, "", 3, 0),
            ("drop: job", 5, 0, 0, "Acme (YC W25) Is Hiring", 1, 0),
            # Dropped: flagged. Every dead story in 2025-06 had an empty title as well,
            # but the flag is checked on its own so a future dead-with-title row dies.
            ("drop: dead", 1, 1, 0, "Spam title", 1, 0),
            ("drop: dead with a real score", 1, 1, 0, "Flagged but scored", 56, 3),
            ("drop: deleted", 1, 0, 1, "Removed by author", 8, 1),
            # Dropped: unusable. 591 of the 24,435 live stories in 2025-06 look like this.
            ("drop: empty title", 1, 0, 0, "", 12, 2),
            ("drop: whitespace title", 1, 0, 0, "   ", 12, 2),
            ("drop: zero score", 1, 0, 0, "Never scored", 0, 0),
            ("drop: null title", 1, 0, 0, None, 12, 2),
            ("drop: null score", 1, 0, 0, "No score at all", None, 0),
        ],
        columns=["note", "type", "dead", "deleted", "title", "score", "descendants"],
    ).assign(
        id=lambda f: range(1, len(f) + 1),
        time=lambda f: pd.to_datetime("2025-06-01") + pd.to_timedelta(range(len(f)), unit="h"),
        by=lambda f: [f"user{i}" for i in range(len(f))],
        url="https://example.com",
    )


def test_filter_keeps_only_usable_stories(raw_items: pd.DataFrame) -> None:
    """Four of the sixteen rows survive, and they are the four marked keep."""
    stories = raw_items[raw_items["type"] == STORY_TYPE_CODE]
    kept = filter_stories(stories)

    assert len(kept) == 4
    assert set(kept["note"]) == {
        "keep: ordinary story",
        "keep: score of exactly 1",
        "keep: text post, no url",
        "keep: high score",
    }
    assert (kept["score"] > 0).all()
    assert (kept["title"].str.strip() != "").all()


def test_filter_drops_each_reason_independently(raw_items: pd.DataFrame) -> None:
    """Every drop reason is exercised by at least one row, and none of them survives."""
    kept_notes = set(filter_stories(raw_items[raw_items["type"] == STORY_TYPE_CODE])["note"])
    dropped = [n for n in raw_items["note"] if n.startswith("drop:")]
    assert dropped, "fixture must contain rows that get dropped"
    assert not (kept_notes & set(dropped))


def test_filter_sorts_by_time_and_resets_index(raw_items: pd.DataFrame) -> None:
    shuffled = raw_items.sample(frac=1.0, random_state=0)
    kept = filter_stories(shuffled[shuffled["type"] == STORY_TYPE_CODE])
    assert kept["time"].is_monotonic_increasing
    assert list(kept.index) == list(range(len(kept)))


def test_filter_works_without_the_flag_columns(raw_items: pd.DataFrame) -> None:
    """The ingested table drops ``dead`` and ``deleted``, so the filter must not need them."""
    stories = raw_items[raw_items["type"] == STORY_TYPE_CODE].drop(columns=["dead", "deleted"])
    kept = filter_stories(stories)
    # The three flagged rows now survive, because nothing records that they were flagged.
    assert len(kept) == 7


def test_min_title_tokens(raw_items: pd.DataFrame) -> None:
    """Three of the four keepers reach four tokens. Token counts, not word counts.

    "Rust 1.90 released" is 4: the decimal point splits ``1.90`` into ``1`` and ``90``.
    "Ask HN: how do you test?" is 6. "Show HN: a thing" is 4. "My side project" is 3
    and is the one that drops.
    """
    stories = raw_items[raw_items["type"] == STORY_TYPE_CODE]
    assert len(tokenise(normalise_title("Rust 1.90 released"))) == 4
    assert len(tokenise(normalise_title("My side project"))) == 3
    assert len(filter_stories(stories, min_title_tokens=4)) == 3


# --------------------------------------------------------------------------------------
# The int8 type encoding
# --------------------------------------------------------------------------------------


def test_decode_item_type_matches_the_recorded_encoding() -> None:
    """The mapping derived in Phase 1 and confirmed against the live HN API.

    Each code was checked by fetching one id from
    ``https://hacker-news.firebaseio.com/v0/item/<id>.json``: 44147768 story,
    44147746 comment, 44192767 poll, 44192768 pollopt, 44169039 job.
    """
    decoded = decode_item_type(pd.Series([1, 2, 3, 4, 5]))
    assert list(decoded) == ["story", "comment", "poll", "pollopt", "job"]


def test_story_type_code_is_one() -> None:
    assert STORY_TYPE_CODE == 1
    assert ITEM_TYPE_BY_CODE[STORY_TYPE_CODE] == "story"


def test_decode_item_type_flags_an_unknown_code_instead_of_raising() -> None:
    """A sixth item type added upstream must show up in a value count, not kill ingest."""
    assert list(decode_item_type(pd.Series([1, 9]))) == ["story", "unknown:9"]


def test_find_story_type_code_is_not_fooled_by_polls_and_jobs(raw_items: pd.DataFrame) -> None:
    """Polls and jobs carry both a title and a score, so "the only type with both" fails.

    Measured on ``2025-06``: code 1 has 31,529 rows, code 3 has 3, code 5 has 46. The
    story code is picked by count, not by presence.
    """
    both = raw_items[(raw_items["title"].fillna("") != "") & (raw_items["score"].fillna(0) > 0)]
    assert set(both["type"]) > {STORY_TYPE_CODE}, "fixture must contain non-story titled rows"
    assert find_story_type_code(raw_items) == STORY_TYPE_CODE


# --------------------------------------------------------------------------------------
# Leak audit against the columns ingest actually writes
# --------------------------------------------------------------------------------------


def test_ingested_columns_are_rejected_as_a_feature_set() -> None:
    """``PROJECTED_COLUMNS`` is exactly what ``data/stories.parquet`` carries.

    ``score`` and ``descendants`` are in it, because the target and the audit need them.
    Neither may reach a model, so the allowlist must reject the frame wholesale.
    """
    assert "score" in PROJECTED_COLUMNS
    assert "descendants" in PROJECTED_COLUMNS
    assert leaked_columns_present(PROJECTED_COLUMNS) == frozenset({"score", "descendants"})
    with pytest.raises(LeakedFeatureError):
        validate_feature_set(PROJECTED_COLUMNS)


def test_kids_is_absent_by_construction_and_would_be_rejected_anyway() -> None:
    """Both halves matter.

    ``kids`` is never projected, so it cannot leak today. The second assertion is what
    protects a future phase that adds a column: if ``kids`` ever appears in the frame,
    the allowlist still refuses it.
    """
    assert "kids" not in PROJECTED_COLUMNS
    assert leaked_columns_present(["title", "kids"]) == frozenset({"kids"})
    with pytest.raises(LeakedFeatureError):
        validate_feature_set(["title", "by", "url", "time", "kids"])


# --------------------------------------------------------------------------------------
# Tokeniser behaviour that the words comparison turned on
# --------------------------------------------------------------------------------------


def test_tokeniser_keeps_the_tokens_words_splits() -> None:
    """The measured disagreement classes against the dump's ``words`` column.

    ``words`` splits on the apostrophe and the hyphen: 42.1% and 18.2% of the tokens
    this tokeniser produces and ``words`` does not, over 5,000 rows of ``2026-06``.
    Keeping them whole is the reason the project uses this one.
    """
    assert tokenise("don't ship ad-free gpt-4 on s3") == [
        "don't",
        "ship",
        "ad-free",
        "gpt-4",
        "on",
        "s3",
    ]


def test_normalise_title_unescapes_twice() -> None:
    """The dump carries double-escaped entities, so one unescape pass is not enough."""
    assert normalise_title("Google&amp;#x27;s plan") == "google's plan"
    assert normalise_title("Rock &amp; Roll\n  live") == "rock & roll live"
