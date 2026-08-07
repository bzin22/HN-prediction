"""Leaked-column guard.

The banned names are hard-coded here on purpose. If they were imported from
``schema.py``, editing that module could move a column from banned to allowed and the
test would follow it and stay green. Written this way, adding ``score`` to the allowlist
turns the suite red.
"""

import pytest

from hn_upvotes.features import schema

# Post-hoc columns, restated independently of the module under test.
# score        the thing being predicted
# descendants  comment count, grows alongside the score after the post is live
# kids         direct child comment ids, the same information as a list
BANNED = frozenset({"score", "descendants", "kids"})

# The allowlist, also restated independently. `text` joined it in Phase 2, as a
# deliberate widening: the poster writes the body before pressing submit, so its value is
# fixed at submission time. Adding a name here is the reviewed step, not a formality.
EXPECTED_ALLOWLIST = frozenset({"title", "by", "url", "time", "text"})


def test_allowlist_holds_only_the_five_submission_time_columns():
    assert schema.ALLOWED_FEATURE_COLUMNS == EXPECTED_ALLOWLIST


def test_body_text_is_allowed_and_the_banned_names_still_are_not():
    """Pins the Phase 2 widening in both directions, so neither half drifts."""
    assert "text" in schema.ALLOWED_FEATURE_COLUMNS
    assert schema.validate_feature_set(["title", "text"]) == frozenset({"title", "text"})
    for column in sorted(BANNED):
        with pytest.raises(schema.LeakedFeatureError):
            schema.validate_feature_set(["text", column])


def test_no_post_hoc_column_is_allowlisted():
    overlap = BANNED & schema.ALLOWED_FEATURE_COLUMNS
    assert overlap == frozenset(), f"post-hoc columns reached the allowlist: {sorted(overlap)}"


def test_every_post_hoc_column_is_named_as_banned():
    assert BANNED <= schema.LEAKED_COLUMNS


@pytest.mark.parametrize("column", sorted(BANNED))
def test_validate_rejects_each_post_hoc_column(column):
    with pytest.raises(schema.LeakedFeatureError):
        schema.validate_feature_set(["title", column])


def test_validate_accepts_the_allowlist_and_returns_it():
    assert schema.validate_feature_set(["title", "by"]) == frozenset({"title", "by"})
    assert schema.validate_feature_set(schema.ALLOWED_FEATURE_COLUMNS) == (
        schema.ALLOWED_FEATURE_COLUMNS
    )


def test_validate_rejects_a_column_nobody_approved():
    # `id` is real, is available at submission time, and is still rejected until somebody
    # adds it deliberately. That is what makes this an allowlist. It is bookkeeping, not
    # a feature: a model that learned from the id would be reading submission order.
    with pytest.raises(schema.UnknownFeatureError):
        schema.validate_feature_set(["title", "id"])


def test_leaked_columns_present_reports_what_a_frame_is_carrying():
    columns = ["id", "title", "by", "time", "score", "descendants"]
    assert schema.leaked_columns_present(columns) == frozenset({"score", "descendants"})
    assert schema.leaked_columns_present(["title", "by"]) == frozenset()
