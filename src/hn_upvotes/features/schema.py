"""The feature allowlist.

This module is the single place that decides what a model is allowed to see. It is an
allowlist, not a blocklist: a column that nobody has explicitly approved is rejected,
so adding a new source column to the dataset cannot silently leak it into a feature.

Three columns are called out by name because they are the ones that sink this project
if they slip through. ``score``, ``descendants`` and ``kids`` all describe what happened
*after* a post went live. Using any of them to predict the score is circular.

Nothing here is a stub. ``tests/test_feature_schema.py`` fails if a banned column
reaches the allowlist.
"""

from __future__ import annotations

from collections.abc import Iterable

#: Columns that exist at the moment a post is submitted, and are therefore legal inputs.
#:
#: ``title``  the submitted headline
#: ``by``     the submitting account
#: ``url``    the linked address, absent for text posts
#: ``time``   the submission timestamp, unix seconds in the source dump
ALLOWED_FEATURE_COLUMNS: frozenset[str] = frozenset({"title", "by", "url", "time"})

#: Columns that only acquire a value after the post is live. Never a feature.
#:
#: ``score``        the thing being predicted
#: ``descendants``  the comment count, which grows alongside the score
#: ``kids``         the direct child comment ids, the same information as a list
LEAKED_COLUMNS: frozenset[str] = frozenset({"score", "descendants", "kids"})

#: Columns kept through the pipeline for bookkeeping. Available to splitters, joins and
#: reporting, never handed to a feature builder.
METADATA_COLUMNS: frozenset[str] = frozenset({"id", "type", "deleted", "dead"})

#: The prediction target column in the raw dump. Read by ``target.normalise``, and by
#: nothing else.
TARGET_COLUMN = "score"


class LeakedFeatureError(ValueError):
    """A post-hoc column was proposed as a feature."""


class UnknownFeatureError(ValueError):
    """A column was proposed that is not on the allowlist."""


def leaked_columns_present(columns: Iterable[str]) -> frozenset[str]:
    """Return the subset of ``columns`` that is banned as post-hoc information.

    Cheap enough to call on a whole dataframe's columns before every training run.
    """
    return frozenset(columns) & LEAKED_COLUMNS


def validate_feature_set(columns: Iterable[str]) -> frozenset[str]:
    """Check a proposed feature set against the allowlist and return it.

    Raises ``LeakedFeatureError`` if any column is post-hoc, which is reported first
    because it is the more serious mistake. Raises ``UnknownFeatureError`` for any
    other column that is not allowlisted. Returns the validated set so this can be
    used inline: ``cols = validate_feature_set(cols)``.
    """
    proposed = frozenset(columns)

    leaked = proposed & LEAKED_COLUMNS
    if leaked:
        raise LeakedFeatureError(
            f"post-hoc columns cannot be features: {sorted(leaked)}. "
            "These are consequences of the post going live, not inputs available at "
            "submission time."
        )

    unknown = proposed - ALLOWED_FEATURE_COLUMNS
    if unknown:
        raise UnknownFeatureError(
            f"columns not on the allowlist: {sorted(unknown)}. "
            f"Allowed: {sorted(ALLOWED_FEATURE_COLUMNS)}. Add a column here only after "
            "confirming its value is known at submission time."
        )

    return proposed
