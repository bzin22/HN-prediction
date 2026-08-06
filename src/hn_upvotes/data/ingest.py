"""Read the Hacker News dump into a stories table.

Source is ``open-index/hacker-news`` on Hugging Face: monthly Parquet files, zstd
compressed, licence ``odc-by``. 49.1M rows covering every item type. Phase 1 filters to
stories with a title and a score and reports the real usable row count.

Queried with DuckDB over the Parquet files directly. No BigQuery, no billing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: Columns present in the source Parquet files.
SOURCE_COLUMNS: tuple[str, ...] = (
    "id",
    "deleted",
    "type",
    "by",
    "time",
    "text",
    "dead",
    "parent",
    "poll",
    "kids",
    "url",
    "score",
    "title",
    "parts",
    "descendants",
    "words",
)


@dataclass(frozen=True)
class IngestReport:
    """What ingest actually produced. Written next to the output for provenance.

    The row counts go in the README instead of the figure quoted in the original brief,
    which came from a different snapshot.
    """

    shards_read: int
    rows_scanned: int
    rows_kept: int
    earliest: pd.Timestamp
    latest: pd.Timestamp
    story_type_code: int


def list_monthly_shards(root: Path) -> list[Path]:
    """Return the monthly Parquet shards under ``root``, in chronological order.

    Sorted by filename, which encodes the month, so downstream code can stream in time
    order without opening anything.
    """
    raise NotImplementedError


def decode_item_type(raw: pd.Series) -> pd.Series:
    """Decode the ``int8`` ``type`` column into item type names.

    The dump stores ``type`` as a small integer rather than the string the HN API
    returns. The mapping is not documented upstream, so Phase 1 derives it by joining a
    sample against the live API and records it here rather than guessing.
    """
    raise NotImplementedError


def find_story_type_code(sample: pd.DataFrame) -> int:
    """Work out which ``type`` code means "story", from a sample of rows.

    Stories are the only item type that carries both a ``title`` and a ``score``, which
    is enough to identify the code without calling the API.
    """
    raise NotImplementedError


def load_stories(
    shards: list[Path],
    story_type_code: int,
    columns: tuple[str, ...] = SOURCE_COLUMNS,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load story rows from the given shards via DuckDB.

    Filters to the story type, drops deleted and dead items, and requires a non-null
    title and score. ``limit`` caps the row count for a quick development pass.
    """
    raise NotImplementedError


def ingest(
    source_root: Path,
    destination: Path,
    limit: int | None = None,
) -> IngestReport:
    """Run the full ingest and write a single Parquet file of stories.

    Returns the report so the caller can print the real row count rather than an
    assumed one.
    """
    raise NotImplementedError
