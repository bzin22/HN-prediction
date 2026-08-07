"""Read the Hacker News dump into a stories table.

Source is ``open-index/hacker-news`` on Hugging Face: monthly Parquet files, zstd
compressed, licence ``odc-by``. 49,119,480 rows covering every item type, measured
2026-08-05. Filtered to stories with a title and a score.

Queried with DuckDB over the Parquet files directly. No BigQuery, no billing. DuckDB
pushes the column projection into the Parquet reader, so pulling eight of the sixteen
columns transfers 776 MB rather than the 11.93 GB the full column set occupies. Measure
it with :func:`projected_transfer_bytes` before pulling.

Three things about the source that are not in its documentation and cost an afternoon
to find:

* ``by`` is a reserved word in DuckDB. It has to be double quoted in every query.
* Missing values are sentinels, not SQL ``NULL``. An absent title is ``''`` and an
  absent score is ``0``. ``count(title)`` returns the row count on a table of comments.
* ``time`` is ``TIMESTAMP_MICROS``, UTC, not the unix seconds the HN API returns. DuckDB
  reads it as ``TIMESTAMPTZ`` and will render it in the session timezone unless it is
  cast with ``AT TIME ZONE 'UTC'``.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

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

#: The nine columns pulled across the full history. ``words`` is the one large column
#: left behind (4.67 GB of the 11.93 GB total); it is sampled for one month by
#: :func:`load_words_sample`.
#:
#: ``text`` is the body the poster wrote under the headline. Its 6.18 GB headline figure
#: covers all 49 million items and is dominated by 45 million comment bodies, so the
#: story-only share is far smaller. It is a legal feature: it exists at submission.
#:
#: ``score`` and ``descendants`` are here because they are needed to build and to audit
#: the target, not because they are features. ``features/schema.py`` rejects both.
PROJECTED_COLUMNS: tuple[str, ...] = (
    "id",
    "type",
    "time",
    "title",
    "text",
    "score",
    "by",
    "url",
    "descendants",
)

#: The ``int8`` ``type`` encoding. Not documented upstream. See :func:`decode_item_type`
#: for the evidence.
ITEM_TYPE_BY_CODE: dict[int, str] = {
    1: "story",
    2: "comment",
    3: "poll",
    4: "pollopt",
    5: "job",
}

#: The code that means "story". Everything downstream filters on this.
STORY_TYPE_CODE = 1

#: The dataset, and the glob DuckDB reads it through.
DATASET = "open-index/hacker-news"
REMOTE_ROOT = f"hf://datasets/{DATASET}/data"
REMOTE_GLOB = f"{REMOTE_ROOT}/*/*.parquet"

#: One row per committed month. ``committed_at`` is the instant every score in that
#: month's file was observed, which is what makes the settling-lag gate measurable.
STATS_URL = f"https://huggingface.co/datasets/{DATASET}/raw/main/stats.csv"

#: Stop condition for the pull. A projection above this means DuckDB is not pushing the
#: column list into the Parquet reader and something is wrong.
MAX_PROJECTED_BYTES = 15 * 1000**3

#: Months whose archived ``score`` was captured at or near submission instead of after
#: the post finished scoring. **Their scores are not usable as labels.**
#:
#: Found in Phase 1 by refetching a sample of 70 stories per month from
#: ``https://hacker-news.firebaseio.com/v0/item/<id>.json`` and comparing. A clean month
#: matches the live API on 83% to 100% of rows and its mean is within a few percent. A
#: month in this list matches on 34% to 54% and its mean is 5 to 8 times too low:
#: ``2023-05`` reads 2.27 against a live 11.43, ``2026-07`` reads 1.76 against 21.33.
#: ``2024-01`` and ``2025-11`` both match on 100%.
#:
#: The split is clean, not a judgement call. Of the 233 months with at least 1,000
#: stories, 20 have a mean ``log1p(score)`` below 1.10 and 212 are above 1.26. Nothing
#: sits between 1.20 and 1.26. ``2023-12`` is the one transitional month, at 1.199 and
#: 64% agreement, and is excluded with the rest.
#:
#: Two windows, so this is a recurring upstream regression rather than a one-off.
UNSETTLED_SCORE_MONTHS: frozenset[str] = frozenset(
    [f"2023-{m:02d}" for m in range(1, 13)] + ["2022-12"] + [f"2026-{m:02d}" for m in range(1, 9)]
)


def drop_unsettled_months(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose month is in :data:`UNSETTLED_SCORE_MONTHS`.

    Every measurement that reads ``score`` as a final value goes through this, and so
    must every training label. The rows are kept in ``data/stories.parquet`` rather than
    filtered at ingest, because their titles, authors and timestamps are still true and
    a later phase may want them for something that is not a label.
    """
    month = pd.to_datetime(frame["time"]).dt.strftime("%Y-%m")
    return frame.loc[~month.isin(UNSETTLED_SCORE_MONTHS)].copy()


#: Missing values are sentinels rather than NULL, so the filter tests the sentinels.
#: ``dead`` and ``deleted`` are ``UINT8`` flags, never null.
_STORY_PREDICATE = """
    type = {story_type_code}
    AND coalesce(dead, 0) = 0
    AND coalesce(deleted, 0) = 0
    AND title IS NOT NULL AND title <> ''
    AND score IS NOT NULL AND score > 0
"""

_STORY_PROJECTION = """
    id,
    type,
    time AT TIME ZONE 'UTC' AS time,
    title,
    text,
    score,
    "by",
    url,
    descendants
"""


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

    @property
    def keep_rate(self) -> float:
        return self.rows_kept / self.rows_scanned if self.rows_scanned else 0.0

    def summary(self) -> str:
        return (
            f"{self.rows_kept:,} usable stories from {self.rows_scanned:,} items "
            f"across {self.shards_read} monthly shards "
            f"({self.keep_rate:.1%} kept), "
            f"{self.earliest:%Y-%m-%d} to {self.latest:%Y-%m-%d}, "
            f"story type code {self.story_type_code}"
        )


def connect(threads: int | None = None):  # noqa: ANN201 - duckdb is an optional extra
    """Open a DuckDB connection that can read ``hf://`` paths.

    ``httpfs`` auto-installs on first use. The progress bar is off because this runs
    under ``make`` and in a notebook, where it produces megabytes of carriage returns.
    """
    import duckdb

    con = duckdb.connect()
    # Inside Jupyter, DuckDB validates the progress-bar setting against ipywidgets before
    # honouring it, so turning the bar *off* raises when the widget package is absent.
    # The bar stays off either way. Not worth an extra dependency.
    with contextlib.suppress(duckdb.InvalidInputException):
        con.execute("SET enable_progress_bar = false")
    if threads is not None:
        con.execute(f"SET threads = {threads}")
    con.execute("INSTALL httpfs")
    con.execute("LOAD httpfs")
    return con


def projected_transfer_bytes(con, source: str = REMOTE_GLOB) -> pd.DataFrame:
    """Compressed bytes per column across the source, from the Parquet footers.

    Reads only the file footers, so it costs seconds and answers the question the pull
    depends on: does the column projection actually get pushed down. Compare the
    ``PROJECTED_COLUMNS`` subtotal against :data:`MAX_PROJECTED_BYTES` before pulling.
    """
    return con.execute(
        f"""
        SELECT
            regexp_replace(path_in_schema, ', list, element$', '') AS column_name,
            sum(total_compressed_size)::BIGINT AS compressed_bytes
        FROM parquet_metadata('{source}')
        GROUP BY 1
        ORDER BY 2 DESC
        """
    ).df()


def list_remote_months(con, source: str = REMOTE_GLOB) -> list[str]:
    """Return the ``YYYY-MM`` months the dataset publishes, in chronological order.

    Read from the file names rather than from ``stats.csv``, because the archive is live
    and the two can disagree by a month at the boundary.
    """
    frame = con.execute(
        f"""
        SELECT DISTINCT regexp_extract(file_name, '([0-9]{{4}}-[0-9]{{2}})\\.parquet$', 1) AS month
        FROM parquet_metadata('{source}')
        ORDER BY 1
        """
    ).df()
    return [m for m in frame["month"].tolist() if m]


def load_commit_times(url: str = STATS_URL) -> pd.DataFrame:
    """Read ``stats.csv``: one row per committed month, with its observation instant.

    ``committed_at`` is when that month's file was fetched from the HN API, so every
    score in it was read at that instant. Subtracting a story's ``time`` from it gives
    the story's age when its score was observed, which is the whole of gate 3.
    """
    stats = pd.read_csv(url)
    year = stats["year"].astype(int).astype(str)
    stats["month"] = year + "-" + stats["month"].astype(int).map("{:02d}".format)
    stats["committed_at"] = pd.to_datetime(stats["committed_at"], utc=True).dt.tz_localize(None)
    return stats[["month", "lowest_id", "highest_id", "count", "size_bytes", "committed_at"]]


def list_monthly_shards(root: Path) -> list[Path]:
    """Return the monthly Parquet shards under ``root``, in chronological order.

    Sorted by filename, which encodes the month, so downstream code can stream in time
    order without opening anything.
    """
    return sorted(Path(root).glob("*.parquet"))


def decode_item_type(raw: pd.Series) -> pd.Series:
    """Decode the ``int8`` ``type`` column into item type names.

    The dump stores ``type`` as a small integer rather than the string the HN API
    returns, and the mapping is not documented upstream. It was derived, not guessed.

    Evidence, from ``data/2025/2025-06.parquet``. Each code was characterised by which
    columns it carries, then one id per code was fetched from the live HN API at
    ``https://hacker-news.firebaseio.com/v0/item/<id>.json`` and its ``type`` string
    read off. All five agreed:

    ==== ========= =========== ================================================
    Code Name      Checked id  Shape in the dump
    ==== ========= =========== ================================================
    1    story     44147768    title, score, url, no parent
    2    comment   44147746    parent, no title, no score
    3    poll      44192767    title, score, descendants, no url
    4    pollopt   44192768    score, no title, ids follow their poll's
    5    job       44169039    title, url, score of 1, no descendants
    ==== ========= =========== ================================================

    Unknown codes come back as ``"unknown:<code>"`` rather than raising, so a future
    item type added upstream shows up in a value count instead of killing the ingest.
    """
    codes = pd.Series(raw).astype("Int64")
    names = codes.map(ITEM_TYPE_BY_CODE)
    missing = names.isna() & codes.notna()
    if missing.any():
        names = names.where(~missing, "unknown:" + codes.astype(str))
    return names.astype("object")


def find_story_type_code(sample: pd.DataFrame) -> int:
    """Work out which ``type`` code means "story", from a sample of rows.

    The obvious rule, "stories are the only type with both a title and a score", is
    false in this dump. Polls (code 3) and jobs (code 5) carry both. Measured on
    ``2025-06``: code 1 has 31,529 rows, code 3 has 3, code 5 has 46.

    So the rule used is: among the codes that carry both a title and a score, the story
    code is the most common by three orders of magnitude. That holds for every month.
    """
    frame = sample
    titled = frame["title"].notna() & (frame["title"].astype(str) != "")
    scored = frame["score"].notna() & (pd.to_numeric(frame["score"], errors="coerce") > 0)
    candidates = frame.loc[titled & scored, "type"]
    if candidates.empty:
        raise ValueError("no rows in the sample carry both a title and a score")
    counts = candidates.value_counts()
    return int(counts.index[0])


def load_stories(
    shards: list[Path],
    story_type_code: int = STORY_TYPE_CODE,
    columns: tuple[str, ...] = PROJECTED_COLUMNS,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load story rows from the given shards via DuckDB.

    Filters to the story type, drops deleted and dead items, and requires a non-null
    title and score. ``limit`` caps the row count for a quick development pass.

    Works on remote ``hf://`` paths and on local shards alike, so the same filter runs
    during the pull and during a re-read.
    """
    if not shards:
        raise ValueError("no shards to read")
    paths = [str(p) for p in shards]
    con = connect()
    try:
        return _select_stories(con, paths, story_type_code, columns, limit)
    finally:
        con.close()


def _select_stories(
    con,
    paths: list[str],
    story_type_code: int,
    columns: tuple[str, ...],
    limit: int | None,
) -> pd.DataFrame:
    predicate = _STORY_PREDICATE.format(story_type_code=story_type_code)
    sql = f"""
        SELECT {_STORY_PROJECTION}
        FROM read_parquet({paths!r})
        WHERE {predicate}
        ORDER BY time
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    frame = con.execute(sql).df()
    return frame[list(columns)]


def download_shard(con, month: str, shard_dir: Path, overwrite: bool = False) -> Path:
    """Pull one month, filtered and projected, to a local Parquet shard.

    Existing shards are skipped, so an interrupted pull resumes instead of restarting.
    The shards are scratch: nothing after Phase 1 reads them. ``make clean-shards``
    removes them once ingest has reported its row count.
    """
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    out = shard_dir / f"{month}.parquet"
    # A shard written before a change to PROJECTED_COLUMNS is missing a column the
    # caller now wants, and resuming would reuse it silently. Check the columns rather
    # than the file name.
    if out.exists() and not overwrite and _shard_is_current(con, out):
        return out

    year = month.split("-")[0]
    source = f"{REMOTE_ROOT}/{year}/{month}.parquet"
    predicate = _STORY_PREDICATE.format(story_type_code=STORY_TYPE_CODE)
    # Written to a temporary name and renamed, so an interrupted pull never leaves a
    # truncated shard that the resume would then skip.
    tmp = out.with_suffix(".parquet.partial")
    con.execute(
        f"""
        COPY (
            SELECT {_STORY_PROJECTION}
            FROM read_parquet('{source}')
            WHERE {predicate}
            ORDER BY time
        ) TO '{tmp}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    tmp.replace(out)
    return out


def _shard_is_current(con, shard: Path) -> bool:
    """True if an existing shard carries every column in :data:`PROJECTED_COLUMNS`."""
    try:
        present = {
            row[0]
            for row in con.execute(
                f"SELECT name FROM (DESCRIBE SELECT * FROM read_parquet('{shard}'))"
            ).fetchall()
        }
    except Exception:  # noqa: BLE001 - a shard we cannot read is a shard we re-pull
        return False
    return set(PROJECTED_COLUMNS).issubset(present)


def download_shards(
    shard_dir: Path,
    months: list[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Pull every month to ``shard_dir``. Returns the shard paths in time order."""
    con = connect()
    try:
        months = months or list_remote_months(con)
        paths = []
        for i, month in enumerate(months, 1):
            path = download_shard(con, month, shard_dir, overwrite=overwrite)
            paths.append(path)
            logger.info("shard %d/%d %s", i, len(months), month)
        return paths
    finally:
        con.close()


def load_words_sample(month: str = "2026-06", limit: int | None = None) -> pd.DataFrame:
    """Pull ``words`` for a single month, for the tokeniser comparison.

    ``words`` is 4.67 GB across the full history, so it is sampled rather than ingested.
    It is also not what its name suggests: it tokenises ``text``, never ``title``, and
    it is sorted and deduplicated, so word order is gone. See ``docs/design.md``.

    ``2026-06`` is the default because ``words`` is empty for every month from 2025-01
    to 2025-11 and is populated either side of that gap.
    """
    year = month.split("-")[0]
    source = f"{REMOTE_ROOT}/{year}/{month}.parquet"
    con = connect()
    try:
        sql = f"""
            SELECT id, type, title, text, words
            FROM read_parquet('{source}')
            WHERE len(words) > 0
        """
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return con.execute(sql).df()
    finally:
        con.close()


def ingest(
    source_root: Path,
    destination: Path,
    limit: int | None = None,
) -> IngestReport:
    """Run the full ingest and write a single Parquet file of stories.

    ``source_root`` is the local shard directory written by :func:`download_shards`.
    Returns the report so the caller can print the real row count rather than an
    assumed one.
    """
    shards = list_monthly_shards(source_root)
    if not shards:
        raise FileNotFoundError(
            f"no shards under {source_root}. Run download_shards() first, or `make ingest`."
        )

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    con = connect()
    try:
        paths = [str(p) for p in shards]
        # The shards are already filtered, so this is a concatenation plus a re-sort.
        con.execute(
            f"""
            COPY (
                SELECT {", ".join(f'"{c}"' for c in PROJECTED_COLUMNS)}
                FROM read_parquet({paths!r})
                ORDER BY time
                {f"LIMIT {int(limit)}" if limit is not None else ""}
            ) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        kept, earliest, latest = con.execute(
            f"SELECT count(*), min(time), max(time) FROM read_parquet('{destination}')"
        ).fetchone()
        scanned = int(
            con.execute(f"SELECT sum(count) FROM read_csv_auto('{STATS_URL}')").fetchone()[0]
        )
    finally:
        con.close()

    return IngestReport(
        shards_read=len(shards),
        rows_scanned=scanned,
        rows_kept=int(kept),
        earliest=pd.Timestamp(earliest),
        latest=pd.Timestamp(latest),
        story_type_code=STORY_TYPE_CODE,
    )


def main() -> None:
    """``make ingest``. Measure the projection, pull the shards, write the table."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    shard_dir = Path("data/shards")
    destination = Path("data/stories.parquet")

    con = connect()
    try:
        sizes = projected_transfer_bytes(con)
    finally:
        con.close()
    wanted = sizes["column_name"].isin(PROJECTED_COLUMNS)
    projected = int(sizes.loc[wanted, "compressed_bytes"].sum())
    total = int(sizes["compressed_bytes"].sum())
    logger.info(
        "projected transfer %s bytes (%.2f GB) of %s bytes (%.2f GB) for all columns",
        f"{projected:,}",
        projected / 1000**3,
        f"{total:,}",
        total / 1000**3,
    )
    if projected > MAX_PROJECTED_BYTES:
        raise SystemExit(
            f"projected transfer {projected:,} bytes exceeds the {MAX_PROJECTED_BYTES:,} byte "
            "limit. The column projection is not being pushed into the Parquet reader."
        )

    download_shards(shard_dir)
    report = ingest(shard_dir, destination)
    logger.info("%s", report.summary())
    logger.info("wrote %s", destination)
    logger.info("the shards under %s are scratch. `make clean-shards` removes them.", shard_dir)


if __name__ == "__main__":
    main()
