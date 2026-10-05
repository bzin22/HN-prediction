"""Check full Wikipedia export with local data. No corpus download is required."""

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb", reason="the data extra is not installed")

from hn_upvotes.embeddings import corpora  # noqa: E402


def test_wikipedia_export_reads_every_batch_and_keeps_the_text_filter(tmp_path, monkeypatch):
    source = tmp_path / "wiki.parquet"
    with duckdb.connect() as connection:
        connection.execute(
            "CREATE TABLE articles AS SELECT repeat('word ', 45) || i::VARCHAR AS text "
            "FROM range(20005) AS t(i)"
        )
        connection.execute("INSERT INTO articles VALUES (NULL), ('short article')")
        connection.execute("COPY articles TO ? (FORMAT PARQUET)", [str(source)])
    monkeypatch.setattr(corpora, "WIKIPEDIA_PARQUET_GLOB", str(source))
    destination = tmp_path / "wiki.txt"

    assert corpora.prepare_wikipedia_corpus(destination) == destination
    lines = destination.read_text().splitlines()
    assert len(lines) == 20005
    assert {int(line.rsplit(" ", 1)[1]) for line in lines} == set(range(20005))
    assert len(list(corpora.stream_plain_text(destination))) == 20005
    assert not Path(str(destination) + ".partial").exists()


@pytest.mark.parametrize("existing", [False, True])
def test_failed_export_never_publishes_a_partial_corpus(tmp_path, monkeypatch, existing):
    destination = tmp_path / "wiki.txt"
    if existing:
        destination.write_text("previous complete corpus\n")

    class BrokenReader:
        calls = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, query):
            return self

        def fetchmany(self, size):
            self.calls += 1
            if self.calls == 1:
                return [("first article",)]
            raise OSError("source read failed")

    monkeypatch.setattr(duckdb, "connect", BrokenReader)
    with pytest.raises(OSError, match="source read failed"):
        corpora.prepare_wikipedia_corpus(destination)
    if existing:
        assert destination.read_text() == "previous complete corpus\n"
    else:
        assert not destination.exists()
    assert not Path(str(destination) + ".partial").exists()


def test_hn_excludes_validation_boundary_and_later_text(tmp_path):
    source = tmp_path / "hn's.parquet"
    with duckdb.connect() as connection:
        connection.execute("CREATE TABLE stories(title VARCHAR, text VARCHAR, time TIMESTAMP)")
        connection.execute(
            "INSERT INTO stories VALUES "
            "('earlier title', '<p>earlier body</p>', '2021-11-30 23:59:59'), "
            "('boundary', 'excluded', '2021-12-01'), ('later', 'excluded', '2024-01-01')"
        )
        connection.execute("COPY stories TO ? (FORMAT PARQUET)", [str(source)])
    assert list(corpora.stream_hn_text(source, before="2021-12-01")) == [
        ["earlier", "title"],
        ["earlier", "body"],
    ]
    assert len(list(corpora.stream_hn_text(source))) == 6
