"""The corpora the four training stages read, behind one shape: an iterator of token lists.

Every source yields lines of tokens, because that is what
:func:`~hn_upvotes.embeddings.train.train_embeddings_from_lines` consumes and because a line
is the boundary a context window must not cross. Two words either side of a line break are
not neighbours.

Four sources:

* **text8**, the gate corpus. One 100 MB line of lowercase Wikipedia with punctuation
  removed, so it is read in chunks by :func:`~hn_upvotes.data.preprocess.stream_corpus`.
* **English Wikipedia**, read without a corpus-size limit.
* **Hacker News titles and bodies**, from the cached story table.
* **A planted-structure synthetic corpus**, which needs no network and is what the dry run
  and the tests use.

Needs the ``train`` extra for the training side; the Hacker News reader also needs ``data``
for duckdb.
"""

from __future__ import annotations

import urllib.request
import zipfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from hn_upvotes.data.preprocess import normalise_title, stream_corpus, tokenise

#: Where the cached Hacker News table lives. Built by ``make ingest``; 365 MB as of
#: 2026-08-06, 4,739,207 stories of which 352,771 (7.4%) carry body text.
HN_CORPUS_PATH = Path.home() / ".cache" / "hn-prediction" / "stories.parquet"

#: The Hacker News corpus in tokens, titles and bodies, **counted rather than estimated** with
#: ``make hn-token-count`` on 2026-08-07: 75,283,676 tokens over 5,091,739 lines, 14.8 tokens a
#: line. Re-count after an ingest, since the archive grows.
#:
#: The bodies are most of it. 4,739,207 titles at about 8 tokens is 38M, so the 352,532 lines
#: of body text carry the other half of the corpus on 7% of the rows.
HN_CORPUS_TOKENS = 75_283_676

#: text8's canonical home. 31 MB zipped, 100 MB unpacked, 17.0M tokens.
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"

#: English Wikipedia snapshot, read with DuckDB over ``hf://``. Local export tests pass;
#: the full remote acquisition has not been run. There is no row or token cap.
WIKIPEDIA_PARQUET_GLOB = "hf://datasets/wikimedia/wikipedia/20231101.en/*.parquet"


def stream_text8(path: Path) -> Iterator[list[str]]:
    """text8 as token lists. It is one enormous line, so chunking is the whole trick."""
    return stream_corpus(Path(path))


def download_text8(destination: Path) -> Path:
    """Fetch and unpack text8 if it is not already there, and return the text file path.

    **Run, and it works.** The learning-rate sweep needed text8, so this path is exercised:
    31 MB zipped from ``mattmahoney.net``, 100,000,000 bytes and 17,005,207 tokens unpacked.
    """
    destination = Path(destination)
    text_path = destination / "text8"
    if text_path.exists():
        return text_path
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "text8.zip"
    urllib.request.urlopen  # noqa: B018 - named so the network call is greppable
    with urllib.request.urlopen(TEXT8_URL) as response, archive.open("wb") as handle:
        handle.write(response.read())
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    return text_path


def take_tokens(lines: Iterator[list[str]], token_budget: int | None) -> Iterator[list[str]]:
    """Stop a corpus stream after ``token_budget`` tokens, cutting the last line short.

    text8 is one 100 MB line, so the budget cannot be applied by counting lines. This wraps
    any stream of token lists, which is what the learning-rate sweep needs to give four
    training runs the identical corpus without holding it in memory.
    """
    if token_budget is None:
        yield from lines
        return
    taken = 0
    for tokens in lines:
        if taken + len(tokens) > token_budget:
            remaining = token_budget - taken
            if remaining > 0:
                yield tokens[:remaining]
            return
        taken += len(tokens)
        yield tokens


def stream_plain_text(path: Path, token_budget: int | None = None) -> Iterator[list[str]]:
    """A plain-text corpus, one document per line, stopped at ``token_budget`` tokens.

    The training chain reads the entire file. Optional token limits remain available for
    bounded diagnostic experiments.
    """
    taken = 0
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            tokens = tokenise(normalise_title(line))
            if not tokens:
                continue
            if token_budget is not None and taken + len(tokens) > token_budget:
                remaining = token_budget - taken
                if remaining > 0:
                    yield tokens[:remaining]
                return
            taken += len(tokens)
            yield tokens


def prepare_wikipedia_corpus(destination: Path) -> Path:
    """Export all eligible English Wikipedia articles as one document per line.

    Read only the text column, in bounded batches. There is no row or token limit.
    Keep the existing text filter: exclude null text and articles of 200 characters or less.
    Publish the file only after the export succeeds, so a failed download cannot look complete.
    Both objectives reuse this file. Remote acquisition still requires a live validation run.
    """
    import duckdb

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    query = f"""
        SELECT text
        FROM '{WIKIPEDIA_PARQUET_GLOB}'
        WHERE text IS NOT NULL AND length(text) > 200
    """
    try:
        with duckdb.connect() as connection, temporary.open("w", encoding="utf-8") as handle:
            reader = connection.execute(query)
            while rows := reader.fetchmany(10_000):
                for (text,) in rows:
                    handle.write(" ".join(str(text).split()) + "\n")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def stream_hn_text(
    parquet_path: Path = HN_CORPUS_PATH,
    include_bodies: bool = True,
    token_budget: int | None = None,
    before: str | None = None,
) -> Iterator[list[str]]:
    """Hacker News titles, and the bodies where there are any, as separate lines.

    A title and its body are yielded as two lines rather than one, because they are
    different registers and the last word of a headline is not the neighbour of the first
    word of the paragraph under it.

    Bodies arrive as HTML from the API, so they go through
    :func:`~hn_upvotes.features.body.strip_html` first. Without it the vocabulary fills up
    with ``href``, ``rel`` and ``nofollow``.

    Titles are about 8 tokens and a body over 1,000 characters is roughly 200 words with
    real sentence structure, which is why the bodies are worth the extra column: a context
    window has something to work with.

    ``before`` is an exclusive UTC date cutoff. Use the validation boundary when the
    embeddings will be compared on a held-back validation period.
    """
    import duckdb

    from hn_upvotes.features.body import strip_html

    columns = "title, text" if include_bodies else "title, '' AS text"
    query = f"SELECT {columns} FROM read_parquet(?)"
    parameters = [str(Path(parquet_path))]
    if before is not None:
        query += " WHERE time < CAST(? AS TIMESTAMP)"
        parameters.append(before)
    taken = 0
    with duckdb.connect() as connection:
        reader = connection.execute(query, parameters)
        while True:
            chunk = reader.fetchmany(10_000)
            if not chunk:
                return
            for title, body in chunk:
                lines = [tokenise(normalise_title(title or ""))]
                if include_bodies and body:
                    lines.append(tokenise(normalise_title(strip_html(body))))
                for tokens in lines:
                    if not tokens:
                        continue
                    if token_budget is not None and taken + len(tokens) > token_budget:
                        remaining = token_budget - taken
                        if remaining > 0:
                            yield tokens[:remaining]
                        return
                    taken += len(tokens)
                    yield tokens


def count_hn_tokens(
    parquet_path: Path = HN_CORPUS_PATH, include_bodies: bool = True
) -> tuple[int, int]:
    """Tokens and lines in the Hacker News corpus, counted rather than estimated.

    Count both titles and bodies for runtime estimates. Recount after the input data or
    training-period filter changes.
    """
    tokens = 0
    lines = 0
    for line in stream_hn_text(parquet_path, include_bodies=include_bodies):
        tokens += len(line)
        lines += 1
    return tokens, lines


#: The synonym groups the planted corpus is built from. Each group gets two members that
#: never appear together and always appear between the same two context words, so their
#: distributions are identical and a working implementation must place them next to each
#: other.
PLANTED_GROUPS: tuple[str, ...] = ("red", "gas", "boat", "song", "chip", "bank")


def planted_synonym_corpus(repeats: int = 200, seed: int = 0) -> list[list[str]]:
    """A corpus with a known right answer, for the dry run and the tests.

    Each group ``g`` contributes sentences ``{g}left{1,2} {g}one {g}right{1,2}`` and the
    same with ``{g}two``. So ``redone`` and ``redtwo`` share every context word and never
    co-occur with each other. Distributional similarity therefore has to put them
    together, and co-occurrence cannot be what does it.

    At the default ``repeats=200`` this is 2,400 lines and 7,200 tokens, which trains in a
    couple of seconds and is enough for both objectives to recover all six pairs.
    """
    rng = np.random.default_rng(seed)
    lines: list[list[str]] = []
    for _ in range(repeats):
        for group in PLANTED_GROUPS:
            for member in ("one", "two"):
                lines.append(
                    [
                        f"{group}left{rng.integers(1, 3)}",
                        f"{group}{member}",
                        f"{group}right{rng.integers(1, 3)}",
                    ]
                )
    rng.shuffle(lines)
    return lines


def planted_synonym_pairs() -> list[tuple[str, str]]:
    """The pairs :func:`planted_synonym_corpus` plants, as the answer key."""
    return [(f"{group}one", f"{group}two") for group in PLANTED_GROUPS]


def topic_corpus(
    topics: int = 40,
    words_per_topic: int = 25,
    lines_per_topic: int = 400,
    line_length: int = 12,
    seed: int = 0,
) -> tuple[list[list[str]], dict[str, int]]:
    """A corpus of ``topics`` disjoint word sets, and the map from word to its topic.

    Each line draws its words from one topic, so words that share a topic co-occur and words
    that do not never co-occur. A working implementation puts each word's neighbours inside
    its own topic; chance level is ``1 / topics``, so 40 topics puts chance at 0.025.

    This is the dry run's gate corpus rather than :func:`planted_synonym_corpus`, and the
    reason is measured: on the planted corpus gensim recovers 0 of the 6 pairs while this
    implementation recovers all 6, because 36 word types over 7,200 tokens is too little for
    gensim to pull its vectors apart. Comparing against a reference that has not converged
    tells you nothing. At the defaults here the corpus is 1,000 word types over 192,000
    tokens and both implementations reach a topic purity of 1.00, so the comparison is
    between two models that both work.
    """
    rng = np.random.default_rng(seed)
    words_by_topic = [
        [f"t{topic}w{word}" for word in range(words_per_topic)] for topic in range(topics)
    ]
    lines = [
        list(rng.choice(words_by_topic[topic], size=line_length))
        for topic in range(topics)
        for _ in range(lines_per_topic)
    ]
    rng.shuffle(lines)
    topic_of = {word: topic for topic, words in enumerate(words_by_topic) for word in words}
    return lines, topic_of


def main(argv: list[str] | None = None) -> int:
    """Count the Hacker News corpus for runtime estimates."""
    import argparse

    parser = argparse.ArgumentParser(description=count_hn_tokens.__doc__.splitlines()[0])
    parser.add_argument("--hn-corpus", type=Path, default=HN_CORPUS_PATH)
    parser.add_argument("--titles-only", action="store_true", help="exclude body text")
    args = parser.parse_args(argv)

    tokens, lines = count_hn_tokens(args.hn_corpus, include_bodies=not args.titles_only)
    print(f"{tokens:,} tokens over {lines:,} lines in {args.hn_corpus}")
    print(f"{tokens / max(lines, 1):.1f} tokens per line")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
