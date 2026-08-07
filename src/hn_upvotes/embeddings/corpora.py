"""The corpora the four training stages read, behind one shape: an iterator of token lists.

Every source yields lines of tokens, because that is what
:func:`~hn_upvotes.embeddings.train.train_embeddings_from_lines` consumes and because a line
is the boundary a context window must not cross. Two words either side of a line break are
not neighbours.

Four sources:

* **text8**, the gate corpus. One 100 MB line of lowercase Wikipedia with punctuation
  removed, so it is read in chunks by :func:`~hn_upvotes.data.preprocess.stream_corpus`.
* **A Wikipedia subset**, sized from measured throughput rather than picked.
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

#: text8's canonical home. 31 MB zipped, 100 MB unpacked, 17.0M tokens.
TEXT8_URL = "http://mattmahoney.net/dc/text8.zip"

#: The Wikipedia dump the subset is cut from, read with duckdb over ``hf://`` exactly as
#: ``data/ingest.py`` reads the Hacker News dump. **This path has not been exercised**: it
#: needs a download, and the build-only brief forbids one. Treat it as unverified.
WIKIPEDIA_PARQUET_GLOB = "hf://datasets/wikimedia/wikipedia/20231101.en/*.parquet"

#: Bytes per token in English plain text, used only to turn a token budget into a rough
#: size for the write-up. Measured on text8: 100,000,000 bytes over 17,005,207 tokens.
BYTES_PER_TOKEN = 5.9


def stream_text8(path: Path) -> Iterator[list[str]]:
    """text8 as token lists. It is one enormous line, so chunking is the whole trick."""
    return stream_corpus(Path(path))


def download_text8(destination: Path) -> Path:
    """Fetch and unpack text8 if it is not already there, and return the text file path.

    **Not run yet.** The build-only brief forbids downloading a corpus, so this is code the
    first real gate run will exercise, not code that has been exercised.
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


def stream_plain_text(path: Path, token_budget: int | None = None) -> Iterator[list[str]]:
    """A plain-text corpus, one document per line, stopped at ``token_budget`` tokens.

    The budget is the point of the Wikipedia stage: the subset size comes from a measured
    throughput and a wall-clock ceiling, so the reader takes a token count rather than the
    caller trimming a file to a guessed size.
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


def prepare_wikipedia_subset(destination: Path, token_budget: int) -> Path:
    """Write a plain-text Wikipedia subset of about ``token_budget`` tokens.

    Reads the Hugging Face dump with duckdb over ``hf://``, the same way
    ``data/ingest.py`` reads the Hacker News dump, and pushes the projection down so only
    the ``text`` column crosses the network.

    **Not run yet, and this is the one part of the chain that has not been proved.**
    Exercising it means downloading a corpus, which the build-only brief rules out. The
    chain's Wikipedia stage will fail here first if the query or the dataset path is wrong,
    and that failure is contained: stage 3 still produces the ``hn-only`` variant.
    """
    import duckdb

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Rows are far longer than a token, so this asks for a generous multiple and the
    # reader's own budget does the exact cut.
    rows = max(int(token_budget / 300), 1)
    query = f"""
        SELECT text
        FROM '{WIKIPEDIA_PARQUET_GLOB}'
        WHERE text IS NOT NULL AND length(text) > 200
        LIMIT {rows}
    """
    with destination.open("w", encoding="utf-8") as handle:
        for (text,) in duckdb.sql(query).fetchall():
            handle.write(" ".join(str(text).split()) + "\n")
    return destination


def stream_hn_text(
    parquet_path: Path = HN_CORPUS_PATH,
    include_bodies: bool = True,
    token_budget: int | None = None,
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
    """
    import duckdb

    from hn_upvotes.features.body import strip_html

    columns = "title, text" if include_bodies else "title, '' AS text"
    query = f"SELECT {columns} FROM '{Path(parquet_path)}'"
    taken = 0
    reader = duckdb.sql(query)
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
