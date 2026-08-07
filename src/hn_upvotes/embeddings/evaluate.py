"""Intrinsic evaluation of the trained embeddings.

Intrinsic scores are a sanity check, not the result. The result is what the embeddings
do on the downstream score prediction. These exist so a broken implementation is caught
before it costs a fusion training run, and so the implementation can be validated
against gensim on the same corpus.

:func:`nearest_neighbours` and :func:`cosine_similarity_matrix` are implemented, because
they need nothing but a trained matrix and they are how the smoke test checks that a word
landed next to its synonym. :func:`analogy_accuracy`, :func:`wordsim_spearman` and
:func:`compare_against_gensim` are still stubs: each needs a downloaded corpus or
evaluation set, which arrives with the first real training run.

Needs the ``train`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hn_upvotes.embeddings.train import TrainedEmbeddings

#: Words checked by hand for nearest-neighbour plausibility on HN vocabulary.
HN_PROBE_WORDS: tuple[str, ...] = ("rust", "yc", "llm", "startup", "kubernetes", "docker")


@dataclass(frozen=True)
class IntrinsicReport:
    """Everything the intrinsic evaluation measures, for one embedding variant."""

    analogy_accuracy: float
    analogy_covered: int
    wordsim_spearman: float
    wordsim_covered: int
    neighbours: dict[str, list[str]]


def nearest_neighbours(
    embeddings: TrainedEmbeddings,
    word: str,
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return the ``k`` nearest words by cosine similarity, with their similarities.

    The query word itself is excluded, since its similarity to itself is 1 by definition
    and would take the first slot in every result.
    """
    vocabulary = embeddings.vocabulary
    if word not in vocabulary.word_to_index:
        raise KeyError(f"{word!r} is not in the vocabulary")
    index = vocabulary.word_to_index[word]
    similarity = cosine_similarity_matrix(embeddings.matrix, embeddings.matrix[index][None, :])[0]
    similarity[index] = -np.inf
    ranked = np.argsort(-similarity)[:k]
    return [(vocabulary.index_to_word[int(i)], float(similarity[int(i)])) for i in ranked]


def analogy_accuracy(embeddings: TrainedEmbeddings, questions_path: Path) -> tuple[float, int]:
    """Accuracy on the Google analogy set, and how many questions were in vocabulary.

    Uses 3CosAdd with the three input words excluded from the candidate set. Coverage is
    returned alongside accuracy because a small vocabulary can post a flattering score
    on the handful of questions it can answer.
    """
    raise NotImplementedError


def wordsim_spearman(embeddings: TrainedEmbeddings, pairs_path: Path) -> tuple[float, int]:
    """Spearman correlation against WordSim-353 human ratings, and pair coverage."""
    raise NotImplementedError


def compare_against_gensim(
    embeddings: TrainedEmbeddings,
    corpus_path: Path,
) -> dict[str, float]:
    """Train gensim on the same corpus and settings, and compare intrinsic scores.

    The correctness check for the from-scratch implementation. Matching gensim within
    noise on text8 is the gate for scaling up to the Wikipedia subset. Not a benchmark
    of speed: gensim's Cython will win that by a wide margin and that is expected.
    """
    raise NotImplementedError


def evaluate(
    embeddings: TrainedEmbeddings,
    analogy_path: Path,
    wordsim_path: Path,
    probe_words: tuple[str, ...] = HN_PROBE_WORDS,
) -> IntrinsicReport:
    """Run every intrinsic check and return one report."""
    raise NotImplementedError


def cosine_similarity_matrix(matrix: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Cosine similarity of every query row against every embedding row.

    Returns (queries, vocabulary). A zero row, which is what an untrained word looks
    like, gets a norm of 1 rather than dividing by zero, so it scores 0 against
    everything instead of producing a NaN that poisons the ranking.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    if queries.ndim == 1:
        queries = queries[None, :]
    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norms = np.linalg.norm(queries, axis=1)
    matrix_norms[matrix_norms == 0] = 1.0
    query_norms[query_norms == 0] = 1.0
    return (queries / query_norms[:, None]) @ (matrix / matrix_norms[:, None]).T
