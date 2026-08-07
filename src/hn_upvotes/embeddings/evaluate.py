"""Intrinsic evaluation of the trained embeddings.

Intrinsic scores are a sanity check, not the result. The result is what the embeddings
do on the downstream score prediction. These exist so a broken implementation is caught
before it costs a fusion training run, and so the implementation can be validated
against gensim on the same corpus.

The comparison against gensim is the chain's stage 0 gate. It works by scoring both models
on the same task with the same function, so the task is supplied by the caller rather than
fixed here: the real text8 run passes :func:`analogy_task`, and the dry run passes a
synthetic corpus's own planted answer. That is what lets one gate serve both without needing
a downloaded evaluation set to be present.

Needs the ``train`` extra.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hn_upvotes.embeddings.train import SGNSConfig, TrainedEmbeddings
from hn_upvotes.training.metrics import spearman

#: Words checked by hand for nearest-neighbour plausibility on HN vocabulary.
HN_PROBE_WORDS: tuple[str, ...] = ("rust", "yc", "llm", "startup", "kubernetes", "docker")

#: How far below gensim a task score may sit before the gate calls our vectors meaningfully
#: worse, as a fraction of gensim's score. At 0.20, scoring 0.48 against gensim's 0.60 passes
#: and 0.40 fails. This is the gate's main rule.
MAXIMUM_TASK_SHORTFALL = 0.20

#: Floor on the share of each word's ten nearest neighbours that both models agree on.
#: **Calibrated on the synthetic topic corpus, not on text8.** A correct implementation
#: scores 0.53 there and a randomised matrix scores 0.02, so the exact cut does not matter
#: much and this sits well clear of the broken end. Two correct implementations do not agree
#: closely on neighbour lists even when both are good, which is why the floor is low and why
#: the task comparison rather than this number is what the gate leans on. Re-derive on text8
#: at the first real gate run.
MINIMUM_NEIGHBOUR_OVERLAP = 0.15

#: Rejected as a gate metric, recorded so it is not tried again. Spearman between our cosine
#: similarity and gensim's over randomly sampled word pairs measures almost nothing: on the
#: synthetic topic corpus a correct implementation scores 0.012 while scoring 1.00 on the
#: task, because 97.5% of random pairs are cross-topic and the corpus says nothing about how
#: those should rank. The metric is dominated by pairs whose true answer is undefined.
_RANDOM_PAIR_AGREEMENT_IS_NOT_A_GATE = True


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


def _normalised_rows(matrix: np.ndarray) -> np.ndarray:
    """Unit-length rows, with zero rows left at zero rather than turned into NaN."""
    matrix = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def analogy_accuracy(embeddings: TrainedEmbeddings, questions_path: Path) -> tuple[float, int]:
    """Accuracy on the Google analogy set, and how many questions were in vocabulary.

    Uses 3CosAdd with the three input words excluded from the candidate set. Coverage is
    returned alongside accuracy because a small vocabulary can post a flattering score
    on the handful of questions it can answer.

    The file is the standard ``questions-words.txt``: ``: section`` header lines and
    four-word questions, ``a b c d``, meaning ``a`` is to ``b`` as ``c`` is to ``d``.
    """
    lookup = embeddings.vocabulary.word_to_index
    unit = _normalised_rows(embeddings.matrix)
    correct = 0
    covered = 0
    for line in Path(questions_path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(":"):
            continue
        words = line.lower().split()
        if len(words) != 4 or any(word not in lookup for word in words):
            continue
        covered += 1
        first, second, third, expected = (lookup[word] for word in words)
        query = unit[second] - unit[first] + unit[third]
        scores = unit @ query
        # The three inputs are excluded, or the nearest vector is almost always one of them.
        scores[[first, second, third]] = -np.inf
        if int(np.argmax(scores)) == expected:
            correct += 1
    return (correct / covered if covered else 0.0, covered)


def wordsim_spearman(embeddings: TrainedEmbeddings, pairs_path: Path) -> tuple[float, int]:
    """Spearman correlation against WordSim-353 human ratings, and pair coverage.

    The file is ``word1,word2,score`` with a header line, which is how WordSim-353 ships.
    """
    lookup = embeddings.vocabulary.word_to_index
    unit = _normalised_rows(embeddings.matrix)
    ours: list[float] = []
    theirs: list[float] = []
    for line in Path(pairs_path).read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.replace("\t", ",").split(",")]
        if len(parts) < 3:
            continue
        left, right, rating = parts[0].lower(), parts[1].lower(), parts[2]
        try:
            human = float(rating)
        except ValueError:
            continue  # the header row
        if left not in lookup or right not in lookup:
            continue
        ours.append(float(unit[lookup[left]] @ unit[lookup[right]]))
        theirs.append(human)
    if len(ours) < 2:
        return (0.0, len(ours))
    return (spearman(np.asarray(ours), np.asarray(theirs)), len(ours))


@dataclass(frozen=True)
class GateTask:
    """One scored task both models are put through, so their scores can be compared.

    ``score`` takes a :class:`~hn_upvotes.embeddings.train.TrainedEmbeddings` and returns a
    number where higher is better. The gate does not care what the task is, only that the
    same function scores both models, which is what lets the real text8 run use the Google
    analogy set and the dry run use a synthetic corpus's own planted answer.
    """

    name: str
    score: Callable[[TrainedEmbeddings], float]


@dataclass(frozen=True)
class TaskComparison:
    """One task's score for each model, and whether ours fell meaningfully short."""

    name: str
    ours: float
    gensim: float

    @property
    def shortfall(self) -> float:
        """How far below gensim we scored, as a fraction of gensim's score.

        Negative when we beat gensim. Zero when gensim scored zero, because a task neither
        model can do says nothing about ours.
        """
        if self.gensim == 0:
            return 0.0
        return 1.0 - self.ours / self.gensim


@dataclass(frozen=True)
class GensimComparison:
    """How our vectors compare to gensim's, trained on the same corpus and settings.

    ``tasks``
        The scored comparisons. This is what the gate decides on: same task, same corpus,
        two implementations, and ours must not come in meaningfully below.
    ``neighbour_overlap``
        Mean share of each word's top-``k`` neighbours that both models agree on, over the
        most frequent shared words. A structural cross-check rather than a quality measure.
        Two good implementations still disagree on plenty of neighbour lists, so this is a
        floor against garbage and not a target to maximise.
    ``our_tokens_per_second`` / ``gensim_tokens_per_second``
        Recorded because the ratio is worth knowing, not because it gates anything. gensim's
        Cython is expected to win and does.
    """

    neighbour_overlap: float
    shared_vocabulary: int
    tasks: tuple[TaskComparison, ...] = ()
    our_tokens_per_second: float = float("nan")
    gensim_tokens_per_second: float = float("nan")

    def as_dict(self) -> dict:
        """Flat mapping, for the run manifest."""
        return {
            "neighbour_overlap": self.neighbour_overlap,
            "shared_vocabulary": self.shared_vocabulary,
            "our_tokens_per_second": self.our_tokens_per_second,
            "gensim_tokens_per_second": self.gensim_tokens_per_second,
            "tasks": [
                {
                    "name": task.name,
                    "ours": task.ours,
                    "gensim": task.gensim,
                    "shortfall": task.shortfall,
                }
                for task in self.tasks
            ],
        }


@dataclass(frozen=True)
class GateReport:
    """The stage 0 verdict, and every number behind it."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    comparison: GensimComparison | None = None

    def as_dict(self) -> dict:
        """For the run manifest."""
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "comparison": self.comparison.as_dict() if self.comparison else None,
        }


def train_gensim(lines: Iterable[list[str]], config: SGNSConfig):
    """Train gensim's Word2Vec on the same corpus with matching settings.

    Every hyperparameter is passed explicitly, including two whose defaults would otherwise
    make the comparison meaningless:

    * ``sample`` defaults to ``1e-3`` in gensim against the paper's ``1e-5``. Left alone the
      gate would compare a model that kept 82% of its tokens against one that kept 31%.
    * ``workers=1``, because gensim's default is multi-threaded Hogwild and a run is then
      not reproducible from the seed.

    One difference cannot be passed away: gensim's subsampling formula is
    ``sqrt(t/f) + t/f`` where the paper's is ``sqrt(t/f)``, so gensim keeps slightly more of
    the middle of the distribution. See
    :func:`~hn_upvotes.embeddings.negative_sampling.subsample_probabilities`.
    """
    from gensim.models import Word2Vec

    return Word2Vec(
        sentences=lines,
        vector_size=config.dimension,
        window=config.window,
        min_count=config.min_count,
        max_final_vocab=config.vocabulary_cap,
        sg=1 if config.objective == "skipgram" else 0,
        negative=config.negative_samples,
        ns_exponent=config.noise_power,
        sample=config.subsample_threshold,
        alpha=config.learning_rate,
        min_alpha=config.min_learning_rate,
        epochs=config.epochs,
        seed=config.seed,
        workers=1,
    )


def analogy_task(questions_path: Path) -> GateTask:
    """Google analogy accuracy as a gate task. The real text8 gate's primary measure."""
    return GateTask(
        name="analogy accuracy",
        score=lambda embeddings: analogy_accuracy(embeddings, questions_path)[0],
    )


def wordsim_task(pairs_path: Path) -> GateTask:
    """WordSim-353 correlation as a gate task."""
    return GateTask(
        name="wordsim-353 spearman",
        score=lambda embeddings: wordsim_spearman(embeddings, pairs_path)[0],
    )


def synonym_recall(embeddings: TrainedEmbeddings, pairs: Sequence[tuple[str, str]]) -> float:
    """Share of ``pairs`` whose first word has the second as its nearest neighbour.

    The scoring rule for a corpus with a planted answer. Pairs with a word missing from the
    vocabulary count as misses, because a model that dropped the word cannot be said to have
    found it.
    """
    found = 0
    for first, second in pairs:
        if first not in embeddings.vocabulary.word_to_index:
            continue
        neighbours = nearest_neighbours(embeddings, first, k=1)
        if neighbours and neighbours[0][0] == second:
            found += 1
    return found / len(pairs) if pairs else 0.0


def neighbour_topic_purity(
    embeddings: TrainedEmbeddings, topic_of: dict[str, int], k: int = 10
) -> float:
    """Share of each word's ``k`` nearest neighbours that come from its own topic.

    The scoring rule for the synthetic topic corpus. Chance level is one over the number of
    topics, so 40 topics puts chance at 0.025.
    """
    scores = []
    for word, topic in topic_of.items():
        if word not in embeddings.vocabulary.word_to_index:
            continue
        neighbours = [name for name, _ in nearest_neighbours(embeddings, word, k=k)]
        if not neighbours:
            continue
        scores.append(sum(topic_of.get(name, -1) == topic for name in neighbours) / len(neighbours))
    return float(np.mean(scores)) if scores else 0.0


def compare_with_gensim_vectors(
    embeddings: TrainedEmbeddings,
    gensim_vectors,
    tasks: Sequence[GateTask] = (),
    neighbour_words: int = 200,
    neighbour_k: int = 10,
    gensim_tokens_per_second: float = float("nan"),
) -> GensimComparison:
    """Compare an already-trained pair of models. See :class:`GensimComparison`."""
    shared = [
        word
        for word in embeddings.vocabulary.index_to_word
        if word in gensim_vectors.key_to_index and word != "<unk>"
    ]
    if len(shared) < 3:
        raise ValueError(f"only {len(shared)} shared words: nothing to compare")

    ours = _normalised_rows(
        np.stack([embeddings.matrix[embeddings.vocabulary.word_to_index[w]] for w in shared])
    )
    theirs = _normalised_rows(np.stack([gensim_vectors[w] for w in shared]))

    # The most frequent shared words, which are the ones both models had enough evidence
    # for. Ranking a rare word's neighbours is noise in both models.
    counts = np.asarray(
        [embeddings.vocabulary.counts[embeddings.vocabulary.word_to_index[w]] for w in shared]
    )
    probes = np.argsort(-counts)[: min(neighbour_words, len(shared) - 1)]
    k = min(neighbour_k, len(shared) - 2)
    overlaps = [
        len(
            _top_indices(ours @ ours[probe], probe, k)
            & _top_indices(theirs @ theirs[probe], probe, k)
        )
        / k
        for probe in probes
    ]

    gensim_embeddings = _as_trained_embeddings(gensim_vectors, embeddings.config)
    scored = tuple(
        TaskComparison(
            name=task.name,
            ours=float(task.score(embeddings)),
            gensim=float(task.score(gensim_embeddings)),
        )
        for task in tasks
    )

    return GensimComparison(
        neighbour_overlap=float(np.mean(overlaps)) if overlaps else 0.0,
        shared_vocabulary=len(shared),
        tasks=scored,
        our_tokens_per_second=embeddings.tokens_per_second,
        gensim_tokens_per_second=gensim_tokens_per_second,
    )


def _top_indices(scores: np.ndarray, exclude: int, k: int) -> set[int]:
    """Indices of the ``k`` highest scores, with ``exclude`` removed."""
    scores = scores.copy()
    scores[exclude] = -np.inf
    return {int(i) for i in np.argpartition(-scores, k)[:k]}


def _as_trained_embeddings(gensim_vectors, config: SGNSConfig) -> TrainedEmbeddings:
    """Wrap gensim's vectors in our artefact type so the same evaluators run on both."""
    from hn_upvotes.data.preprocess import Vocabulary

    words = list(gensim_vectors.index_to_key)
    return TrainedEmbeddings(
        matrix=np.stack([gensim_vectors[word] for word in words]),
        vocabulary=Vocabulary(
            word_to_index={word: i for i, word in enumerate(words)},
            index_to_word=words,
            counts=np.asarray(
                [gensim_vectors.get_vecattr(word, "count") for word in words], dtype=np.int64
            ),
            unknown_index=-1,
        ),
        config=config,
        tokens_per_second=float("nan"),
    )


def gate_against_gensim(
    embeddings: TrainedEmbeddings,
    lines: Iterable[list[str]],
    tasks: Sequence[GateTask] = (),
) -> GateReport:
    """Stage 0. Train gensim on the same corpus, compare, and decide whether to carry on.

    The single most valuable thing in the chain. Without it a bug costs a night of
    Wikipedia training and shows up in the morning; with it the chain stops in minutes and
    the machine sits idle instead, which is far cheaper.

    Two rules, and the first is the one that matters:

    1. On every task, our score must be within :data:`MAXIMUM_TASK_SHORTFALL` of gensim's.
       Same corpus, same settings, same scoring function, so a gap is our implementation.
    2. Our nearest-neighbour lists must overlap gensim's by at least
       :data:`MINIMUM_NEIGHBOUR_OVERLAP`, which catches vectors that score well by accident.

    Fails closed. If gensim will not train, or there is no shared vocabulary to compare
    over, that is an aborted gate rather than a passed one. A gate that could not run has
    not passed.
    """
    try:
        started = time.perf_counter()
        gensim_model = train_gensim(lines, embeddings.config)
        gensim_elapsed = time.perf_counter() - started
        gensim_rate = (
            gensim_model.corpus_total_words * embeddings.config.epochs / gensim_elapsed
            if gensim_elapsed > 0
            else float("inf")
        )
        comparison = compare_with_gensim_vectors(
            embeddings, gensim_model.wv, tasks=tasks, gensim_tokens_per_second=gensim_rate
        )
    except Exception as error:  # noqa: BLE001 - a gate that cannot run has not passed
        return GateReport(
            passed=False, reasons=[f"gate could not run: {type(error).__name__}: {error}"]
        )

    reasons = []
    for task in comparison.tasks:
        if task.shortfall > MAXIMUM_TASK_SHORTFALL:
            reasons.append(
                f"{task.name}: ours {task.ours:.3f} against gensim's {task.gensim:.3f}, "
                f"{task.shortfall:.1%} short of it"
            )
    if comparison.neighbour_overlap < MINIMUM_NEIGHBOUR_OVERLAP:
        reasons.append(
            f"nearest-neighbour overlap with gensim is {comparison.neighbour_overlap:.3f}, "
            f"below the {MINIMUM_NEIGHBOUR_OVERLAP} floor"
        )
    if not comparison.tasks:
        reasons.append("no gate task was supplied, so quality was never compared")
    return GateReport(passed=not reasons, reasons=reasons, comparison=comparison)


def compare_against_gensim(
    embeddings: TrainedEmbeddings,
    corpus_path: Path,
    tasks: Sequence[GateTask] = (),
) -> dict:
    """Train gensim on the same corpus and settings, and compare intrinsic scores.

    The correctness check for the from-scratch implementation. Matching gensim within
    noise on text8 is the gate for scaling up to the Wikipedia subset. Not a benchmark
    of speed: gensim's Cython will win that by a wide margin and that is expected.

    Reads the whole corpus into memory, because gensim wants a sequence it can pass over
    several times. Use :func:`gate_against_gensim` directly on a corpus too large for that.
    """
    from hn_upvotes.data.preprocess import stream_corpus

    lines = list(stream_corpus(Path(corpus_path)))
    gensim_model = train_gensim(lines, embeddings.config)
    return compare_with_gensim_vectors(embeddings, gensim_model.wv, tasks=tasks).as_dict()


def evaluate(
    embeddings: TrainedEmbeddings,
    analogy_path: Path,
    wordsim_path: Path,
    probe_words: tuple[str, ...] = HN_PROBE_WORDS,
) -> IntrinsicReport:
    """Run every intrinsic check and return one report.

    Probe words that are not in the vocabulary are skipped rather than raising, because the
    HN probe list is deliberately specific and a Wikipedia-only variant will not have ``yc``.
    """
    accuracy, analogy_covered = (
        analogy_accuracy(embeddings, analogy_path) if Path(analogy_path).exists() else (0.0, 0)
    )
    correlation, wordsim_covered = (
        wordsim_spearman(embeddings, wordsim_path) if Path(wordsim_path).exists() else (0.0, 0)
    )
    neighbours: dict[str, list[str]] = {}
    for word in probe_words:
        if word in embeddings.vocabulary.word_to_index:
            neighbours[word] = [w for w, _ in nearest_neighbours(embeddings, word, k=10)]
    return IntrinsicReport(
        analogy_accuracy=accuracy,
        analogy_covered=analogy_covered,
        wordsim_spearman=correlation,
        wordsim_covered=wordsim_covered,
        neighbours=neighbours,
    )


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
