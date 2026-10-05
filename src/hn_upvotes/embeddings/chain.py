"""The training chain: two objectives, four stages each, without time limits.

Four stages per objective, and it is an order rather than a list:

0. **gate** Train on text8, train gensim on the same corpus with the same settings, compare.
   **A failure here aborts that objective.** That is the point of it: a bug otherwise costs
   a night of Wikipedia training and shows up in the morning, where this costs ten minutes
   and leaves the machine idle instead.
1. **wiki-only** All eligible text from the English Wikipedia snapshot.
   Both objectives read the same corpus without a token limit.
2. **fine-tuned** Stage 1's vectors, carried on over Hacker News titles and bodies at a tenth
   of the learning rate. Words Hacker News has and Wikipedia does not start random.
3. **hn-only** Random initialisation, Hacker News text only. Last on purpose: it needs
   nothing from stages 1 or 2, so a Wikipedia failure does not cost us this variant.

Stages 1 to 3 produce three embedding variants per objective, so one run produces **six**:
Skip-gram and CBOW, each as ``wiki-only``, ``fine-tuned`` and ``hn-only``. Every artefact is
named ``{objective}-{stage}.npz`` and every manifest record carries its objective, because
the whole point of running both is comparing them and a variant whose objective is a guess
compares nothing.

The objectives are independent all the way down. A CBOW gate that aborts costs the three
CBOW variants and none of the Skip-gram ones, and the reverse.

Everything unattended operation needs is here: a per-epoch checkpoint, ``--resume`` from the
newest one, per-stage failure isolation, and one JSON manifest written as it goes so the
morning's question is answered by one file rather than a log.

``--dry-run`` walks every stage of both objectives on synthetic corpora in about 20 seconds,
which is how the chain is verified without spending a night on it.

Needs the ``train`` extra, and the ``data`` extra for the Hacker News stages.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from hn_upvotes.data.preprocess import build_vocabulary
from hn_upvotes.data.splits import DEFAULT_VALIDATION_START
from hn_upvotes.embeddings import corpora
from hn_upvotes.embeddings.checkpoint import newest_checkpoint
from hn_upvotes.embeddings.evaluate import (
    GateReport,
    GateTask,
    analogy_task,
    gate_against_gensim,
    neighbour_topic_purity,
    wordsim_task,
)
from hn_upvotes.embeddings.train import (
    SGNSConfig,
    TrainedEmbeddings,
    select_device,
    train_embeddings_from_lines,
)

STAGE_GATE = "gate"
STAGE_WIKI = "wiki-only"
STAGE_FINETUNE = "fine-tuned"
STAGE_HN = "hn-only"

OBJECTIVE_SKIPGRAM = "skipgram"
OBJECTIVE_CBOW = "cbow"

#: Both objectives, Skip-gram first. Both read the same Wikipedia corpus.
BOTH_OBJECTIVES: tuple[str, ...] = (OBJECTIVE_SKIPGRAM, OBJECTIVE_CBOW)

#: The three variants each objective produces. The gate is not one of them: it keeps nothing.
VARIANT_STAGES: tuple[str, ...] = (STAGE_WIKI, STAGE_FINETUNE, STAGE_HN)

#: Measured on this machine with ``make throughput``: **Skip-gram**, dimension 300, batch
#: 1024, CPU with sparse gradients, over a 2M-token Zipfian synthetic corpus. In-vocabulary
#: corpus tokens per second, keyed by the number of negative samples.
#:
#: k=5 is 1.80x faster than k=15, not the 2.7x a count of dot products predicts (6 against
#: 16). The gap is fixed per-batch overhead that does not scale with k.
MEASURED_TOKENS_PER_SECOND: dict[int, int] = {5: 138_362, 15: 77_019}

#: How much faster CBOW is per corpus token, measured at dimension 300, ``k=15``, batch 1024,
#: CPU with sparse gradients: 181,487 tokens/s against Skip-gram's 81,603. The reason is the
#: example count, not the arithmetic: with a window of 5, Skip-gram produces up to ten
#: training examples per position and CBOW exactly one.
#:
#: Not re-measured at ``k=5``. It is used only to say how long the CBOW stages are expected
#: to take, never to size a corpus, so a wrong number here costs an estimate rather than an
#: artefact.
CBOW_SPEEDUP = 181_487 / 81_603

#: text8's token count, from the file: 100,000,000 bytes, 17,005,207 tokens.
TEXT8_TOKENS = 17_005_207

#: The synthetic corpus every dry-run stage reads. **Sized for CBOW, not for Skip-gram.**
#:
#: With a window of 5, Skip-gram produces up to ten training examples per position and CBOW
#: exactly one, so on the same corpus CBOW takes 3.8x fewer optimiser steps: 25,134 against
#: 95,347 per epoch on the corpus this replaced. That corpus was sized when only Skip-gram
#: ran, and adding CBOW showed it up immediately: CBOW scored a topic purity of 0.392 against
#: gensim's 1.000 and aborted its own gate, purely from having seen too little.
#:
#: 10 topics of 15 words over 40,000 tokens is the cheapest shape measured where **both**
#: objectives reach a purity of 1.000 against gensim's 1.000 in 3 epochs. Chance is 1/10, and
#: a deliberately broken matrix scores 0.104, so the gate still has the margin it needs.
DRY_RUN_CORPUS = {"topics": 10, "words_per_topic": 15, "lines_per_topic": 400, "line_length": 10}

#: Throughput haircut for running at the full 100,000-word vocabulary rather than the 40,576
#: the calibration corpus produced. Measured at 4%: 109,390 pairs/s at 40,576 words against
#: 105,142 at 100,000.
LARGE_VOCABULARY_FACTOR = 0.96

#: The fine-tuning stage's learning rate, as a fraction of the from-scratch rate. A tenth,
#: so Wikipedia's structure is adjusted rather than overwritten by a much smaller corpus.
FINE_TUNE_RATE_FRACTION = 0.1

#: Stage statuses that leave a usable matrix on disk. ``cut_short`` counts because a matrix
#: with less training than configured still beats starting from noise, and
#: ``already_complete`` counts because a resumed stage that had nothing left to do has its
#: artefact from the earlier invocation.
_USABLE_STATUSES = frozenset({"completed", "cut_short", "already_complete"})

#: Set in the environment once the process is running under ``caffeinate``, so re-execing
#: cannot loop.
_CAFFEINATE_SENTINEL = "HN_CHAIN_CAFFEINATED"


@dataclass(frozen=True)
class ChainConfig:
    """Where the chain writes, which corpora it reads, and how many epochs it trains."""

    output_directory: Path = Path("artifacts/embeddings")
    corpus_directory: Path = Path("data/corpora")
    hn_corpus_path: Path = corpora.HN_CORPUS_PATH
    hn_before: str = DEFAULT_VALIDATION_START + "-01"
    analogy_path: Path | None = None
    wordsim_path: Path | None = None

    #: Which objectives to run, in order. Both by default: the CBOW against Skip-gram
    #: comparison is the project's stated experiment, so it is not an option to forget.
    objectives: tuple[Literal["cbow", "skipgram"], ...] = BOTH_OBJECTIVES
    dimension: int = 300
    epochs: int = 5

    dry_run: bool = False
    resume: bool = False
    seed: int = 0

    @property
    def checkpoint_directory(self) -> Path:
        return self.output_directory / "checkpoints"

    @property
    def manifest_path(self) -> Path:
        return self.output_directory / "run-manifest.json"

    def artefact_path(self, objective: str, stage: str) -> Path:
        """``{objective}-{stage}.npz``. Six of these come out of a full run.

        The objective is in the file name rather than only in the manifest, because a
        variant on disk gets loaded by name months later and a ``wiki-only.npz`` that could
        be either objective is a variant nobody can use in the comparison.
        """
        return self.output_directory / f"{objective}-{stage}.npz"

    def checkpoint_stage(self, objective: str, stage: str) -> str:
        """``{objective}-{stage}``, which is what checkpoints are named for.

        Qualifying the checkpoint name is what keeps a resume honest across objectives:
        ``newest_checkpoint`` matches on this string, so a finished Skip-gram Wikipedia stage
        cannot be picked up as a starting point for CBOW's.
        """
        return f"{objective}-{stage}"


def stage_seconds(objective: str, tokens: int, epochs: int, negative_samples: int) -> float:
    """Estimate training time from token count and measured throughput. Never stop a run."""
    rate = MEASURED_TOKENS_PER_SECOND.get(negative_samples, MEASURED_TOKENS_PER_SECOND[5])
    rate *= LARGE_VOCABULARY_FACTOR
    if objective == OBJECTIVE_CBOW:
        rate *= CBOW_SPEEDUP
    return tokens * epochs / rate


def estimate_training_time(
    config: ChainConfig,
    wikipedia_tokens: int,
    hn_tokens: int = corpora.HN_CORPUS_TOKENS,
    gensim_gate_seconds: float = 0.0,
) -> dict:
    """Estimate stage runtimes from supplied corpus counts, without time limits.

    Downloads, vocabulary preparation, evaluation, and real reader overhead are excluded.
    Supply a measured gensim runtime when available; its default is not an estimate.
    """
    if wikipedia_tokens < 0 or hn_tokens < 0 or gensim_gate_seconds < 0:
        raise ValueError("token counts and gensim runtime cannot be negative")
    tokens_by_stage = {
        STAGE_GATE: TEXT8_TOKENS,
        STAGE_WIKI: wikipedia_tokens,
        STAGE_FINETUNE: hn_tokens,
        STAGE_HN: hn_tokens,
    }
    rows = []
    for objective in config.objectives:
        for stage, tokens in tokens_by_stage.items():
            negatives = 5 if stage == STAGE_WIKI else 15
            expected = stage_seconds(objective, tokens, config.epochs, negatives)
            if stage == STAGE_GATE:
                expected += gensim_gate_seconds
            rows.append(
                {
                    "objective": objective,
                    "stage": stage,
                    "corpus_tokens": tokens,
                    "negative_samples": negatives,
                    "expected_seconds": round(expected, 1),
                }
            )
    return {
        "objectives": list(config.objectives),
        "epochs": config.epochs,
        "wikipedia_tokens": wikipedia_tokens,
        "hn_tokens": hn_tokens,
        "stages": rows,
        "expected_hours": round(sum(row["expected_seconds"] for row in rows) / 3600, 2),
    }


class RunManifest:
    """One JSON file describing the night, rewritten after every state change.

    Rewritten rather than appended so it is always valid JSON, and written to a temporary
    name then moved into place so a crash mid-write cannot truncate it. The captain reads
    this file in the morning instead of a log.
    """

    def __init__(self, path: Path, config: ChainConfig, device: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The effective settings, not the requested ones. A dry run shrinks the dimension and
        # the epoch count, and a manifest saying 300 dimensions over a run that used 32 would
        # be worse than useless.
        effective = _stage_config(config, STAGE_HN, config.objectives[0])
        self.data: dict = {
            "started": _now(),
            "finished": None,
            "dry_run": config.dry_run,
            "resume": config.resume,
            "device": device,
            "objectives": list(config.objectives),
            "dimension": effective.dimension,
            "epochs": effective.epochs,
            "batch_size": effective.batch_size,
            "subsample_threshold": effective.subsample_threshold,
            "outcome": "running",
            "sleep_assertion": sleep_assertion_state(),
            "objective_outcomes": {},
            "stages": [],
        }
        self.write()

    def start_stage(self, objective: str, stage: str, detail: dict | None = None) -> dict:
        """Open a record for one stage of one objective.

        ``name`` is the qualified ``{objective}-{stage}``, and ``objective`` and ``stage`` are
        both carried separately so the manifest can be grouped either way without parsing a
        string back apart.
        """
        record: dict = {
            "name": f"{objective}-{stage}",
            "objective": objective,
            "stage": stage,
            "started": _now(),
            "finished": None,
            "status": "running",
            "detail": detail or {},
        }
        self.data["stages"].append(record)
        self.write()
        return record

    def finish_stage(self, record: dict, status: str, **fields) -> None:
        record["finished"] = _now()
        record["status"] = status
        record.update(fields)
        self.write()

    def record_failure(self, record: dict, error: BaseException) -> None:
        """Status ``failed``, with the traceback, so the morning does not need the log."""
        self.finish_stage(
            record,
            "failed",
            failure={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )

    def finish(self, outcome: str) -> None:
        self.data["finished"] = _now()
        self.data["outcome"] = outcome
        self.write()

    def write(self) -> None:
        temporary = self.path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(self.data, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sleep_assertion_state() -> dict:
    """Whether this run wrapped itself in ``caffeinate``, and what macOS currently reports.

    ``wrapped`` is definitive: the chain sets a sentinel in the environment when it re-execs
    itself under ``caffeinate``, so this reports what the chain did rather than inferring it.

    ``system_wide`` is context only. Other processes hold the same flags, so the totals are
    not on their own evidence that this run is the reason for them.
    """
    if sys.platform != "darwin" or shutil.which("pmset") is None:
        return {"available": False, "wrapped": os.environ.get(_CAFFEINATE_SENTINEL) == "1"}
    try:
        output = subprocess.run(
            ["pmset", "-g", "assertions"], capture_output=True, text=True, timeout=15, check=True
        ).stdout
    except (subprocess.SubprocessError, OSError) as error:
        return {"available": True, "error": f"{type(error).__name__}: {error}"}

    tracked = (
        "PreventUserIdleSystemSleep",
        "PreventSystemSleep",
        "PreventUserIdleDisplaySleep",
        "PreventDiskIdle",
    )
    system_wide = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] in tracked:
            system_wide[parts[0]] = parts[1] == "1"
    return {
        "available": True,
        "wrapped": os.environ.get(_CAFFEINATE_SENTINEL) == "1",
        "flags": "-ism",
        "system_wide": system_wide,
    }


@dataclass
class StageOutcome:
    """What one stage produced, for the stage after it to decide on."""

    status: str
    embeddings: TrainedEmbeddings | None = None
    gate: GateReport | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """True when a later stage can build on this one's vectors.

        An older stage cut short by a deadline still counts. Its matrix had less training than it
        was configured for, which is recorded, but it is real and fine-tuning from it beats
        fine-tuning from noise.
        """
        return self.embeddings is not None and self.status in _USABLE_STATUSES


def _stage_config(config: ChainConfig, stage: str, objective: str) -> SGNSConfig:
    """The training settings for one stage of one objective.

    Two deliberate differences from the defaults. ``k`` drops from 15 to 5 for Wikipedia,
    which is what the paper recommends for a large corpus. The fine-tuning stage drops the
    learning rate to a tenth so it adjusts Wikipedia's structure rather than overwriting it.

    Everything else is held identical across the two objectives, including the seed, because
    the comparison is only about the objective if nothing else moved.
    """
    base = replace(
        SGNSConfig(),
        objective=objective,
        dimension=config.dimension,
        epochs=config.epochs,
        seed=config.seed,
        device=select_device().type,
    )
    if config.dry_run:
        # Small enough that four stages finish in seconds, and subsampling turned down
        # because a synthetic corpus of 40,000 tokens has nothing to thin out.
        base = replace(
            base,
            dimension=32,
            epochs=3,
            batch_size=512,
            subsample_threshold=1e-3,
            vocabulary_cap=2_000,
        )
    if stage == STAGE_WIKI:
        base = replace(base, negative_samples=5)
    if stage == STAGE_FINETUNE:
        base = replace(
            base,
            learning_rate=base.learning_rate * FINE_TUNE_RATE_FRACTION,
            min_learning_rate=base.min_learning_rate * FINE_TUNE_RATE_FRACTION,
        )
    return base


def _train_stage(
    objective: str,
    stage: str,
    reopen_corpus: Callable[[], Iterator[list[str]]],
    stage_config: SGNSConfig,
    chain_config: ChainConfig,
    manifest: RunManifest,
    record: dict,
    initial: TrainedEmbeddings | None = None,
) -> StageOutcome:
    """Build the vocabulary, train all configured epochs, checkpoint, and save the artefact."""
    vocabulary = build_vocabulary(
        reopen_corpus(),
        min_count=stage_config.min_count,
        max_size=stage_config.vocabulary_cap,
    )
    record["detail"]["vocabulary"] = len(vocabulary)
    record["detail"]["corpus_tokens"] = int(vocabulary.counts.sum())
    manifest.write()

    checkpoint_stage = chain_config.checkpoint_stage(objective, stage)
    artefact_path = chain_config.artefact_path(objective, stage)
    resume = (
        newest_checkpoint(chain_config.checkpoint_directory, checkpoint_stage)
        if chain_config.resume
        else None
    )
    if resume is not None:
        if (
            resume.vocabulary.index_to_word != vocabulary.index_to_word
            or resume.vocabulary.counts.tolist() != vocabulary.counts.tolist()
        ):
            raise ValueError(
                "checkpoint vocabulary or counts differ from the current corpus; "
                "use a new output directory when replacing a limited corpus"
            )
        record["detail"]["resumed_from_epoch"] = resume.epochs_done

    trained = train_embeddings_from_lines(
        reopen_corpus,
        vocabulary,
        stage_config,
        output_path=artefact_path,
        initial=initial,
        checkpoint_directory=chain_config.checkpoint_directory,
        stage=checkpoint_stage,
        resume=resume,
        progress_every=5_000,
    )
    if trained.cut_short:
        status = "cut_short"
    elif trained.tokens_trained == 0:
        # Resumed from a checkpoint that had already finished. Nothing trained, so saying
        # "completed" would claim work this invocation did not do.
        status = "already_complete"
    else:
        status = "completed"
    manifest.finish_stage(
        record,
        status,
        artefact=str(artefact_path),
        tokens_per_second=(round(trained.tokens_per_second, 1) if trained.tokens_trained else None),
        tokens_trained=trained.tokens_trained,
        epochs_completed=trained.epochs_completed,
        epochs_configured=stage_config.epochs,
        epoch_losses=[round(x, 4) for x in trained.epoch_losses],
        partial_epoch_loss=(round(trained.partial_epoch_loss, 4) if trained.cut_short else None),
        negative_samples=stage_config.negative_samples,
        learning_rate=stage_config.learning_rate,
    )
    return StageOutcome(status=status, embeddings=trained)


def _gate_corpus_and_task(
    config: ChainConfig,
) -> tuple[Callable[[], Iterator[list[str]]], list[GateTask], dict]:
    """The gate's corpus and how both models are scored on it.

    Real run: text8, scored on the Google analogy set and WordSim-353 where those files are
    present. Dry run: the synthetic topic corpus, scored on topic purity, because it needs no
    network and has a known right answer.
    """
    if config.dry_run:
        lines, topic_of = corpora.topic_corpus(**DRY_RUN_CORPUS)
        tasks = [
            GateTask(
                name="topic purity",
                score=lambda embeddings: neighbour_topic_purity(embeddings, topic_of),
            )
        ]
        detail = {
            "corpus": "synthetic topic corpus",
            "lines": len(lines),
            "tokens": sum(len(line) for line in lines),
            "chance_level": round(1 / DRY_RUN_CORPUS["topics"], 3),
        }
        return (lambda: iter(lines)), tasks, detail

    text8_path = corpora.download_text8(config.corpus_directory)
    tasks = []
    if config.analogy_path and Path(config.analogy_path).exists():
        tasks.append(analogy_task(config.analogy_path))
    if config.wordsim_path and Path(config.wordsim_path).exists():
        tasks.append(wordsim_task(config.wordsim_path))
    return (
        lambda: corpora.stream_text8(text8_path),
        tasks,
        {"corpus": str(text8_path)},
    )


def _run_gate(config: ChainConfig, manifest: RunManifest, objective: str) -> StageOutcome:
    """Stage 0. Train, compare against gensim, and decide whether this objective goes ahead.

    Run once per objective, because it is validating that objective's implementation. CBOW
    and Skip-gram differ by one line of ``forward``, and that one line is the masked average,
    which is the sharp edge of the whole implementation. A Skip-gram gate says nothing about
    it. gensim is trained with ``sg`` set to match, so each objective is compared against its
    own reference.
    """
    reopen, tasks, detail = _gate_corpus_and_task(config)
    record = manifest.start_stage(objective, STAGE_GATE, detail)
    stage_config = _stage_config(config, STAGE_GATE, objective)

    outcome = _train_stage(
        objective,
        STAGE_GATE,
        reopen,
        stage_config,
        config,
        manifest,
        record,
    )
    if outcome.embeddings is None:
        return StageOutcome(status="failed", notes=["gate produced no vectors"])

    gate = gate_against_gensim(outcome.embeddings, _Reiterable(reopen), tasks=tasks)
    manifest.finish_stage(
        record,
        "passed" if gate.passed else "aborted",
        gate=gate.as_dict(),
        artefact=str(config.artefact_path(objective, STAGE_GATE)),
        tokens_per_second=(
            round(outcome.embeddings.tokens_per_second, 1)
            if outcome.embeddings.tokens_trained
            else None
        ),
        epochs_completed=outcome.embeddings.epochs_completed,
        epochs_configured=stage_config.epochs,
        epoch_losses=[round(x, 4) for x in outcome.embeddings.epoch_losses],
    )
    return StageOutcome(
        status="passed" if gate.passed else "aborted",
        embeddings=outcome.embeddings,
        gate=gate,
    )


class _Reiterable:
    """Wraps a corpus factory so gensim can pass over it more than once.

    gensim's ``Word2Vec`` reads the corpus once to count the vocabulary and once per epoch. A
    generator is exhausted after the first pass and gensim then trains on nothing, silently.
    """

    def __init__(self, factory: Callable[[], Iterator[list[str]]]) -> None:
        self._factory = factory

    def __iter__(self) -> Iterator[list[str]]:
        return iter(self._factory())


def _run_wikipedia(config: ChainConfig, manifest: RunManifest, objective: str) -> StageOutcome:
    """Train on the shared English Wikipedia corpus without time or token limits."""
    stage_config = _stage_config(config, STAGE_WIKI, objective)
    # A new name prevents an old time-limited subset from being reused as the full corpus.
    corpus_path = config.corpus_directory / "wikipedia-20231101-en.txt"
    detail = {
        "corpus": "synthetic" if config.dry_run else corpora.WIKIPEDIA_PARQUET_GLOB,
        "corpus_path": None if config.dry_run else str(corpus_path),
    }
    record = manifest.start_stage(objective, STAGE_WIKI, detail)

    if config.dry_run:
        lines, _ = corpora.topic_corpus(**DRY_RUN_CORPUS, seed=1)

        def reopen() -> Iterator[list[str]]:
            return iter(lines)
    else:
        if not corpus_path.exists():
            corpora.prepare_wikipedia_corpus(corpus_path)

        def reopen() -> Iterator[list[str]]:
            return corpora.stream_plain_text(corpus_path)

    return _train_stage(
        objective,
        STAGE_WIKI,
        reopen,
        stage_config,
        config,
        manifest,
        record,
    )


def _hn_corpus(config: ChainConfig, seed: int) -> Callable[[], Iterator[list[str]]]:
    """Hacker News titles and bodies, or a synthetic stand-in for the dry run."""
    if config.dry_run:
        lines, _ = corpora.topic_corpus(**DRY_RUN_CORPUS, seed=seed)
        return lambda: iter(lines)
    return lambda: corpora.stream_hn_text(
        config.hn_corpus_path, include_bodies=True, before=config.hn_before
    )


def _run_finetune(
    config: ChainConfig, manifest: RunManifest, objective: str, wikipedia: StageOutcome
) -> StageOutcome:
    """Stage 2. Carry stage 1's vectors on over Hacker News at a tenth of the rate.

    Fine-tunes from **this objective's** Wikipedia vectors. Warm-starting CBOW from
    Skip-gram's matrix would train one objective on another's output and the comparison
    would be meaningless, which is why the outcome is passed in rather than looked up.
    """
    stage_config = _stage_config(config, STAGE_FINETUNE, objective)
    record = manifest.start_stage(
        objective,
        STAGE_FINETUNE,
        {
            "initialised_from": f"{objective}-{STAGE_WIKI}",
            "learning_rate": stage_config.learning_rate,
            "rate_fraction_of_scratch": FINE_TUNE_RATE_FRACTION,
        },
    )
    if not wikipedia.usable:
        note = (
            f"skipped: stage {objective}-{STAGE_WIKI} finished as {wikipedia.status!r}, so "
            f"there are no Wikipedia vectors to fine-tune from. Stage {objective}-{STAGE_HN} "
            f"is unaffected, and so is the other objective."
        )
        manifest.finish_stage(record, "skipped", notes=[note])
        return StageOutcome(status="skipped", notes=[note])

    return _train_stage(
        objective,
        STAGE_FINETUNE,
        _hn_corpus(config, seed=2),
        stage_config,
        config,
        manifest,
        record,
        initial=wikipedia.embeddings,
    )


def _run_hn_only(config: ChainConfig, manifest: RunManifest, objective: str) -> StageOutcome:
    """Stage 3. Random initialisation, Hacker News only. Depends on nothing before it."""
    stage_config = _stage_config(config, STAGE_HN, objective)
    record = manifest.start_stage(objective, STAGE_HN, {"initialised_from": "random"})
    return _train_stage(
        objective,
        STAGE_HN,
        _hn_corpus(config, seed=2),
        stage_config,
        config,
        manifest,
        record,
    )


def run_chain(config: ChainConfig) -> dict:
    """Run every objective in ``config.objectives``, four stages each, and return the manifest.

    Two levels of isolation, and they are separate on purpose:

    * **Between objectives.** A CBOW gate that aborts, or a CBOW stage that raises, costs the
      three CBOW variants and nothing else. Skip-gram's three still run.
    * **Between stages inside an objective.** Stages 1 to 3 are each wrapped, so one blowing
      up is recorded with its traceback and the next one still runs. The gate is the
      exception: it is a gate, and a failed gate stops that objective on purpose.
    """
    config.output_directory.mkdir(parents=True, exist_ok=True)
    device = select_device().type
    manifest = RunManifest(config.manifest_path, config, device)

    for objective in config.objectives:
        try:
            _run_objective(config, manifest, objective)
        except Exception as error:  # noqa: BLE001 - objectives must fail independently
            # The stages inside an objective are already guarded, so reaching here means the
            # bookkeeping around them broke. Recording it and carrying on is still right: the
            # next objective has nothing to do with this one.
            record = manifest.start_stage(
                objective, "objective", {"note": "crashed outside a stage"}
            )
            manifest.record_failure(record, error)
            _record_objective_outcome(manifest, objective, "failed", [])

    outcomes = manifest.data["objective_outcomes"]
    produced = [name for outcome in outcomes.values() for name in outcome["variants_produced"]]
    cut_short = [name for outcome in outcomes.values() for name in outcome["variants_cut_short"]]
    expected = len(config.objectives) * len(VARIANT_STAGES)
    if len(produced) == expected and not cut_short:
        overall = "completed"
    elif produced:
        overall = "partial"
    elif all(outcome["outcome"] == "aborted_at_gate" for outcome in outcomes.values()):
        overall = "aborted_at_gate"
    else:
        overall = "failed"
    manifest.finish(overall)
    manifest.data["variants_produced"] = produced
    manifest.data["variants_cut_short"] = cut_short
    manifest.data["variants_expected"] = expected
    manifest.write()
    return manifest.data


def _run_objective(config: ChainConfig, manifest: RunManifest, objective: str) -> dict:
    """One objective's four stages, and its own entry in the manifest's outcome map."""
    try:
        gate = _run_gate(config, manifest, objective)
    except Exception as error:  # noqa: BLE001 - a gate that crashed has not passed
        record = manifest.start_stage(objective, STAGE_GATE, {"note": "crashed before reporting"})
        manifest.record_failure(record, error)
        gate = StageOutcome(status="failed")

    if gate.status != "passed":
        reasons = gate.gate.reasons if gate.gate else gate.notes
        for stage in VARIANT_STAGES:
            record = manifest.start_stage(objective, stage)
            manifest.finish_stage(
                record,
                "skipped",
                notes=[f"skipped: the {objective}-{STAGE_GATE} stage did not pass"],
            )
        return _record_objective_outcome(manifest, objective, "aborted_at_gate", list(reasons))

    wikipedia = _run_stage_guarded(_run_wikipedia, config, manifest, objective, STAGE_WIKI)
    _run_stage_guarded(
        lambda c, m, o: _run_finetune(c, m, o, wikipedia),
        config,
        manifest,
        objective,
        STAGE_FINETUNE,
    )
    _run_stage_guarded(_run_hn_only, config, manifest, objective, STAGE_HN)
    return _record_objective_outcome(manifest, objective, None, [])


def _record_objective_outcome(
    manifest: RunManifest, objective: str, forced: str | None, gate_reasons: list[str]
) -> dict:
    """Summarise one objective's variants into the manifest, and return that summary."""
    statuses = {
        stage["stage"]: stage["status"]
        for stage in manifest.data["stages"]
        if stage.get("objective") == objective
    }
    produced = [
        f"{objective}-{stage}"
        for stage in VARIANT_STAGES
        if statuses.get(stage) in _USABLE_STATUSES
    ]
    cut_short = [
        f"{objective}-{stage}" for stage in VARIANT_STAGES if statuses.get(stage) == "cut_short"
    ]
    # A stage that ran out of clock produced a usable matrix but not the run it was asked
    # for, so the objective is partial. Reporting it as completed would be the exact thing
    # the cut_short flag exists to prevent.
    if forced is not None:
        outcome = forced
    elif len(produced) == len(VARIANT_STAGES) and not cut_short:
        outcome = "completed"
    elif produced:
        outcome = "partial"
    else:
        outcome = "failed"
    summary = {
        "outcome": outcome,
        "variants_produced": produced,
        "variants_cut_short": cut_short,
        "gate_reasons": gate_reasons,
    }
    manifest.data["objective_outcomes"][objective] = summary
    manifest.write()
    return summary


def _run_stage_guarded(
    runner: Callable[[ChainConfig, RunManifest, str], StageOutcome],
    config: ChainConfig,
    manifest: RunManifest,
    objective: str,
    stage: str,
) -> StageOutcome:
    """Run one stage so a failure is recorded and the night carries on without it."""
    try:
        return runner(config, manifest, objective)
    except Exception as error:  # noqa: BLE001 - stages must fail independently
        record = next(
            (
                candidate
                for candidate in reversed(manifest.data["stages"])
                if candidate.get("objective") == objective
                and candidate.get("stage") == stage
                and candidate["status"] == "running"
            ),
            None,
        )
        if record is None:
            record = manifest.start_stage(objective, stage)
        manifest.record_failure(record, error)
        return StageOutcome(status="failed", notes=[f"{type(error).__name__}: {error}"])


def _describe_assertion(assertion: dict) -> str:
    """One line on the assertion, for the summary."""
    if not assertion.get("wrapped"):
        return "not wrapped in caffeinate"
    return f"wrapped in caffeinate {assertion.get('flags', '-ism')}"


def summarise(manifest: dict) -> str:
    """One screen the captain can read in the morning."""
    produced = manifest.get("variants_produced") or []
    lines = [
        f"outcome: {manifest.get('outcome')}   device: {manifest.get('device')}   "
        f"dry_run: {manifest.get('dry_run')}",
        f"started {manifest.get('started')}  finished {manifest.get('finished')}",
        f"sleep assertion: {_describe_assertion(manifest.get('sleep_assertion') or {})}",
        f"objectives: {', '.join(manifest.get('objectives') or [])}   "
        f"variants: {len(produced)} of {manifest.get('variants_expected', '?')}",
        "",
        f"{'objective-stage':<24} {'status':<16} {'epochs':>7} {'tokens/s':>10}  notes",
    ]
    for stage in manifest.get("stages", []):
        epochs = stage.get("epochs_completed")
        configured = stage.get("epochs_configured")
        epoch_text = f"{epochs}/{configured}" if epochs is not None else "-"
        rate = stage.get("tokens_per_second")
        rate_text = f"{rate:,.0f}" if isinstance(rate, int | float) else "-"
        notes = "; ".join(stage.get("notes", []))
        if stage.get("gate"):
            tasks = stage["gate"].get("comparison", {}) or {}
            scored = tasks.get("tasks", []) if isinstance(tasks, dict) else []
            notes = (
                "; ".join(
                    f"{task['name']} ours {task['ours']:.3f} gensim {task['gensim']:.3f}"
                    for task in scored
                )
                or notes
            )
            if stage["gate"].get("reasons"):
                notes = " | ".join(stage["gate"]["reasons"])
        if stage.get("failure"):
            notes = f"{stage['failure']['type']}: {stage['failure']['message']}"
        lines.append(
            f"{stage['name']:<24} {stage['status']:<16} {epoch_text:>7} {rate_text:>10}  {notes}"
        )
    outcomes = manifest.get("objective_outcomes") or {}
    if outcomes:
        lines.append("")
        for objective, outcome in outcomes.items():
            produced_here = ", ".join(outcome["variants_produced"]) or "none"
            lines.append(f"{objective:<10} {outcome['outcome']:<16} {produced_here}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="walk all four stages on synthetic corpora in seconds",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="carry on from the newest checkpoint of each stage",
    )
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts/embeddings"))
    parser.add_argument("--corpus-directory", type=Path, default=Path("data/corpora"))
    parser.add_argument("--hn-corpus", type=Path, default=corpora.HN_CORPUS_PATH)
    parser.add_argument("--analogy-path", type=Path, default=None)
    parser.add_argument("--wordsim-path", type=Path, default=None)
    parser.add_argument(
        "--objectives",
        nargs="+",
        choices=(OBJECTIVE_SKIPGRAM, OBJECTIVE_CBOW),
        default=list(BOTH_OBJECTIVES),
        help="which objectives to run, in order; both by default, producing six variants",
    )
    parser.add_argument("--dimension", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--detach",
        action="store_true",
        help="relaunch in the background, logging to a file, and return immediately",
    )
    parser.add_argument(
        "--no-caffeinate",
        action="store_true",
        help="do not hold a sleep assertion (for CI, where there is nothing to keep awake)",
    )
    args = parser.parse_args(argv)

    if args.detach:
        return _detach(argv, args.output_directory, caffeinate=not args.no_caffeinate)
    if not args.no_caffeinate and os.environ.get(_CAFFEINATE_SENTINEL) is None:
        _reexec_under_caffeinate(argv)

    config = ChainConfig(
        output_directory=args.output_directory,
        corpus_directory=args.corpus_directory,
        hn_corpus_path=args.hn_corpus,
        analogy_path=args.analogy_path,
        wordsim_path=args.wordsim_path,
        objectives=tuple(dict.fromkeys(args.objectives)),
        dimension=args.dimension,
        epochs=args.epochs,
        dry_run=args.dry_run,
        resume=args.resume,
    )
    manifest = run_chain(config)
    print(summarise(manifest))
    print(f"\nmanifest: {config.manifest_path}")
    return 0 if manifest.get("outcome") in {"completed", "partial"} else 1


def _chain_command(argv: list[str] | None, drop: tuple[str, ...] = ()) -> list[str]:
    """This module's own command line, with ``drop``ped flags removed."""
    forwarded = [
        argument
        for argument in (argv if argv is not None else sys.argv[1:])
        if argument not in drop
    ]
    return [sys.executable, "-m", "hn_upvotes.embeddings.chain", *forwarded]


def caffeinate_command(command: list[str]) -> list[str]:
    """Wrap ``command`` in ``caffeinate -ism`` so the assertion lasts exactly as long as it.

    ``-i`` idle, ``-s`` system, ``-m`` disk. **No ``-d``**, so the display is not held. Tying
    it to the command means it is dropped the moment the command exits, rather than a fixed
    timeout that can expire early or outlive the work.

    Returns ``command`` unchanged off macOS or when ``caffeinate`` is missing, so this is a
    no-op in CI rather than a failure.
    """
    if sys.platform != "darwin" or shutil.which("caffeinate") is None:
        return command
    return ["caffeinate", "-ism", *command]


def _reexec_under_caffeinate(argv: list[str] | None) -> None:
    """Replace this process with itself under ``caffeinate``. Does not return if it works.

    A sentinel in the environment stops the replacement happening twice.
    """
    command = caffeinate_command(_chain_command(argv))
    if command[0] != "caffeinate":
        return
    os.environ[_CAFFEINATE_SENTINEL] = "1"
    os.execvp(command[0], command)


def _detach(argv: list[str] | None, output_directory: Path, caffeinate: bool = True) -> int:
    """Relaunch this module in a new session, logging to a file, and return.

    ``start_new_session`` puts the child in its own process group and session, so closing the
    terminal sends its hangup to the shell and not to the run. An overnight run that dies
    with its shell is not an overnight run.

    The child is wrapped in ``caffeinate`` here rather than re-execing, so the sleep assertion
    belongs to the detached run and lasts exactly as long as it does.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    log_path = output_directory / "chain.log"
    command = _chain_command(argv, drop=("--detach",))
    if caffeinate:
        command = caffeinate_command(command)
    with log_path.open("ab") as log:
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1", _CAFFEINATE_SENTINEL: "1"},
        )
    print(f"chain detached as pid {child.pid}, logging to {log_path}")
    print(f"sleep assertion: {'caffeinate -ism, held for the run' if caffeinate else 'none'}")
    print(f"manifest will be at {output_directory / 'run-manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
