"""The overnight training chain: four stages, one process, nobody watching.

Run order, and it is an order rather than a list:

0. **gate** Train on text8, train gensim on the same corpus with the same settings, compare.
   **A failure here aborts the whole chain.** That is the point of it: a bug otherwise costs
   a night of Wikipedia training and shows up in the morning, where this costs ten minutes
   and leaves the machine idle instead.
1. **wiki-only** The Wikipedia subset. Its size is computed from measured throughput and the
   stage's wall-clock ceiling, not picked.
2. **fine-tuned** Stage 1's vectors, carried on over Hacker News titles and bodies at a tenth
   of the learning rate. Words Hacker News has and Wikipedia does not start random.
3. **hn-only** Random initialisation, Hacker News text only. Last on purpose: it needs
   nothing from stages 1 or 2, so a Wikipedia failure does not cost us this variant.

Stages 1 to 3 produce the three embedding variants the whole project compares.

Everything unattended operation needs is here: a per-epoch checkpoint, ``--resume`` from the
newest one, a wall-clock budget per stage that saves and yields rather than eating the next
stage's time, per-stage failure isolation, and one JSON manifest written as it goes so the
morning's question is answered by one file rather than a log.

``--dry-run`` walks all four stages on synthetic corpora in a few seconds, which is how the
chain is verified without spending a night on it.

Needs the ``train`` extra, and the ``data`` extra for the Hacker News stages.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from hn_upvotes.data.preprocess import build_vocabulary
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

#: Measured on this machine with ``make throughput``: skip-gram, dimension 300, batch 1024,
#: CPU with sparse gradients, over a 2M-token Zipfian synthetic corpus. In-vocabulary corpus
#: tokens per second, keyed by the number of negative samples.
#:
#: k=5 is 1.80x faster than k=15, not the 2.7x a count of dot products predicts (6 against
#: 16). The gap is fixed per-batch overhead that does not scale with k.
MEASURED_TOKENS_PER_SECOND: dict[int, int] = {5: 138_362, 15: 77_019}

#: Throughput haircut for running at the full 100,000-word vocabulary rather than the 40,576
#: the calibration corpus produced. Measured at 4%: 109,390 pairs/s at 40,576 words against
#: 105,142 at 100,000.
LARGE_VOCABULARY_FACTOR = 0.96

#: Below this, the Wikipedia variant is barely larger than text8's 100 MB and the stage has
#: lost its point. The chain stops and reports rather than quietly shipping a small corpus.
MINIMUM_WIKIPEDIA_BYTES = 300_000_000

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
    """Where the chain writes, how long each stage may take, and which corpora it reads."""

    output_directory: Path = Path("artifacts/embeddings")
    corpus_directory: Path = Path("data/corpora")
    hn_corpus_path: Path = corpora.HN_CORPUS_PATH
    analogy_path: Path | None = None
    wordsim_path: Path | None = None

    objective: Literal["cbow", "skipgram"] = "skipgram"
    dimension: int = 300
    epochs: int = 5

    #: Wall-clock ceilings. The Wikipedia one is the captain's two hours and is a ceiling
    #: rather than a target: the subset size is computed to fit inside it.
    gate_budget_seconds: float = 30 * 60
    wikipedia_budget_seconds: float = 2 * 60 * 60
    finetune_budget_seconds: float = 90 * 60
    hn_budget_seconds: float = 90 * 60

    dry_run: bool = False
    resume: bool = False
    seed: int = 0

    @property
    def checkpoint_directory(self) -> Path:
        return self.output_directory / "checkpoints"

    @property
    def manifest_path(self) -> Path:
        return self.output_directory / "run-manifest.json"

    def artefact_path(self, stage: str) -> Path:
        return self.output_directory / f"{stage}.npz"


def wikipedia_token_budget(
    budget_seconds: float,
    epochs: int,
    negative_samples: int = 5,
    tokens_per_second: float | None = None,
) -> tuple[int, float]:
    """How many Wikipedia tokens fit in ``budget_seconds``, and roughly how many bytes.

    The stage is sized from measurement rather than picked and timed. The arithmetic, with
    the numbers this machine measured:

    ``138,362 tokens/s x 0.96 x 7,200 s / 5 epochs = 191,271,628 tokens``

    and at 5.9 bytes per token that is about 1.13 GB. Every epoch reads the whole subset, so
    dividing by ``epochs`` is what makes the ceiling a ceiling.
    """
    rate = tokens_per_second or MEASURED_TOKENS_PER_SECOND.get(
        negative_samples, MEASURED_TOKENS_PER_SECOND[5]
    )
    tokens = int(rate * LARGE_VOCABULARY_FACTOR * budget_seconds / max(epochs, 1))
    return tokens, tokens * corpora.BYTES_PER_TOKEN


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
        effective = _stage_config(config, STAGE_HN)
        self.data: dict = {
            "started": _now(),
            "finished": None,
            "dry_run": config.dry_run,
            "resume": config.resume,
            "device": device,
            "objective": effective.objective,
            "dimension": effective.dimension,
            "epochs": effective.epochs,
            "batch_size": effective.batch_size,
            "subsample_threshold": effective.subsample_threshold,
            "outcome": "running",
            "sleep_assertion": sleep_assertion_state(),
            "stages": [],
        }
        self.write()

    def start_stage(self, name: str, detail: dict | None = None) -> dict:
        record: dict = {
            "name": name,
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

        A stage cut short by its budget still counts. Its matrix had less training than it
        was configured for, which is recorded, but it is real and fine-tuning from it beats
        fine-tuning from noise.
        """
        return self.embeddings is not None and self.status in _USABLE_STATUSES


def _stage_config(config: ChainConfig, stage: str) -> SGNSConfig:
    """The training settings for one stage.

    Two deliberate differences from the defaults. ``k`` drops from 15 to 5 for Wikipedia,
    which is what the paper recommends for a large corpus. The fine-tuning stage drops the
    learning rate to a tenth so it adjusts Wikipedia's structure rather than overwriting it.
    """
    base = replace(
        SGNSConfig(),
        objective=config.objective,
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
    stage: str,
    reopen_corpus: Callable[[], Iterator[list[str]]],
    stage_config: SGNSConfig,
    chain_config: ChainConfig,
    manifest: RunManifest,
    record: dict,
    budget_seconds: float,
    initial: TrainedEmbeddings | None = None,
) -> StageOutcome:
    """Build the vocabulary, train under a deadline, checkpoint, and save the artefact."""
    vocabulary = build_vocabulary(
        reopen_corpus(),
        min_count=stage_config.min_count,
        max_size=stage_config.vocabulary_cap,
    )
    record["detail"]["vocabulary"] = len(vocabulary)
    record["detail"]["corpus_tokens"] = int(vocabulary.counts.sum())
    manifest.write()

    resume = (
        newest_checkpoint(chain_config.checkpoint_directory, stage) if chain_config.resume else None
    )
    if resume is not None:
        record["detail"]["resumed_from_epoch"] = resume.epochs_done

    trained = train_embeddings_from_lines(
        reopen_corpus,
        vocabulary,
        stage_config,
        output_path=chain_config.artefact_path(stage),
        initial=initial,
        checkpoint_directory=chain_config.checkpoint_directory,
        stage=stage,
        resume=resume,
        deadline=time.monotonic() + budget_seconds,
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
        artefact=str(chain_config.artefact_path(stage)),
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
        lines, topic_of = corpora.topic_corpus(
            topics=20, words_per_topic=20, lines_per_topic=200, line_length=10
        )
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
            "chance_level": round(1 / 20, 3),
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


def _run_gate(config: ChainConfig, manifest: RunManifest) -> StageOutcome:
    """Stage 0. Train, compare against gensim, and decide whether the night goes ahead."""
    reopen, tasks, detail = _gate_corpus_and_task(config)
    record = manifest.start_stage(STAGE_GATE, detail)
    stage_config = _stage_config(config, STAGE_GATE)

    outcome = _train_stage(
        STAGE_GATE,
        reopen,
        stage_config,
        config,
        manifest,
        record,
        config.gate_budget_seconds,
    )
    if outcome.embeddings is None:
        return StageOutcome(status="failed", notes=["gate produced no vectors"])

    gate = gate_against_gensim(outcome.embeddings, _Reiterable(reopen), tasks=tasks)
    manifest.finish_stage(
        record,
        "passed" if gate.passed else "aborted",
        gate=gate.as_dict(),
        artefact=str(config.artefact_path(STAGE_GATE)),
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


def _run_wikipedia(config: ChainConfig, manifest: RunManifest) -> StageOutcome:
    """Stage 1. Size the subset from measured throughput, then train on that much."""
    stage_config = _stage_config(config, STAGE_WIKI)
    tokens, approximate_bytes = wikipedia_token_budget(
        config.wikipedia_budget_seconds,
        stage_config.epochs,
        stage_config.negative_samples,
    )
    detail = {
        "budget_seconds": config.wikipedia_budget_seconds,
        "tokens_per_second_assumed": MEASURED_TOKENS_PER_SECOND[stage_config.negative_samples],
        "large_vocabulary_factor": LARGE_VOCABULARY_FACTOR,
        "token_budget": tokens,
        "approximate_bytes": int(approximate_bytes),
    }
    record = manifest.start_stage(STAGE_WIKI, detail)

    if not config.dry_run and approximate_bytes < MINIMUM_WIKIPEDIA_BYTES:
        note = (
            f"two hours buys {tokens:,} tokens, about {approximate_bytes / 1e6:.0f} MB, "
            f"below the {MINIMUM_WIKIPEDIA_BYTES / 1e6:.0f} MB floor. Stopping for a "
            f"decision rather than shrinking the corpus quietly. Levers: fewer epochs, "
            f"a smaller dimension, or accepting a smaller subset."
        )
        manifest.finish_stage(record, "needs_decision", notes=[note])
        return StageOutcome(status="needs_decision", notes=[note])

    if config.dry_run:
        lines, _ = corpora.topic_corpus(
            topics=20, words_per_topic=20, lines_per_topic=200, line_length=10, seed=1
        )

        def reopen() -> Iterator[list[str]]:
            return iter(lines)
    else:
        subset_path = config.corpus_directory / "wikipedia-subset.txt"
        if not subset_path.exists():
            corpora.prepare_wikipedia_subset(subset_path, tokens)

        def reopen() -> Iterator[list[str]]:
            return corpora.stream_plain_text(subset_path, token_budget=tokens)

    return _train_stage(
        STAGE_WIKI,
        reopen,
        stage_config,
        config,
        manifest,
        record,
        config.wikipedia_budget_seconds,
    )


def _hn_corpus(config: ChainConfig, seed: int) -> Callable[[], Iterator[list[str]]]:
    """Hacker News titles and bodies, or a synthetic stand-in for the dry run."""
    if config.dry_run:
        lines, _ = corpora.topic_corpus(
            topics=20, words_per_topic=20, lines_per_topic=200, line_length=10, seed=seed
        )
        return lambda: iter(lines)
    return lambda: corpora.stream_hn_text(config.hn_corpus_path, include_bodies=True)


def _run_finetune(
    config: ChainConfig, manifest: RunManifest, wikipedia: StageOutcome
) -> StageOutcome:
    """Stage 2. Carry stage 1's vectors on over Hacker News at a tenth of the rate."""
    stage_config = _stage_config(config, STAGE_FINETUNE)
    record = manifest.start_stage(
        STAGE_FINETUNE,
        {
            "initialised_from": STAGE_WIKI,
            "learning_rate": stage_config.learning_rate,
            "rate_fraction_of_scratch": FINE_TUNE_RATE_FRACTION,
        },
    )
    if not wikipedia.usable:
        note = (
            f"skipped: stage {STAGE_WIKI} finished as {wikipedia.status!r}, so there are no "
            f"Wikipedia vectors to fine-tune from. Stage {STAGE_HN} is unaffected."
        )
        manifest.finish_stage(record, "skipped", notes=[note])
        return StageOutcome(status="skipped", notes=[note])

    return _train_stage(
        STAGE_FINETUNE,
        _hn_corpus(config, seed=2),
        stage_config,
        config,
        manifest,
        record,
        config.finetune_budget_seconds,
        initial=wikipedia.embeddings,
    )


def _run_hn_only(config: ChainConfig, manifest: RunManifest) -> StageOutcome:
    """Stage 3. Random initialisation, Hacker News only. Depends on nothing before it."""
    stage_config = _stage_config(config, STAGE_HN)
    record = manifest.start_stage(STAGE_HN, {"initialised_from": "random"})
    return _train_stage(
        STAGE_HN,
        _hn_corpus(config, seed=2),
        stage_config,
        config,
        manifest,
        record,
        config.hn_budget_seconds,
    )


def run_chain(config: ChainConfig) -> dict:
    """Run all four stages and return the manifest.

    Stages 1 to 3 are each wrapped, so one blowing up is recorded with its traceback and the
    next one still runs. The gate is the exception: it is a gate, and a failed gate stops the
    night on purpose.
    """
    config.output_directory.mkdir(parents=True, exist_ok=True)
    device = select_device().type
    manifest = RunManifest(config.manifest_path, config, device)

    try:
        gate = _run_gate(config, manifest)
    except Exception as error:  # noqa: BLE001 - a gate that crashed has not passed
        record = manifest.start_stage(STAGE_GATE, {"note": "crashed before reporting"})
        manifest.record_failure(record, error)
        gate = StageOutcome(status="failed")

    if gate.status != "passed":
        reasons = gate.gate.reasons if gate.gate else gate.notes
        for stage in (STAGE_WIKI, STAGE_FINETUNE, STAGE_HN):
            record = manifest.start_stage(stage)
            manifest.finish_stage(
                record,
                "skipped",
                notes=[f"skipped: the {STAGE_GATE} stage did not pass"],
            )
        manifest.finish("aborted_at_gate")
        manifest.data["gate_reasons"] = list(reasons)
        manifest.write()
        return manifest.data

    wikipedia = _run_stage_guarded(_run_wikipedia, config, manifest, STAGE_WIKI)
    _run_stage_guarded(
        lambda c, m: _run_finetune(c, m, wikipedia), config, manifest, STAGE_FINETUNE
    )
    _run_stage_guarded(_run_hn_only, config, manifest, STAGE_HN)

    statuses = {stage["name"]: stage["status"] for stage in manifest.data["stages"]}
    variants = [STAGE_WIKI, STAGE_FINETUNE, STAGE_HN]
    produced = [name for name in variants if statuses.get(name) in _USABLE_STATUSES]
    cut_short = [name for name in variants if statuses.get(name) == "cut_short"]
    # A stage that ran out of clock produced a usable matrix but not the run it was asked
    # for, so the night is partial. Reporting it as completed would be the exact thing the
    # cut_short flag exists to prevent.
    complete = len(produced) == len(variants) and not cut_short
    manifest.finish("completed" if complete else "partial")
    manifest.data["variants_produced"] = produced
    manifest.data["variants_cut_short"] = cut_short
    manifest.write()
    return manifest.data


def _run_stage_guarded(
    runner: Callable[[ChainConfig, RunManifest], StageOutcome],
    config: ChainConfig,
    manifest: RunManifest,
    stage: str,
) -> StageOutcome:
    """Run one stage so a failure is recorded and the night carries on without it."""
    try:
        return runner(config, manifest)
    except Exception as error:  # noqa: BLE001 - stages must fail independently
        record = next(
            (
                candidate
                for candidate in reversed(manifest.data["stages"])
                if candidate["name"] == stage and candidate["status"] == "running"
            ),
            None,
        )
        if record is None:
            record = manifest.start_stage(stage)
        manifest.record_failure(record, error)
        return StageOutcome(status="failed", notes=[f"{type(error).__name__}: {error}"])


def _describe_assertion(assertion: dict) -> str:
    """One line on the assertion, for the summary."""
    if not assertion.get("wrapped"):
        return "not wrapped in caffeinate"
    return f"wrapped in caffeinate {assertion.get('flags', '-ism')}"


def summarise(manifest: dict) -> str:
    """One screen the captain can read in the morning."""
    lines = [
        f"outcome: {manifest.get('outcome')}   device: {manifest.get('device')}   "
        f"dry_run: {manifest.get('dry_run')}",
        f"started {manifest.get('started')}  finished {manifest.get('finished')}",
        f"sleep assertion: {_describe_assertion(manifest.get('sleep_assertion') or {})}",
        "",
        f"{'stage':<12} {'status':<16} {'epochs':>7} {'tokens/s':>10}  notes",
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
            f"{stage['name']:<12} {stage['status']:<16} {epoch_text:>7} {rate_text:>10}  {notes}"
        )
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
    parser.add_argument("--objective", choices=("cbow", "skipgram"), default="skipgram")
    parser.add_argument("--dimension", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--wikipedia-hours",
        type=float,
        default=2.0,
        help="ceiling for the Wikipedia stage; the subset size is computed to fit it",
    )
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
        objective=args.objective,
        dimension=args.dimension,
        epochs=args.epochs,
        wikipedia_budget_seconds=args.wikipedia_hours * 3600,
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
