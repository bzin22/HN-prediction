"""Train both objectives on full Wikipedia plus earlier HN text, then remove run-owned text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import traceback
from collections.abc import Iterator
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hn_upvotes.data.preprocess import Vocabulary, build_vocabulary
from hn_upvotes.data.splits import DEFAULT_VALIDATION_START
from hn_upvotes.embeddings import chain, corpora
from hn_upvotes.embeddings.evaluate import analogy_accuracy, wordsim_spearman
from hn_upvotes.embeddings.learning_rate_sweep import bundled_evaluation_sets
from hn_upvotes.embeddings.train import SGNSConfig, TrainedEmbeddings, train_embeddings_from_lines

OWNED_FILES = ("text8", "text8.zip", "wikipedia-20231101-en.txt", "hn-training.txt")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class RunStatus:
    """Write status and a heartbeat while training runs outside the terminal."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data = {"started": now(), "pid": os.getpid(), "status": "starting"}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.update()
        self.thread.start()

    def update(self, **values):
        with self.lock:
            self.data.update(values, updated_at=now())
            temporary = self.path.with_suffix(".partial.json")
            temporary.write_text(json.dumps(self.data, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.path)

    def _heartbeat(self):
        while not self.stop.wait(30):
            self.update()

    def close(self):
        self.stop.set()
        self.thread.join()


def file_profile(path: Path) -> dict:
    """Count nonempty documents and record the exact file that supplied training text."""
    digest = hashlib.sha256()
    lines = 0
    size = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            size += len(line)
            lines += bool(line.strip())
    return {"path": str(path), "bytes": size, "lines": lines, "sha256": digest.hexdigest()}


def prepare_hn(source: Path, destination: Path, before: str) -> dict:
    """Write tokenised HN documents. Keep the source Parquet file intact."""
    temporary = destination.with_suffix(".partial")
    tokens = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for line in corpora.stream_hn_text(source, before=before):
            handle.write(" ".join(line) + "\n")
            tokens += len(line)
    temporary.replace(destination)
    return {**file_profile(destination), "tokens": tokens, "before": before, "source": str(source)}


def tokenised_lines(path: Path) -> Iterator[list[str]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if tokens := line.split():
                yield tokens


def joint_corpus(wikipedia: Path, hn: Path, wiki_lines: int, hn_lines: int):
    """Interleave complete documents by source progress, without repetition or truncation.

    Source proportions come from the full inputs. Each source reaches its end at roughly
    the same point in the epoch. Word windows never cross document boundaries.
    """
    readers = [iter(corpora.stream_plain_text(wikipedia)), iter(tokenised_lines(hn))]
    totals = [max(wiki_lines, 1), max(hn_lines, 1)]
    used = [0, 0]
    active = {0, 1}
    try:
        while active:
            index = min(active, key=lambda i: used[i] / totals[i])
            try:
                tokens = next(readers[index])
            except StopIteration:
                active.remove(index)
                continue
            used[index] += 1
            yield tokens
    finally:
        for reader in readers:
            reader.close()


def validate_model(path: Path, objective: str, epochs: int) -> dict:
    """Reload saved weights and check completion before data deletion."""
    trained = TrainedEmbeddings.load(path)
    if trained.config.objective != objective or trained.cut_short:
        raise ValueError(f"wrong objective or incomplete model: {path}")
    if trained.epochs_completed != epochs or len(trained.epoch_losses) != epochs:
        raise ValueError(f"missing completed epochs: {path}")
    if not np.isfinite(trained.matrix).all() or not np.isfinite(trained.epoch_losses).all():
        raise ValueError(f"non-finite weights or losses: {path}")
    if trained.matrix.shape != (len(trained.vocabulary), trained.config.dimension):
        raise ValueError(f"matrix and vocabulary differ: {path}")
    if trained.matrix.std() == 0 or trained.epoch_losses[-1] >= trained.epoch_losses[0]:
        raise ValueError(f"model has not demonstrated learning: {path}")
    return {"path": str(path), "epochs": epochs, "vocabulary": len(trained.vocabulary)}


def cleanup_corpus(directory: Path, models: dict[str, Path], epochs: int) -> list[str]:
    """Delete only this run's text files, after both final models pass reload checks."""
    if set(models) != set(chain.BOTH_OBJECTIVES):
        raise ValueError("both final models are required before cleanup")
    for objective, path in models.items():
        validate_model(path, objective, epochs)
    if directory.is_symlink() or (directory / ".run-owned").is_symlink():
        raise ValueError("refusing to clean a redirected corpus directory")
    if (directory / ".run-owned").read_text() != str(directory.resolve()):
        raise ValueError("corpus ownership marker does not match")
    paths = [directory / name for name in OWNED_FILES]
    if any(path.is_symlink() for path in paths):
        raise ValueError("refusing to delete a redirected corpus file")
    removed = []
    for path in paths:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def run(directory: Path, hn_source: Path, text8_source: Path | None = None) -> int:
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    corpus_directory = directory / "corpus"
    corpus_directory.mkdir()  # A fresh directory prevents reuse or deletion of unrelated data.
    (corpus_directory / ".run-owned").write_text(str(corpus_directory.resolve()))
    status = RunStatus(directory / "status.json")
    epochs = 5
    before = DEFAULT_VALIDATION_START + "-01"
    try:
        evaluation = bundled_evaluation_sets()
        if set(evaluation) != {"analogy", "wordsim"}:
            raise ValueError("both intrinsic evaluation datasets are required")
        if text8_source is not None:
            shutil.copyfile(text8_source, corpus_directory / "text8")
        config = chain.ChainConfig(
            output_directory=directory / "models",
            corpus_directory=corpus_directory,
            hn_corpus_path=hn_source,
            hn_before=before,
            analogy_path=evaluation["analogy"],
            wordsim_path=evaluation["wordsim"],
            epochs=epochs,
        )
        manifest = chain.RunManifest(config.manifest_path, config, "cpu")
        manifest.data.update(mode="joint", variants_expected=2, hn_before=before)
        manifest.write()
        status.update(
            mode="joint",
            hn_before=before,
            wikipedia_snapshot="20231101.en",
            config=asdict(SGNSConfig(negative_samples=5, device="cpu")),
            cleanup="after both models are saved and verified",
            source_hn=str(hn_source),
        )
        gate_results = {}
        for objective in chain.BOTH_OBJECTIVES:
            status.update(status="validating", active_stage=f"{objective}-gate")
            print(f"{now()} Starting {objective} validation", flush=True)
            result = chain._run_gate(config, manifest, objective)
            gate_results[objective] = {
                "status": result.status,
                "gate": result.gate.as_dict() if result.gate else None,
            }
            status.update(gates=gate_results)
        if any(result["status"] != "passed" for result in gate_results.values()):
            manifest.finish("aborted_at_gate")
            status.update(status="aborted_at_gate", finished=now(), active_stage=None)
            return 1

        status.update(status="preparing", active_stage="wikipedia-export")
        wiki = corpus_directory / "wikipedia-20231101-en.txt"
        corpora.prepare_wikipedia_corpus(wiki)
        wiki_profile = file_profile(wiki)
        status.update(wikipedia=wiki_profile, active_stage="hn-export")
        hn = corpus_directory / "hn-training.txt"
        hn_profile = prepare_hn(hn_source, hn, before)
        status.update(hn=hn_profile, active_stage="joint-vocabulary")

        def reopen():
            return joint_corpus(wiki, hn, wiki_profile["lines"], hn_profile["lines"])

        settings = SGNSConfig(negative_samples=5, device="cpu", epochs=epochs)
        vocabulary = build_vocabulary(reopen(), settings.min_count, settings.vocabulary_cap)
        vocabulary.save(directory / "vocabulary.json")
        # Reload the saved vocabulary as a check that its mapping is durable.
        vocabulary = Vocabulary.load(directory / "vocabulary.json")
        status.update(vocabulary=len(vocabulary), corpus_tokens=int(vocabulary.counts.sum()))
        model_paths = {}
        reports = {}
        for objective in chain.BOTH_OBJECTIVES:
            stage = f"{objective}-joint"
            status.update(status="training", active_stage=stage)
            print(f"{now()} Starting full-corpus {stage}", flush=True)
            path = directory / "models" / f"{stage}.npz"
            record = manifest.start_stage(objective, "joint", {"negative_samples": 5})
            trained = train_embeddings_from_lines(
                reopen,
                vocabulary,
                replace(settings, objective=objective),
                output_path=path,
                checkpoint_directory=directory / "models" / "checkpoints",
                stage=stage,
                progress_every=5_000,
            )
            validate_model(path, objective, epochs)
            status.update(active_stage=f"{stage}-evaluation")
            reports[objective] = {
                "analogy": analogy_accuracy(trained, evaluation["analogy"]),
                "wordsim": wordsim_spearman(trained, evaluation["wordsim"]),
                "epoch_losses": trained.epoch_losses,
            }
            model_paths[objective] = path
            manifest.finish_stage(
                record, "completed", artifact=str(path), scores=reports[objective]
            )
            status.update(
                results=reports, models={key: str(value) for key, value in model_paths.items()}
            )
        (directory / "results.json").write_text(
            json.dumps(reports, indent=2, allow_nan=False) + "\n"
        )
        status.update(status="cleaning", active_stage="delete-run-owned-corpora")
        removed = cleanup_corpus(corpus_directory, model_paths, epochs)
        manifest.finish("completed")
        status.update(status="completed", active_stage=None, finished=now(), deleted_files=removed)
        return 0
    except BaseException as error:
        status.update(
            status="failed",
            finished=now(),
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            cleanup="not performed; keep inputs for diagnosis",
        )
        raise
    finally:
        status.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--hn-source", type=Path, default=corpora.HN_CORPUS_PATH)
    parser.add_argument("--text8-source", type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.run_directory, args.hn_source, args.text8_source))


if __name__ == "__main__":
    main()
