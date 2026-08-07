"""Mid-run state, so an overnight chain that dies at hour seven does not start over.

A checkpoint is **not** a :class:`~hn_upvotes.embeddings.train.TrainedEmbeddings`. The
artefact keeps only the input matrix, because that is the embedding and the output matrix is
scoring scratch. Resuming needs both: restart with a zeroed output matrix and the model has
to relearn every score from nothing, which throws away most of the epoch it was resuming to
save.

One file per epoch per stage, named ``{stage}-epoch{n}.npz``, so the newest is found by
sorting and a stage's history is readable from the directory listing.

Needs the ``train`` extra.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from hn_upvotes.data.preprocess import Vocabulary

#: ``{stage}-epoch{n}.npz``. The epoch is zero padded so a plain sort is a numeric sort.
_CHECKPOINT_NAME = re.compile(r"^(?P<stage>.+)-epoch(?P<epoch>\d+)\.npz$")


@dataclass(frozen=True)
class Checkpoint:
    """Everything needed to carry on training exactly where it stopped.

    ``epochs_done`` counts **completed** epochs. A checkpoint written because the stage ran
    out of wall clock mid-epoch still carries the improved matrices, but reports the epoch
    it was part-way through as not done, so resuming repeats that epoch from its start. The
    partial pass is not lost, it is redone on top of the progress it made.
    """

    stage: str
    epochs_done: int
    input_matrix: np.ndarray
    output_matrix: np.ndarray
    vocabulary: Vocabulary
    config_json: str
    epoch_losses: tuple[float, ...]
    tokens_read: int

    def save(self, directory: Path) -> Path:
        """Write to ``{directory}/{stage}-epoch{n}.npz`` and return the path."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.stage}-epoch{self.epochs_done:03d}.npz"
        # Written to a temporary name and moved into place, so a crash during the write
        # cannot leave a truncated file that resume would then try to load. The temporary
        # name has to end in ``.npz`` itself: ``np.savez`` appends the suffix when it is
        # missing, so a ``.npz.partial`` target is silently written to ``.npz.partial.npz``
        # and the move then fails on a file that is not there.
        temporary = path.with_name(f"{path.stem}.partial.npz")
        np.savez(
            temporary,
            stage=np.asarray(self.stage),
            epochs_done=np.asarray(self.epochs_done),
            input_matrix=self.input_matrix,
            output_matrix=self.output_matrix,
            words=np.asarray(self.vocabulary.index_to_word, dtype=object),
            counts=self.vocabulary.counts,
            unknown_index=np.asarray(self.vocabulary.unknown_index),
            config_json=np.asarray(self.config_json),
            epoch_losses=np.asarray(self.epoch_losses, dtype=np.float64),
            tokens_read=np.asarray(self.tokens_read),
        )
        temporary.replace(path)
        return path

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Read a checkpoint written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=True) as payload:
            words = [str(word) for word in payload["words"]]
            return cls(
                stage=str(payload["stage"]),
                epochs_done=int(payload["epochs_done"]),
                input_matrix=payload["input_matrix"],
                output_matrix=payload["output_matrix"],
                vocabulary=Vocabulary(
                    word_to_index={word: i for i, word in enumerate(words)},
                    index_to_word=words,
                    counts=payload["counts"],
                    unknown_index=int(payload["unknown_index"]),
                ),
                config_json=str(payload["config_json"]),
                epoch_losses=tuple(float(x) for x in payload["epoch_losses"]),
                tokens_read=int(payload["tokens_read"]),
            )

    @property
    def config(self) -> dict:
        """The config the checkpoint was trained under, as a plain dict."""
        return json.loads(self.config_json)


def config_to_json(config: object) -> str:
    """Serialise an :class:`~hn_upvotes.embeddings.train.SGNSConfig` for a checkpoint."""
    return json.dumps(asdict(config))  # type: ignore[call-overload]


def newest_checkpoint(directory: Path, stage: str) -> Checkpoint | None:
    """The furthest-along checkpoint for ``stage``, or ``None`` if there is none.

    Chooses by the epoch number in the filename rather than by modification time, because a
    file copied or restored out of order would otherwise look like the newest.
    """
    path = newest_checkpoint_path(directory, stage)
    return Checkpoint.load(path) if path is not None else None


def newest_checkpoint_path(directory: Path, stage: str) -> Path | None:
    """The path :func:`newest_checkpoint` would load."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    best: tuple[int, Path] | None = None
    for candidate in directory.glob("*.npz"):
        matched = _CHECKPOINT_NAME.match(candidate.name)
        if matched is None or matched.group("stage") != stage:
            continue
        epoch = int(matched.group("epoch"))
        if best is None or epoch > best[0]:
            best = (epoch, candidate)
    return None if best is None else best[1]
