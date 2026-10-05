"""Check source mixing, gate enforcement, and preservation during automatic cleanup."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gensim")
pytest.importorskip("duckdb")
from hn_upvotes.data.preprocess import build_vocabulary  # noqa: E402
from hn_upvotes.embeddings import full_run  # noqa: E402
from hn_upvotes.embeddings.train import SGNSConfig, TrainedEmbeddings  # noqa: E402


def saved_models(directory):
    vocabulary = build_vocabulary([["alpha", "beta"]], min_count=1)
    paths = {}
    for objective in full_run.chain.BOTH_OBJECTIVES:
        path = directory / f"{objective}.npz"
        TrainedEmbeddings(
            matrix=np.arange(len(vocabulary) * 2, dtype=np.float32).reshape(-1, 2),
            vocabulary=vocabulary,
            config=replace(SGNSConfig(), objective=objective, dimension=2),
            tokens_per_second=1,
            epoch_losses=(5, 4, 3, 2, 1),
            epochs_completed=5,
        ).save(path)
        paths[objective] = path
    return paths


def test_joint_reads_each_document_once_on_every_epoch(tmp_path):
    wiki, hn = tmp_path / "wiki", tmp_path / "hn"
    wiki.write_text("alpha\nbeta\ngamma\n")
    hn.write_text("delta epsilon\nzeta\n")
    expected = [["alpha"], ["delta", "epsilon"], ["beta"], ["zeta"], ["gamma"]]
    for _ in range(2):
        assert list(full_run.joint_corpus(wiki, hn, 3, 2)) == expected


def owned_corpus(tmp_path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / ".run-owned").write_text(str(directory.resolve()))
    for filename in full_run.OWNED_FILES:
        (directory / filename).write_text("training data")
    (directory / "unrelated.txt").write_text("keep")
    return directory


def test_cleanup_requires_both_complete_models_and_preserves_other_files(tmp_path):
    directory = owned_corpus(tmp_path)
    models = saved_models(tmp_path)
    with pytest.raises(ValueError, match="both final models"):
        full_run.cleanup_corpus(directory, {"cbow": models["cbow"]}, 5)
    incomplete = TrainedEmbeddings.load(models["cbow"])
    replace(incomplete, cut_short=True).save(models["cbow"])
    with pytest.raises(ValueError, match="incomplete"):
        full_run.cleanup_corpus(directory, models, 5)
    assert all((directory / name).exists() for name in full_run.OWNED_FILES)
    models = saved_models(tmp_path)
    removed = full_run.cleanup_corpus(directory, models, 5)
    assert len(removed) == len(full_run.OWNED_FILES)
    assert (directory / "unrelated.txt").read_text() == "keep"
    assert all(path.exists() for path in models.values())


def test_cleanup_refuses_symlinks_before_deleting_any_file(tmp_path):
    directory = owned_corpus(tmp_path)
    original = tmp_path / "original"
    original.write_text("preserve")
    (directory / "hn-training.txt").unlink()
    (directory / "hn-training.txt").symlink_to(original)
    with pytest.raises(ValueError, match="redirected corpus file"):
        full_run.cleanup_corpus(directory, saved_models(tmp_path), 5)
    assert original.read_text() == "preserve"
    assert (directory / "text8").exists()


def test_failed_gate_stops_before_full_data_acquisition(tmp_path, monkeypatch):
    monkeypatch.setattr(
        full_run, "bundled_evaluation_sets", lambda: {"analogy": tmp_path, "wordsim": tmp_path}
    )
    monkeypatch.setattr(
        full_run.chain, "_run_gate", lambda *args: SimpleNamespace(status="failed", gate=None)
    )

    def forbidden(*args):
        pytest.fail("full corpus acquired after a failed gate")

    monkeypatch.setattr(full_run.corpora, "prepare_wikipedia_corpus", forbidden)
    assert full_run.run(tmp_path / "run", tmp_path / "hn.parquet") == 1
    import json

    assert json.loads((tmp_path / "run/status.json").read_text())["status"] == "aborted_at_gate"


def test_completed_run_removes_only_owned_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        full_run, "bundled_evaluation_sets", lambda: {"analogy": tmp_path, "wordsim": tmp_path}
    )
    monkeypatch.setattr(
        full_run.chain, "_run_gate", lambda *args: SimpleNamespace(status="passed", gate=None)
    )

    def export(path):
        path.write_text(("alpha beta gamma\n") * 10)
        return path

    monkeypatch.setattr(full_run.corpora, "prepare_wikipedia_corpus", export)
    monkeypatch.setattr(
        full_run.corpora, "stream_hn_text", lambda *args, **kwargs: iter([["delta"] * 10])
    )

    def train(reopen, vocabulary, config, output_path, **kwargs):
        assert len(list(reopen())) == 11
        assert kwargs.get("deadline") is None
        model = TrainedEmbeddings(
            np.arange(len(vocabulary) * config.dimension, dtype=np.float32).reshape(
                -1, config.dimension
            ),
            vocabulary,
            config,
            1,
            (5, 4, 3, 2, 1),
            5,
        )
        model.save(output_path)
        return model

    monkeypatch.setattr(full_run, "train_embeddings_from_lines", train)
    monkeypatch.setattr(full_run, "analogy_accuracy", lambda *args: (0.2, 100))
    monkeypatch.setattr(full_run, "wordsim_spearman", lambda *args: (0.5, 100))
    source = tmp_path / "source.parquet"
    source.write_text("original")
    assert full_run.run(tmp_path / "run", source) == 0
    assert source.read_text() == "original"
    assert not (tmp_path / "run/corpus/hn-training.txt").exists()
    assert not (tmp_path / "run/corpus/wikipedia-20231101-en.txt").exists()
    assert (tmp_path / "run/models/cbow-joint.npz").exists()
    assert (tmp_path / "run/models/skipgram-joint.npz").exists()
