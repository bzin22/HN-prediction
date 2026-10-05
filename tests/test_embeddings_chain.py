"""The guarantees that make the overnight chain safe to leave alone.

Each of these is a promise that only matters when nobody is watching, so each is worth a test
rather than a hope:

* both objectives run, and their artefacts cannot be mistaken for each other,
* the gate aborts an objective when its vectors are meaningfully worse than gensim's, and
  costs that objective only,
* a stage that blows up does not take the stages after it, or the other objective, with it,
* stages train every configured epoch without an implicit deadline,
* checkpoints from older cut-short stages can still be resumed,
* a resume carries on rather than starting over, and does not restart a finished objective,
* both objectives read the same complete Wikipedia corpus.

All of it runs on synthetic corpora, CPU only, no network.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the train extra is not installed")
pytest.importorskip("gensim", reason="the train extra is not installed")

from hn_upvotes.data.preprocess import build_vocabulary  # noqa: E402
from hn_upvotes.embeddings import chain as chain_module  # noqa: E402
from hn_upvotes.embeddings.chain import (  # noqa: E402
    BOTH_OBJECTIVES,
    OBJECTIVE_CBOW,
    OBJECTIVE_SKIPGRAM,
    STAGE_FINETUNE,
    STAGE_GATE,
    STAGE_HN,
    STAGE_WIKI,
    VARIANT_STAGES,
    ChainConfig,
    caffeinate_command,
    estimate_training_time,
    run_chain,
)
from hn_upvotes.embeddings.checkpoint import Checkpoint, newest_checkpoint  # noqa: E402
from hn_upvotes.embeddings.corpora import topic_corpus  # noqa: E402
from hn_upvotes.embeddings.train import SGNSConfig, TrainedEmbeddings  # noqa: E402


def _config(tmp_path, **overrides) -> ChainConfig:
    return ChainConfig(
        output_directory=tmp_path / "out",
        corpus_directory=tmp_path / "corpora",
        dry_run=True,
        **overrides,
    )


def _one_objective(tmp_path, objective=OBJECTIVE_SKIPGRAM, **overrides) -> ChainConfig:
    """A single-objective config, for the tests where the second objective adds nothing."""
    return _config(tmp_path, objectives=(objective,), **overrides)


def _statuses(manifest: dict) -> dict[str, str]:
    """Keyed by the qualified ``{objective}-{stage}`` name the manifest records."""
    return {stage["name"]: stage["status"] for stage in manifest["stages"]}


def test_dry_run_completes_every_stage_of_both_objectives_and_writes_six_variants(
    tmp_path, monkeypatch
):
    """The whole chain, end to end, on synthetic corpora.

    This is the review artefact: it proves the ordering, the gate, the checkpointing and the
    manifest all work for both objectives without spending a night on it.
    """
    real_train = chain_module.train_embeddings_from_lines
    stages_without_deadlines = []

    def train_without_deadline(*args, **kwargs):
        assert kwargs.get("deadline") is None
        stages_without_deadlines.append(kwargs["stage"])
        return real_train(*args, **kwargs)

    monkeypatch.setattr(chain_module, "train_embeddings_from_lines", train_without_deadline)
    config = _config(tmp_path)
    manifest = run_chain(config)

    assert len(stages_without_deadlines) == 8
    assert all(
        stage["epochs_completed"] == stage["epochs_configured"] for stage in manifest["stages"]
    )
    assert manifest["outcome"] == "completed"
    assert manifest["objectives"] == list(BOTH_OBJECTIVES)
    expected = {
        f"{objective}-{stage}": "passed" if stage == STAGE_GATE else "completed"
        for objective in BOTH_OBJECTIVES
        for stage in (STAGE_GATE, *VARIANT_STAGES)
    }
    assert _statuses(manifest) == expected

    # Three variants per objective, six in all, and the manifest says which is which.
    assert manifest["variants_expected"] == 6
    assert manifest["variants_produced"] == [
        f"{objective}-{stage}" for objective in BOTH_OBJECTIVES for stage in VARIANT_STAGES
    ]
    for objective in BOTH_OBJECTIVES:
        assert manifest["objective_outcomes"][objective]["outcome"] == "completed"

    for objective in BOTH_OBJECTIVES:
        for stage in VARIANT_STAGES:
            artefact = config.artefact_path(objective, stage)
            assert artefact.exists(), f"{objective}-{stage} produced no artefact"
            loaded = TrainedEmbeddings.load(artefact)
            assert loaded.matrix.shape[0] == len(loaded.vocabulary)
            assert not loaded.cut_short
            # The artefact carries its own objective, so a file cannot lie about what made it.
            assert loaded.config.objective == objective

    # The loss fell in every stage that trained.
    for stage in manifest["stages"]:
        losses = stage.get("epoch_losses") or []
        if len(losses) > 1:
            assert losses[-1] < losses[0], f"{stage['name']} loss did not fall: {losses}"


def test_every_artefact_and_checkpoint_is_named_for_its_objective(tmp_path):
    """Six variants on disk, six distinct names, and no name shared between the objectives.

    A ``wiki-only.npz`` that could be either objective is a variant nobody can use in the
    comparison the whole phase exists to run. The objective goes in the file name, not only
    in the manifest.
    """
    config = _config(tmp_path)
    run_chain(config)

    paths = {
        (objective, stage): config.artefact_path(objective, stage)
        for objective in BOTH_OBJECTIVES
        for stage in VARIANT_STAGES
    }
    assert len({path.name for path in paths.values()}) == 6, "two variants share a file name"
    for (objective, stage), path in paths.items():
        assert path.name == f"{objective}-{stage}.npz"
        assert path.exists()

    # Checkpoints are qualified the same way, which is what keeps a resume from crossing
    # objectives: newest_checkpoint matches on the qualified stage name.
    names = {path.name for path in config.checkpoint_directory.glob("*.npz")}
    assert f"{OBJECTIVE_SKIPGRAM}-{STAGE_WIKI}-epoch003.npz" in names
    assert f"{OBJECTIVE_CBOW}-{STAGE_WIKI}-epoch003.npz" in names
    assert f"{OBJECTIVE_CBOW}-{STAGE_HN}-epoch001.npz" in names
    for objective in BOTH_OBJECTIVES:
        for stage in VARIANT_STAGES:
            found = newest_checkpoint(config.checkpoint_directory, f"{objective}-{stage}")
            assert found is not None, f"no checkpoint for {objective}-{stage}"


def test_the_gate_runs_once_per_objective_against_its_own_gensim_reference(tmp_path):
    """CBOW and Skip-gram differ by the masked average, so one gate cannot validate both.

    gensim's ``sg`` flag is set from the objective under test, so each is compared against
    the reference for the same objective rather than against Skip-gram's.
    """
    config = _config(tmp_path)
    manifest = run_chain(config)

    gates = [stage for stage in manifest["stages"] if stage["stage"] == STAGE_GATE]
    assert [gate["objective"] for gate in gates] == list(BOTH_OBJECTIVES)
    for gate in gates:
        assert gate["status"] == "passed"
        assert gate["gate"]["comparison"]["tasks"], "the gate compared nothing"

    for objective in BOTH_OBJECTIVES:
        assert chain_module._stage_config(config, STAGE_GATE, objective).objective == objective


def test_a_broken_cbow_does_not_cost_the_skipgram_variants(tmp_path, monkeypatch):
    """Break one objective's vectors and the other objective's night must go ahead.

    This is the isolation the two-objective run rests on. Skip-gram runs first, so its three
    variants are already on disk when CBOW's gate aborts, and nothing rolls them back.
    """
    real_train = chain_module.train_embeddings_from_lines

    def sabotage_cbow(*args, **kwargs):
        trained = real_train(*args, **kwargs)
        if trained.config.objective != OBJECTIVE_CBOW:
            return trained
        noise = np.random.default_rng(0).normal(size=trained.matrix.shape).astype(np.float32)
        return TrainedEmbeddings(
            matrix=noise * 0.01,
            vocabulary=trained.vocabulary,
            config=trained.config,
            tokens_per_second=trained.tokens_per_second,
            epoch_losses=trained.epoch_losses,
            epochs_completed=trained.epochs_completed,
            tokens_trained=trained.tokens_trained,
        )

    monkeypatch.setattr(chain_module, "train_embeddings_from_lines", sabotage_cbow)
    config = _config(tmp_path)
    manifest = run_chain(config)

    statuses = _statuses(manifest)
    assert statuses[f"{OBJECTIVE_CBOW}-{STAGE_GATE}"] == "aborted"
    for stage in VARIANT_STAGES:
        assert statuses[f"{OBJECTIVE_CBOW}-{stage}"] == "skipped"
        assert not config.artefact_path(OBJECTIVE_CBOW, stage).exists()

    # The other objective is untouched: three variants, on disk, unaffected.
    assert statuses[f"{OBJECTIVE_SKIPGRAM}-{STAGE_GATE}"] == "passed"
    for stage in VARIANT_STAGES:
        assert statuses[f"{OBJECTIVE_SKIPGRAM}-{stage}"] == "completed"
        assert config.artefact_path(OBJECTIVE_SKIPGRAM, stage).exists()

    outcomes = manifest["objective_outcomes"]
    assert outcomes[OBJECTIVE_SKIPGRAM]["outcome"] == "completed"
    assert outcomes[OBJECTIVE_CBOW]["outcome"] == "aborted_at_gate"
    assert any("topic purity" in reason for reason in outcomes[OBJECTIVE_CBOW]["gate_reasons"])
    assert manifest["outcome"] == "partial"
    assert manifest["variants_produced"] == [
        f"{OBJECTIVE_SKIPGRAM}-{stage}" for stage in VARIANT_STAGES
    ]


def test_a_broken_skipgram_does_not_cost_the_cbow_variants(tmp_path, monkeypatch):
    """The reverse of the last test, because the objective that runs first is not special.

    Skip-gram aborting must not stop CBOW's gate from running afterwards.
    """
    real_train = chain_module.train_embeddings_from_lines

    def sabotage_skipgram(*args, **kwargs):
        trained = real_train(*args, **kwargs)
        if trained.config.objective != OBJECTIVE_SKIPGRAM:
            return trained
        noise = np.random.default_rng(0).normal(size=trained.matrix.shape).astype(np.float32)
        return TrainedEmbeddings(
            matrix=noise * 0.01,
            vocabulary=trained.vocabulary,
            config=trained.config,
            tokens_per_second=trained.tokens_per_second,
            epoch_losses=trained.epoch_losses,
            epochs_completed=trained.epochs_completed,
            tokens_trained=trained.tokens_trained,
        )

    monkeypatch.setattr(chain_module, "train_embeddings_from_lines", sabotage_skipgram)
    config = _config(tmp_path)
    manifest = run_chain(config)

    assert _statuses(manifest)[f"{OBJECTIVE_SKIPGRAM}-{STAGE_GATE}"] == "aborted"
    assert _statuses(manifest)[f"{OBJECTIVE_CBOW}-{STAGE_GATE}"] == "passed"
    for stage in VARIANT_STAGES:
        assert config.artefact_path(OBJECTIVE_CBOW, stage).exists()
        assert not config.artefact_path(OBJECTIVE_SKIPGRAM, stage).exists()


def test_a_broken_implementation_aborts_that_objective_at_the_gate(tmp_path, monkeypatch):
    """Deliberately break the vectors and the night must not go ahead.

    The break is realistic rather than cosmetic: the matrix is replaced with noise, which is
    what a forward pass wired up wrongly produces. gensim still trains correctly on the same
    corpus, so the gate sees a real gap. Measured: topic purity 0.045 against gensim's 1.000,
    a 95.5% shortfall, and neighbour overlap 0.021 against a 0.15 floor.

    The assertion that matters is the last one. No Wikipedia artefact means the hours were
    never spent.
    """
    real_train = chain_module.train_embeddings_from_lines

    def sabotaged(*args, **kwargs):
        trained = real_train(*args, **kwargs)
        noise = np.random.default_rng(0).normal(size=trained.matrix.shape).astype(np.float32)
        return TrainedEmbeddings(
            matrix=noise * 0.01,
            vocabulary=trained.vocabulary,
            config=trained.config,
            tokens_per_second=trained.tokens_per_second,
            epoch_losses=trained.epoch_losses,
            epochs_completed=trained.epochs_completed,
            tokens_trained=trained.tokens_trained,
        )

    monkeypatch.setattr(chain_module, "train_embeddings_from_lines", sabotaged)
    config = _one_objective(tmp_path)
    manifest = run_chain(config)

    assert manifest["outcome"] == "aborted_at_gate"
    statuses = _statuses(manifest)
    assert statuses[f"{OBJECTIVE_SKIPGRAM}-{STAGE_GATE}"] == "aborted"
    for stage in VARIANT_STAGES:
        assert statuses[f"{OBJECTIVE_SKIPGRAM}-{stage}"] == "skipped"

    reasons = manifest["objective_outcomes"][OBJECTIVE_SKIPGRAM]["gate_reasons"]
    assert reasons, "the gate aborted without saying why"
    assert any("topic purity" in reason for reason in reasons)

    # The whole point: no night was spent on the variants.
    assert not config.artefact_path(OBJECTIVE_SKIPGRAM, STAGE_WIKI).exists()
    assert not config.artefact_path(OBJECTIVE_SKIPGRAM, STAGE_HN).exists()


def test_a_gate_that_cannot_run_counts_as_a_failure_not_a_pass(tmp_path, monkeypatch):
    """Fails closed. A gate that crashed has not passed."""
    monkeypatch.setattr(
        chain_module,
        "gate_against_gensim",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("gensim exploded")),
    )
    manifest = run_chain(_one_objective(tmp_path))
    assert manifest["outcome"] == "aborted_at_gate"
    assert _statuses(manifest)[f"{OBJECTIVE_SKIPGRAM}-{STAGE_WIKI}"] == "skipped"


def test_a_failing_stage_does_not_take_the_next_ones_with_it(tmp_path, monkeypatch):
    """Stage 1 blows up. Stage 2 cannot run without it, but stage 3 must still deliver.

    This is why ``hn-only`` runs last: it depends on nothing, so a Wikipedia failure costs one
    variant instead of the whole night.
    """
    real_train = chain_module.train_embeddings_from_lines

    def fail_on_wikipedia(*args, **kwargs):
        if kwargs.get("stage", "").endswith(STAGE_WIKI):
            raise OSError("the Wikipedia subset could not be read")
        return real_train(*args, **kwargs)

    monkeypatch.setattr(chain_module, "train_embeddings_from_lines", fail_on_wikipedia)
    config = _one_objective(tmp_path)
    manifest = run_chain(config)

    statuses = _statuses(manifest)
    assert statuses[f"{OBJECTIVE_SKIPGRAM}-{STAGE_GATE}"] == "passed"
    assert statuses[f"{OBJECTIVE_SKIPGRAM}-{STAGE_WIKI}"] == "failed"
    # Cannot fine-tune from vectors that do not exist, and it says so rather than crashing.
    assert statuses[f"{OBJECTIVE_SKIPGRAM}-{STAGE_FINETUNE}"] == "skipped"
    # The independent variant survived.
    assert statuses[f"{OBJECTIVE_SKIPGRAM}-{STAGE_HN}"] == "completed"
    assert config.artefact_path(OBJECTIVE_SKIPGRAM, STAGE_HN).exists()
    assert manifest["outcome"] == "partial"

    # The traceback is in the manifest, so the morning does not need the log.
    failure = next(s for s in manifest["stages"] if s["stage"] == STAGE_WIKI)["failure"]
    assert failure["type"] == "OSError"
    assert "Traceback" in failure["traceback"]


def _interrupt_wikipedia(monkeypatch):
    """Inject an explicit diagnostic deadline to simulate an older partial checkpoint."""
    real_train = chain_module.train_embeddings_from_lines

    def interrupted(*args, **kwargs):
        if kwargs["stage"].endswith(STAGE_WIKI):
            kwargs["deadline"] = 0.0
        return real_train(*args, **kwargs)

    monkeypatch.setattr(chain_module, "train_embeddings_from_lines", interrupted)
    return real_train


def test_an_explicitly_interrupted_stage_preserves_its_partial_checkpoint(tmp_path, monkeypatch):
    """The lower-level diagnostic deadline still preserves recoverable work."""
    _interrupt_wikipedia(monkeypatch)
    config = _one_objective(tmp_path)
    manifest = run_chain(config)

    wikipedia = next(s for s in manifest["stages"] if s["stage"] == STAGE_WIKI)
    assert wikipedia["status"] == "cut_short"
    assert wikipedia["epochs_completed"] < wikipedia["epochs_configured"]
    # Not reported as a finished night, which is the thing the flag exists to prevent.
    assert manifest["outcome"] == "partial"
    assert manifest["variants_cut_short"] == [f"{OBJECTIVE_SKIPGRAM}-{STAGE_WIKI}"]

    # The work it did is on disk, both as a checkpoint and as a usable artefact.
    stage_name = config.checkpoint_stage(OBJECTIVE_SKIPGRAM, STAGE_WIKI)
    assert newest_checkpoint(config.checkpoint_directory, stage_name) is not None
    assert TrainedEmbeddings.load(config.artefact_path(OBJECTIVE_SKIPGRAM, STAGE_WIKI)).cut_short

    # And the stage after it still ran, from the partial vectors.
    assert _statuses(manifest)[f"{OBJECTIVE_SKIPGRAM}-{STAGE_FINETUNE}"] == "completed"


def test_resume_carries_on_and_the_epoch_count_does_not_drift(tmp_path, monkeypatch):
    """Cut a stage short, resume, and it must finish exactly its configured epochs.

    The bug this guards is real and was found here: a partial epoch's loss appended to the
    per-epoch list made a resumed stage report 4 epochs completed out of 3.
    """
    real_train = _interrupt_wikipedia(monkeypatch)
    config = _one_objective(tmp_path)
    first = run_chain(config)
    assert _statuses(first)[f"{OBJECTIVE_SKIPGRAM}-{STAGE_WIKI}"] == "cut_short"

    monkeypatch.setattr(chain_module, "train_embeddings_from_lines", real_train)
    resumed = run_chain(_one_objective(tmp_path, resume=True))
    wikipedia = next(s for s in resumed["stages"] if s["stage"] == STAGE_WIKI)
    assert wikipedia["status"] == "completed"
    assert wikipedia["epochs_completed"] == wikipedia["epochs_configured"]
    assert len(wikipedia["epoch_losses"]) == wikipedia["epochs_configured"]

    # The stages that had already finished did not retrain, and say so rather than
    # reporting a completed run they did not do.
    assert _statuses(resumed)[f"{OBJECTIVE_SKIPGRAM}-{STAGE_HN}"] == "already_complete"


def test_resume_does_not_restart_the_objective_that_already_finished(tmp_path):
    """Run both objectives, then resume, and neither objective may train again.

    The failure this guards is a resume that reads Skip-gram's checkpoints for CBOW, or
    reads none at all and quietly repeats a night's work. Every stage of both objectives
    must come back ``already_complete``.
    """
    config = _config(tmp_path)
    first = run_chain(config)
    assert first["outcome"] == "completed"

    resumed = run_chain(_config(tmp_path, resume=True))
    for stage in resumed["stages"]:
        if stage["stage"] == STAGE_GATE:
            assert stage["status"] == "passed"
        else:
            assert stage["status"] == "already_complete", stage["name"]
        assert stage.get("tokens_trained", 0) == 0, f"{stage['name']} retrained"

    # Each objective resumed from its own checkpoints, not from the other's.
    for stage in resumed["stages"]:
        resumed_from = stage["detail"].get("resumed_from_epoch")
        assert resumed_from == resumed["epochs"], stage["name"]


def test_checkpoint_carries_both_matrices_and_the_newest_wins(tmp_path):
    """Resuming from the input matrix alone would restart scoring from zero.

    The output matrix is scoring scratch that is discarded at the *end* of training, not
    between epochs, so a checkpoint has to hold it.
    """
    vocabulary = build_vocabulary(iter([["a", "b", "c"]] * 10), min_count=1, max_size=50)
    for epoch, fill in ((1, 0.1), (3, 0.3), (2, 0.2)):
        Checkpoint(
            stage="demo",
            epochs_done=epoch,
            input_matrix=np.full((len(vocabulary), 4), fill, dtype=np.float32),
            output_matrix=np.full((len(vocabulary), 4), -fill, dtype=np.float32),
            vocabulary=vocabulary,
            config_json="{}",
            epoch_losses=(1.0,) * epoch,
            tokens_read=100 * epoch,
        ).save(tmp_path)

    # Chosen by the epoch in the name, not by modification time: epoch 2 was written last.
    newest = newest_checkpoint(tmp_path, "demo")
    assert newest.epochs_done == 3
    assert newest.input_matrix[0][0] == pytest.approx(0.3)
    assert newest.output_matrix[0][0] == pytest.approx(-0.3)
    assert newest.vocabulary.index_to_word == vocabulary.index_to_word
    assert newest.tokens_read == 300

    # A different stage's checkpoints are not confused with this one's.
    assert newest_checkpoint(tmp_path, "other") is None
    # No half-written temporary file is left behind to be loaded later.
    assert not list(tmp_path.glob("*.partial.npz"))


def test_both_objectives_read_the_complete_corpus_and_ignore_old_subsets(tmp_path, monkeypatch):
    """A real reader must reach the last document and reuse the same file for both objectives."""
    config = ChainConfig(corpus_directory=tmp_path, output_directory=tmp_path / "out")
    (tmp_path / "wikipedia-subset.txt").write_text("obsolete subset\n")
    prepared = []
    observed = []

    def prepare(path):
        prepared.append(path)
        path.write_text("first document\nlast document\n")
        return path

    def capture(objective, stage, reopen, *args, **kwargs):
        observed.append((objective, list(reopen())))
        return chain_module.StageOutcome(status="completed")

    monkeypatch.setattr(chain_module.corpora, "prepare_wikipedia_corpus", prepare)
    monkeypatch.setattr(chain_module, "_train_stage", capture)
    manifest = chain_module.RunManifest(config.manifest_path, config, "cpu")
    for objective in BOTH_OBJECTIVES:
        chain_module._run_wikipedia(config, manifest, objective)

    assert prepared == [tmp_path / "wikipedia-20231101-en.txt"]
    assert observed == [
        (objective, [["first", "document"], ["last", "document"]]) for objective in BOTH_OBJECTIVES
    ]
    assert (tmp_path / "wikipedia-subset.txt").read_text() == "obsolete subset\n"


@pytest.mark.parametrize("old_lines", [[["alpha", "beta"]] * 5, [["gamma", "delta"]] * 10])
def test_resume_rejects_a_different_corpus_even_when_matrix_shapes_match(tmp_path, old_lines):
    """A checkpoint from a limited corpus cannot silently become a full-corpus result."""
    config = _one_objective(tmp_path, resume=True)
    stage_config = SGNSConfig(dimension=4, min_count=1)
    old_vocabulary = build_vocabulary(iter(old_lines), min_count=1, max_size=50)
    checkpoint = Checkpoint(
        stage=config.checkpoint_stage(OBJECTIVE_SKIPGRAM, STAGE_WIKI),
        epochs_done=5,
        input_matrix=np.zeros((len(old_vocabulary), 4), dtype=np.float32),
        output_matrix=np.zeros((len(old_vocabulary), 4), dtype=np.float32),
        vocabulary=old_vocabulary,
        config_json="{}",
        epoch_losses=(1.0,) * 5,
        tokens_read=100,
    ).save(config.checkpoint_directory)
    original = checkpoint.read_bytes()
    manifest = chain_module.RunManifest(config.manifest_path, config, "cpu")
    record = manifest.start_stage(OBJECTIVE_SKIPGRAM, STAGE_WIKI)
    with pytest.raises(ValueError, match="differ from the current corpus"):
        chain_module._train_stage(
            OBJECTIVE_SKIPGRAM,
            STAGE_WIKI,
            lambda: iter([["alpha", "beta"]] * 10),
            stage_config,
            config,
            manifest,
            record,
        )
    assert checkpoint.read_bytes() == original
    assert not config.artefact_path(OBJECTIVE_SKIPGRAM, STAGE_WIKI).exists()


def test_runtime_estimates_grow_with_corpus_size_without_clipping():
    """A large corpus can exceed the former stage and total limits in the estimate."""
    estimate = estimate_training_time(ChainConfig(), wikipedia_tokens=2_000_000_000)
    larger = estimate_training_time(ChainConfig(), wikipedia_tokens=4_000_000_000)
    wiki = lambda result: next(  # noqa: E731
        row["expected_seconds"]
        for row in result["stages"]
        if row["objective"] == OBJECTIVE_SKIPGRAM and row["stage"] == STAGE_WIKI
    )
    assert len(estimate["stages"]) == 8
    assert wiki(estimate) > 7_200
    assert wiki(larger) == pytest.approx(2 * wiki(estimate), abs=0.2)
    assert estimate["expected_hours"] > 11


def test_fine_tuning_uses_a_lower_rate_and_starts_from_its_own_objective(tmp_path):
    """Stage 2 adjusts that objective's own Wikipedia vectors rather than the other's.

    Warm-starting CBOW from Skip-gram's matrix would train one objective on the other's
    output, and the comparison the whole phase exists to run would mean nothing.
    """
    config = _config(tmp_path)
    for objective in BOTH_OBJECTIVES:
        scratch = chain_module._stage_config(config, STAGE_HN, objective)
        finetune = chain_module._stage_config(config, STAGE_FINETUNE, objective)
        assert finetune.learning_rate == pytest.approx(scratch.learning_rate * 0.1)
        assert finetune.min_learning_rate == pytest.approx(scratch.min_learning_rate * 0.1)

    manifest = run_chain(config)
    for objective in BOTH_OBJECTIVES:
        stage = next(
            s
            for s in manifest["stages"]
            if s["stage"] == STAGE_FINETUNE and s["objective"] == objective
        )
        assert stage["detail"]["initialised_from"] == f"{objective}-{STAGE_WIKI}"
        expected = chain_module._stage_config(config, STAGE_FINETUNE, objective)
        assert stage["learning_rate"] == pytest.approx(expected.learning_rate)


def test_wikipedia_stage_uses_five_negatives_and_the_rest_use_fifteen(tmp_path):
    """The paper's recommendation: 5 negatives for a large corpus, 15 for a small one."""
    config = _config(tmp_path)
    for objective in BOTH_OBJECTIVES:
        assert chain_module._stage_config(config, STAGE_WIKI, objective).negative_samples == 5
        assert chain_module._stage_config(config, STAGE_GATE, objective).negative_samples == 15
        assert chain_module._stage_config(config, STAGE_HN, objective).negative_samples == 15


def test_the_two_objectives_differ_by_nothing_but_the_objective(tmp_path):
    """Same seed, same dimension, same epochs, same batch. Otherwise it is not a comparison."""
    config = _config(tmp_path)
    for stage in (STAGE_GATE, *VARIANT_STAGES):
        skipgram = chain_module._stage_config(config, stage, OBJECTIVE_SKIPGRAM)
        cbow = chain_module._stage_config(config, stage, OBJECTIVE_CBOW)
        assert skipgram.objective == OBJECTIVE_SKIPGRAM
        assert cbow.objective == OBJECTIVE_CBOW
        from dataclasses import replace

        assert replace(skipgram, objective="") == replace(cbow, objective="")


def test_the_training_command_is_wrapped_in_caffeinate_without_holding_the_display():
    """``-ism`` ties the assertion to the command. ``-d`` is deliberately absent."""
    wrapped = caffeinate_command(["python", "-m", "hn_upvotes.embeddings.chain"])
    if wrapped[0] != "caffeinate":
        pytest.skip("caffeinate is not available on this platform")
    assert wrapped[:2] == ["caffeinate", "-ism"]
    assert "-d" not in wrapped[1]
    # The command itself is preserved unchanged after the flags.
    assert wrapped[2:] == ["python", "-m", "hn_upvotes.embeddings.chain"]


def test_the_manifest_records_the_settings_actually_used(tmp_path):
    """A manifest claiming 300 dimensions over a run that used 32 is worse than none."""
    config = _config(tmp_path)
    manifest = run_chain(config)
    effective = chain_module._stage_config(config, STAGE_HN, OBJECTIVE_SKIPGRAM)
    assert manifest["dimension"] == effective.dimension
    assert manifest["epochs"] == effective.epochs
    assert manifest["dimension"] != config.dimension, "the dry run should shrink the dimension"
    assert manifest["device"] == "cpu"
    assert manifest["objectives"] == list(BOTH_OBJECTIVES)


def test_the_command_line_runs_both_objectives_by_default():
    """Six variants is the point of the phase, so it is the default rather than a flag."""
    assert ChainConfig().objectives == BOTH_OBJECTIVES
    assert set(BOTH_OBJECTIVES) == {OBJECTIVE_SKIPGRAM, OBJECTIVE_CBOW}


def test_gensim_is_given_the_same_subsampling_threshold_we_use():
    """gensim's ``sample`` default is 1e-3 against the paper's 1e-5.

    Left alone the gate would compare a model that kept 82% of its tokens against one that
    kept 31%, and blame the gap on our implementation.
    """
    from hn_upvotes.embeddings.evaluate import train_gensim

    lines, _ = topic_corpus(topics=4, words_per_topic=6, lines_per_topic=40, line_length=6)
    config = SGNSConfig(
        dimension=8, epochs=1, negative_samples=5, subsample_threshold=1e-5, vocabulary_cap=100
    )
    model = train_gensim(lines, config)
    assert model.sample == pytest.approx(1e-5)
    assert model.negative == 5
    assert model.ns_exponent == pytest.approx(0.75)
    assert model.alpha == pytest.approx(config.learning_rate)
    assert model.min_alpha == pytest.approx(config.min_learning_rate)
    # Multi-threaded Hogwild would make a run irreproducible from the seed.
    assert model.workers == 1


def test_gensim_is_trained_on_the_objective_under_test():
    """``sg=1`` for Skip-gram and ``sg=0`` for CBOW, or the gate compares two different tasks."""
    from hn_upvotes.embeddings.evaluate import train_gensim

    lines, _ = topic_corpus(topics=4, words_per_topic=6, lines_per_topic=40, line_length=6)
    base = SGNSConfig(dimension=8, epochs=1, negative_samples=5, vocabulary_cap=100)
    from dataclasses import replace

    assert train_gensim(lines, replace(base, objective=OBJECTIVE_SKIPGRAM)).sg == 1
    assert train_gensim(lines, replace(base, objective=OBJECTIVE_CBOW)).sg == 0
