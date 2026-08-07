"""The guarantees that make the overnight chain safe to leave alone.

Each of these is a promise that only matters when nobody is watching, so each is worth a test
rather than a hope:

* the gate aborts the whole chain when our vectors are meaningfully worse than gensim's,
* a stage that blows up does not take the stages after it with it,
* a stage that runs out of wall clock saves what it has and says it was cut short,
* a resume carries on rather than starting over, and the epoch count does not drift,
* the Wikipedia subset size is arithmetic on a measured number.

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
    MINIMUM_WIKIPEDIA_BYTES,
    STAGE_FINETUNE,
    STAGE_GATE,
    STAGE_HN,
    STAGE_WIKI,
    ChainConfig,
    caffeinate_command,
    run_chain,
    wikipedia_token_budget,
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


def _statuses(manifest: dict) -> dict[str, str]:
    return {stage["name"]: stage["status"] for stage in manifest["stages"]}


def test_dry_run_completes_all_four_stages_and_writes_three_variants(tmp_path):
    """The whole chain, end to end, on synthetic corpora.

    This is the review artefact: it proves the ordering, the gate, the checkpointing and the
    manifest all work without spending a night on it.
    """
    config = _config(tmp_path)
    manifest = run_chain(config)

    assert manifest["outcome"] == "completed"
    assert _statuses(manifest) == {
        STAGE_GATE: "passed",
        STAGE_WIKI: "completed",
        STAGE_FINETUNE: "completed",
        STAGE_HN: "completed",
    }
    assert manifest["variants_produced"] == [STAGE_WIKI, STAGE_FINETUNE, STAGE_HN]

    # The three variants the project compares, each loadable with its vocabulary attached.
    for stage in (STAGE_WIKI, STAGE_FINETUNE, STAGE_HN):
        artefact = config.artefact_path(stage)
        assert artefact.exists(), f"{stage} produced no artefact"
        loaded = TrainedEmbeddings.load(artefact)
        assert loaded.matrix.shape[0] == len(loaded.vocabulary)
        assert not loaded.cut_short

    # A checkpoint per epoch per stage, named so the stages are distinguishable.
    names = sorted(path.name for path in config.checkpoint_directory.glob("*.npz"))
    assert f"{STAGE_HN}-epoch001.npz" in names
    assert f"{STAGE_WIKI}-epoch003.npz" in names

    # The loss fell in every stage that trained.
    for stage in manifest["stages"]:
        losses = stage.get("epoch_losses") or []
        if len(losses) > 1:
            assert losses[-1] < losses[0], f"{stage['name']} loss did not fall: {losses}"


def test_a_broken_implementation_aborts_the_chain_at_the_gate(tmp_path, monkeypatch):
    """Deliberately break the vectors and the night must not go ahead.

    The break is realistic rather than cosmetic: the matrix is replaced with noise, which is
    what a forward pass wired up wrongly produces. gensim still trains correctly on the same
    corpus, so the gate sees a real gap. Measured: topic purity 0.045 against gensim's 1.000,
    a 95.5% shortfall, and neighbour overlap 0.021 against a 0.15 floor.

    The assertion that matters is the last one. No Wikipedia artefact means the eight hours
    were never spent.
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
    config = _config(tmp_path)
    manifest = run_chain(config)

    assert manifest["outcome"] == "aborted_at_gate"
    assert _statuses(manifest)[STAGE_GATE] == "aborted"
    for stage in (STAGE_WIKI, STAGE_FINETUNE, STAGE_HN):
        assert _statuses(manifest)[stage] == "skipped"

    reasons = manifest["gate_reasons"]
    assert reasons, "the gate aborted without saying why"
    assert any("topic purity" in reason for reason in reasons)

    # The whole point: no night was spent on the variants.
    assert not config.artefact_path(STAGE_WIKI).exists()
    assert not config.artefact_path(STAGE_HN).exists()


def test_a_gate_that_cannot_run_counts_as_a_failure_not_a_pass(tmp_path, monkeypatch):
    """Fails closed. A gate that crashed has not passed."""
    monkeypatch.setattr(
        chain_module,
        "gate_against_gensim",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("gensim exploded")),
    )
    manifest = run_chain(_config(tmp_path))
    assert manifest["outcome"] == "aborted_at_gate"
    assert _statuses(manifest)[STAGE_WIKI] == "skipped"


def test_a_failing_stage_does_not_take_the_next_ones_with_it(tmp_path, monkeypatch):
    """Stage 1 blows up. Stage 2 cannot run without it, but stage 3 must still deliver.

    This is why ``hn-only`` runs last: it depends on nothing, so a Wikipedia failure costs one
    variant instead of the whole night.
    """
    real_train = chain_module.train_embeddings_from_lines

    def fail_on_wikipedia(*args, **kwargs):
        if kwargs.get("stage") == STAGE_WIKI:
            raise OSError("the Wikipedia subset could not be read")
        return real_train(*args, **kwargs)

    monkeypatch.setattr(chain_module, "train_embeddings_from_lines", fail_on_wikipedia)
    config = _config(tmp_path)
    manifest = run_chain(config)

    statuses = _statuses(manifest)
    assert statuses[STAGE_GATE] == "passed"
    assert statuses[STAGE_WIKI] == "failed"
    # Cannot fine-tune from vectors that do not exist, and it says so rather than crashing.
    assert statuses[STAGE_FINETUNE] == "skipped"
    # The independent variant survived.
    assert statuses[STAGE_HN] == "completed"
    assert config.artefact_path(STAGE_HN).exists()
    assert manifest["outcome"] == "partial"

    # The traceback is in the manifest, so the morning does not need the log.
    failure = next(s for s in manifest["stages"] if s["name"] == STAGE_WIKI)["failure"]
    assert failure["type"] == "OSError"
    assert "Traceback" in failure["traceback"]


def test_a_stage_out_of_wall_clock_saves_what_it_has_and_says_so(tmp_path):
    """A zero-length budget must yield rather than run to completion or lose the work."""
    config = _config(tmp_path, wikipedia_budget_seconds=0.0)
    manifest = run_chain(config)

    wikipedia = next(s for s in manifest["stages"] if s["name"] == STAGE_WIKI)
    assert wikipedia["status"] == "cut_short"
    assert wikipedia["epochs_completed"] < wikipedia["epochs_configured"]
    # Not reported as a finished night, which is the thing the flag exists to prevent.
    assert manifest["outcome"] == "partial"
    assert manifest["variants_cut_short"] == [STAGE_WIKI]

    # The work it did is on disk, both as a checkpoint and as a usable artefact.
    assert newest_checkpoint(config.checkpoint_directory, STAGE_WIKI) is not None
    assert TrainedEmbeddings.load(config.artefact_path(STAGE_WIKI)).cut_short

    # And the stage after it still ran, from the partial vectors.
    assert _statuses(manifest)[STAGE_FINETUNE] == "completed"


def test_resume_carries_on_and_the_epoch_count_does_not_drift(tmp_path):
    """Cut a stage short, resume, and it must finish exactly its configured epochs.

    The bug this guards is real and was found here: a partial epoch's loss appended to the
    per-epoch list made a resumed stage report 4 epochs completed out of 3.
    """
    config = _config(tmp_path, wikipedia_budget_seconds=0.0)
    first = run_chain(config)
    assert _statuses(first)[STAGE_WIKI] == "cut_short"

    resumed = run_chain(_config(tmp_path, resume=True))
    wikipedia = next(s for s in resumed["stages"] if s["name"] == STAGE_WIKI)
    assert wikipedia["status"] == "completed"
    assert wikipedia["epochs_completed"] == wikipedia["epochs_configured"]
    assert len(wikipedia["epoch_losses"]) == wikipedia["epochs_configured"]

    # The stages that had already finished did not retrain, and say so rather than
    # reporting a completed run they did not do.
    assert _statuses(resumed)[STAGE_HN] == "already_complete"


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


def test_wikipedia_subset_size_is_arithmetic_on_a_measured_number():
    """The subset is sized from throughput and the ceiling, not picked and then timed.

    Measured on this machine: 138,362 in-vocabulary tokens/s for skip-gram at ``k=5``,
    dimension 300, batch 1024, CPU with sparse gradients. Times the 0.96 large-vocabulary
    factor, times a 7,200 second ceiling, divided by 5 epochs:

    ``138,362 x 0.96 x 7,200 / 5 = 191,271,628 tokens``

    and at 5.9 bytes per token that is 1.129 GB, comfortably over the 300 MB floor.
    """
    tokens, approximate_bytes = wikipedia_token_budget(
        budget_seconds=7_200, epochs=5, negative_samples=5
    )
    assert tokens == pytest.approx(191_271_628, rel=1e-6)
    assert approximate_bytes / 1e9 == pytest.approx(1.129, abs=0.01)
    assert approximate_bytes > MINIMUM_WIKIPEDIA_BYTES

    # Dividing by the epoch count is what makes the ceiling a ceiling: every epoch rereads
    # the whole subset, so twice the epochs must halve the corpus.
    half, _ = wikipedia_token_budget(budget_seconds=7_200, epochs=10, negative_samples=5)
    assert half == pytest.approx(tokens / 2, rel=1e-6)

    # k=15 is the slower setting, so it buys fewer tokens in the same time.
    fewer, _ = wikipedia_token_budget(budget_seconds=7_200, epochs=5, negative_samples=15)
    assert fewer < tokens


def test_a_budget_too_small_to_be_worth_it_stops_for_a_decision(tmp_path, monkeypatch):
    """Under about 300 MB the Wikipedia variant is barely bigger than text8's 100 MB.

    That is the captain's call, not the chain's, so it reports the number and the levers
    instead of quietly shipping a small corpus.
    """
    monkeypatch.setattr(chain_module, "MEASURED_TOKENS_PER_SECOND", {5: 1_000, 15: 1_000})
    config = ChainConfig(
        output_directory=tmp_path / "out",
        corpus_directory=tmp_path / "corpora",
        dry_run=False,
        wikipedia_budget_seconds=60.0,
    )
    record = chain_module.RunManifest(config.manifest_path, config, "cpu")
    outcome = chain_module._run_wikipedia(config, record)

    assert outcome.status == "needs_decision"
    assert "below" in outcome.notes[0]
    assert "epochs" in outcome.notes[0], "the note must name the levers"
    assert not config.artefact_path(STAGE_WIKI).exists()


def test_fine_tuning_uses_a_lower_rate_and_starts_from_wikipedia(tmp_path):
    """Stage 2 adjusts Wikipedia's vectors rather than overwriting them."""
    config = _config(tmp_path)
    scratch = chain_module._stage_config(config, STAGE_HN)
    finetune = chain_module._stage_config(config, STAGE_FINETUNE)
    assert finetune.learning_rate == pytest.approx(scratch.learning_rate * 0.1)
    assert finetune.min_learning_rate == pytest.approx(scratch.min_learning_rate * 0.1)

    manifest = run_chain(config)
    stage = next(s for s in manifest["stages"] if s["name"] == STAGE_FINETUNE)
    assert stage["detail"]["initialised_from"] == STAGE_WIKI
    assert stage["learning_rate"] == pytest.approx(finetune.learning_rate)


def test_wikipedia_stage_uses_five_negatives_and_the_rest_use_fifteen(tmp_path):
    """The paper's recommendation: 5 negatives for a large corpus, 15 for a small one."""
    config = _config(tmp_path)
    assert chain_module._stage_config(config, STAGE_WIKI).negative_samples == 5
    assert chain_module._stage_config(config, STAGE_GATE).negative_samples == 15
    assert chain_module._stage_config(config, STAGE_HN).negative_samples == 15


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
    effective = chain_module._stage_config(config, STAGE_HN)
    assert manifest["dimension"] == effective.dimension
    assert manifest["epochs"] == effective.epochs
    assert manifest["dimension"] != config.dimension, "the dry run should shrink the dimension"
    assert manifest["device"] == "cpu"


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
