"""The four things that would silently ruin a word2vec implementation.

None of these needs a network or a corpus download. The synthetic corpora are generated in
memory, everything runs on CPU, and the whole file is a few seconds.

1. **It learns.** The loss falls and a word ends up nearest the synonym that was planted for
   it. Without this the other three are checks on machinery that does not work.
2. **The negative sampler draws from the 0.75-power distribution**, not uniform and not raw
   frequency. Getting this wrong costs accuracy and nothing crashes.
3. **Subsampling keeps the share it should at ``t = 1e-5``**, derived by hand below.
4. **CBOW's masked average ignores padding.** Padding uses word id 0, which is a real row of
   the matrix, so a mask bug quietly averages the unknown vector into every short context and
   divides by the wrong count.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="the train extra is not installed")

from hn_upvotes.data.preprocess import build_vocabulary  # noqa: E402
from hn_upvotes.embeddings.cbow import CBOWObjective, masked_average  # noqa: E402
from hn_upvotes.embeddings.corpora import (  # noqa: E402
    planted_synonym_corpus,
    planted_synonym_pairs,
)
from hn_upvotes.embeddings.evaluate import nearest_neighbours  # noqa: E402
from hn_upvotes.embeddings.negative_sampling import (  # noqa: E402
    build_noise_distribution,
    sample_negatives,
    subsample_probabilities,
)
from hn_upvotes.embeddings.skipgram import SkipGramObjective  # noqa: E402
from hn_upvotes.embeddings.train import (  # noqa: E402
    SGNSConfig,
    build_windows,
    train_embeddings_from_lines,
)

#: Small and CPU-only. Dimension 32 rather than the real 300, because the planted structure
#: is six independent pairs and 32 dimensions hold that comfortably.
SMOKE_CONFIG = SGNSConfig(
    dimension=32,
    window=2,
    negative_samples=15,
    epochs=8,
    batch_size=256,
    # Subsampling off: the planted corpus is 7,200 tokens over 36 word types, so every word
    # looks frequent and t=1e-5 would delete almost all of it. Test 3 covers subsampling.
    subsample_threshold=1.0,
    vocabulary_cap=1_000,
    device="cpu",
    seed=0,
)


def _train(objective: str) -> tuple:
    lines = planted_synonym_corpus()
    vocabulary = build_vocabulary(iter(lines), min_count=5, max_size=1_000)
    config = SGNSConfig(**{**SMOKE_CONFIG.__dict__, "objective": objective})
    return train_embeddings_from_lines(lambda: iter(lines), vocabulary, config), lines


@pytest.mark.parametrize("objective", ["skipgram", "cbow"])
def test_loss_falls_and_planted_synonyms_end_up_nearest(objective):
    """Both objectives learn the planted structure.

    The corpus plants six synonym pairs. Each pair's two members share every context word and
    **never co-occur with each other**, so being nearest neighbours can only come from
    distributional similarity and not from co-occurrence.

    Measured at this configuration: skip-gram's mean epoch loss falls 4.23 to 1.54 and CBOW's
    5.26 to 1.40, and both recover 6 of 6 pairs. The assertions leave room around those.
    """
    trained, _ = _train(objective)

    losses = trained.epoch_losses
    assert len(losses) == SMOKE_CONFIG.epochs
    assert losses[-1] < losses[0] * 0.6, f"loss barely moved: {losses}"

    pairs = planted_synonym_pairs()
    recovered = [
        (first, second)
        for first, second in pairs
        if nearest_neighbours(trained, first, k=1)[0][0] == second
    ]
    assert len(recovered) == len(pairs), (
        f"{objective} recovered {len(recovered)}/{len(pairs)} planted pairs"
    )


def test_initial_loss_is_the_value_a_zero_output_matrix_forces():
    """A sanity check on the loss itself, independent of any corpus.

    The output matrix starts at zeros, so every logit is exactly 0, every sigmoid is 0.5, and
    the per-example loss is ``(1 + k) * log 2``. At ``k = 15`` that is ``16 * 0.693147 =
    11.090``. This pins the loss function: a factor-of-two error or a missing negative term
    would move this number and nothing else in the suite would notice.

    The vocabulary is large so a drawn negative almost never lands on the positive word,
    which would drop a term and lower the loss.
    """
    vocabulary_size, k = 50_000, 15
    model = SkipGramObjective(vocabulary_size, dimension=16)
    centre = torch.arange(64)
    context = torch.arange(64) + 1_000
    negatives = torch.arange(64 * k).reshape(64, k) + 10_000

    loss = model(centre, context, negatives)
    assert float(loss.detach()) == pytest.approx((1 + k) * np.log(2), rel=1e-6)


def test_negative_sampler_draws_from_the_0_75_power_distribution():
    """Not uniform, and not raw frequency.

    Fixture counts 10,000 / 1,000 / 100 / 10. Raw frequency would draw the first word 1,000x
    as often as the third. The 0.75 power flattens that:

    ``10000**0.75 = 1000.000``, ``1000**0.75 = 177.828``, ``100**0.75 = 31.623``,
    ``10**0.75 = 5.623``; total ``1215.074``.

    So the expected shares are 0.82299, 0.14635, 0.02602 and 0.00463. Uniform would be 0.25
    each and raw frequency would be 0.90009, 0.09001, 0.00900 and 0.00090. All three are far
    apart, so this test tells them apart rather than merely checking a sum.
    """
    counts = np.array([10_000, 1_000, 100, 10])
    noise = build_noise_distribution(counts, power=0.75)

    expected = np.array([0.822994, 0.146352, 0.026025, 0.004628])
    assert noise == pytest.approx(expected, abs=1e-5)
    # Distinguishable from the two things it must not be.
    assert not np.allclose(noise, 0.25, atol=0.01), "sampler is uniform"
    assert not np.allclose(noise, counts / counts.sum(), atol=0.01), "sampler is raw frequency"

    drawn = sample_negatives(
        torch.as_tensor(noise, dtype=torch.float32),
        batch_size=2_000,
        k=25,
        generator=torch.Generator().manual_seed(0),
    )
    assert drawn.shape == (2_000, 25)
    observed = np.bincount(drawn.flatten().numpy(), minlength=4) / drawn.numel()
    # 50,000 draws, so the standard error on the largest share is sqrt(0.823*0.177/50000)
    # = 0.0017. A 0.01 tolerance is about six of those.
    assert observed == pytest.approx(expected, abs=0.01)


def test_subsampling_at_1e_5_keeps_the_share_worked_out_by_hand():
    """``P(keep) = min(1, sqrt(t / f))`` at ``t = 1e-5``, checked per word and in total.

    Fixture: one word at 500,000 occurrences, one at 5,000, one at 500, one at 50, so
    **505,550 tokens in total**. Frequencies are 0.989022, 0.009890, 0.000989 and 0.000099.

    Per-word keep probability:

    * ``sqrt(1e-5 / 0.989022) = 0.003180``
    * ``sqrt(1e-5 / 0.009890) = 0.031798``
    * ``sqrt(1e-5 / 0.000989) = 0.100553``
    * ``sqrt(1e-5 / 0.000099) = 0.317978``

    Each is 10x the one before it, because the frequency drops by 100x and the rule takes a
    square root.

    Expected tokens kept:
    ``500000*0.003180 + 5000*0.031798 + 500*0.100553 + 50*0.317978``
    ``= 1589.9 + 159.0 + 50.3 + 15.9 = 1815.1``, which is **0.359% of 505,550**.

    That tiny share is the point: this fixture is deliberately dominated by one very frequent
    word. The measured share on the real HN corpus at the same threshold is 31.3%, because
    real text spreads its mass over many words rather than one.
    """
    counts = np.array([500_000, 5_000, 500, 50])
    keep = subsample_probabilities(counts, threshold=1e-5)

    assert keep == pytest.approx([0.003180, 0.031798, 0.100553, 0.317978], abs=1e-6)
    expected_share = float((counts * keep).sum() / counts.sum())
    assert expected_share == pytest.approx(0.003590, abs=1e-5)

    # And the rule never returns a probability above 1 for a rare word.
    rare = subsample_probabilities(np.array([1, 10, 100_000]), threshold=1e-3)
    assert rare.max() <= 1.0
    assert rare[0] == 1.0


def test_masked_average_ignores_padding_positions():
    """A ragged window must not average in the padding.

    Padding uses word id 0, the unknown token, which is a real row. Two real context vectors
    ``a`` and ``b`` in a width-10 row must average to ``(a + b) / 2``. Averaging the whole
    padded row instead gives ``(a + b + 8 * pad) / 10``, a different vector even when ``pad``
    is zero, because the divisor is wrong.
    """
    vectors = torch.zeros(1, 10, 4)
    vectors[0, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    vectors[0, 1] = torch.tensor([3.0, 4.0, 5.0, 6.0])
    # Padding positions carry a deliberately loud value, so including them cannot go unseen.
    vectors[0, 2:] = 100.0
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[0, :2] = True

    averaged = masked_average(vectors, mask)
    assert averaged[0].tolist() == pytest.approx([2.0, 3.0, 4.0, 5.0])
    # The unmasked mean, which is what a mask bug would produce.
    assert vectors.mean(dim=1)[0][0].item() == pytest.approx(80.4)


def test_masked_average_of_an_all_padding_row_is_zero_not_a_division_by_zero():
    """A row with no real context averages to zeros rather than producing NaN."""
    vectors = torch.full((1, 6, 3), 7.0)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    averaged = masked_average(vectors, mask)
    assert torch.isfinite(averaged).all()
    assert averaged.abs().sum().item() == 0.0


def test_cbow_forward_result_matches_the_mask_not_the_padding():
    """The masking has to survive through ``forward``, not only in the helper.

    Same batch twice: once with the padding positions set to word id 0 and once to a
    different in-vocabulary id. The mask says neither is real, so the loss must be identical.
    A forward pass that ignored the mask would give two different numbers.
    """
    model = CBOWObjective(vocabulary_size=500, dimension=8, seed=0)
    centre = torch.tensor([7, 8])
    negatives = torch.arange(2 * 5).reshape(2, 5) + 100
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    with_zero_padding = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
    with_other_padding = torch.tensor([[1, 2, 42, 99], [3, 77, 12, 5]])

    first = float(model(with_zero_padding, centre, negatives, mask).detach())
    second = float(model(with_other_padding, centre, negatives, mask).detach())
    assert first == pytest.approx(second, rel=1e-6)


def test_build_windows_never_reaches_outside_the_sequence():
    """Every masked-in neighbour is a real position, and no position is its own context."""
    rng = np.random.default_rng(0)
    tokens = np.arange(100, 108)  # distinctive ids, so a wrong gather is visible
    context, mask = build_windows(tokens, window=5, dynamic=True, rng=rng)

    assert context.shape == (8, 10)
    assert mask.shape == (8, 10)
    for row, centre in enumerate(tokens):
        present = context[row][mask[row]]
        assert centre not in present, "a word is its own context"
        assert set(present.tolist()) <= set(tokens.tolist()), "gathered outside the sequence"
    # Every row of a length-8 sequence has at least one neighbour.
    assert mask.any(axis=1).all()


def test_dynamic_window_is_narrower_than_the_static_one():
    """The dynamic window is the paper's cheap distance weighting, so it must actually vary.

    Sampling the reach uniformly from 1 to 5 gives a mean of 3, so a dynamic window keeps
    about 3/5 of the positions a static one would, away from the sequence edges.
    """
    rng = np.random.default_rng(0)
    tokens = np.arange(2_000)
    _, static_mask = build_windows(tokens, window=5, dynamic=False, rng=rng)
    _, dynamic_mask = build_windows(tokens, window=5, dynamic=True, rng=rng)

    static_width = static_mask.sum() / len(tokens)
    dynamic_width = dynamic_mask.sum() / len(tokens)
    assert static_width == pytest.approx(10.0, abs=0.1)
    # Mean reach 3 out of 5 means 6 of the 10 positions, so the ratio is 0.6.
    assert dynamic_width / static_width == pytest.approx(0.6, abs=0.05)


def test_the_two_objectives_share_one_sampler_and_one_loss():
    """The comparison between CBOW and Skip-gram only means something if the rest is shared.

    Both must inherit the sampler, the loss and the two matrices from the same place, and
    define nothing of their own except ``forward``. A duplicated sampler is how the two drift
    apart and the experiment stops being a comparison of objectives.
    """
    from hn_upvotes.embeddings.negative_sampling import NegativeSamplingObjective

    # Dunders the interpreter puts on every class body, including __firstlineno__ and
    # __static_attributes__ which Python 3.13 added.
    interpreter_added = {
        "__doc__",
        "__module__",
        "__qualname__",
        "__firstlineno__",
        "__static_attributes__",
    }
    for objective in (CBOWObjective, SkipGramObjective):
        assert issubclass(objective, NegativeSamplingObjective)
        own = set(vars(objective)) - interpreter_added
        assert own == {"forward"}, f"{objective.__name__} defines more than forward: {own}"


def test_input_embeddings_returns_the_input_matrix_and_the_averaging_variant():
    """The embedding is the input matrix. The averaged variant is a different matrix."""
    model = SkipGramObjective(vocabulary_size=50, dimension=4, seed=0)
    with torch.no_grad():
        model.output_matrix.weight.fill_(1.0)

    plain = model.input_embeddings()
    assert torch.equal(plain, model.input_matrix.weight.detach())

    averaged = model.input_embeddings(average_with_context=True)
    expected = (model.input_matrix.weight.detach() + 1.0) / 2.0
    assert torch.allclose(averaged, expected)
