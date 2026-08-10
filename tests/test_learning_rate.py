"""The learning rate, the batch-size scaling, and the machinery that measured both.

The rate and the batch scaling were both argued before they were measured. They are measured
now, on text8, and these tests hold the conclusions in place:

* the default rate is what the sweep picked, and cannot drift back silently,
* the rule that picked it does not decide on differences smaller than the measurement error,
* the batch-size multiplication in ``optimiser_learning_rate`` is arithmetic that holds.

The measurements themselves are ``make lr-sweep`` and ``make batch-scaling``, both of which
train on text8 and take minutes. Nothing here downloads or trains on a real corpus: these
test the decision rules and the plumbing, against numbers those runs produced.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch", reason="the train extra is not installed")

from hn_upvotes.data.preprocess import build_vocabulary  # noqa: E402
from hn_upvotes.embeddings import learning_rate_sweep as sweep  # noqa: E402
from hn_upvotes.embeddings.batch_scaling import (  # noqa: E402
    MEASURED_RESIDUAL_GAP,
    MINIMUM_CLOSURE_FACTOR,
    Arm,
    ArmResult,
    count_examples,
    verdict,
)
from hn_upvotes.embeddings.corpora import take_tokens, topic_corpus  # noqa: E402
from hn_upvotes.embeddings.train import (  # noqa: E402
    SGNSConfig,
    optimiser_learning_rate,
    train_embeddings_from_lines,
)

#: What ``make lr-sweep`` picked. See ``docs/word2vec.md`` and
#: ``artifacts/learning-rate-sweep.json`` for the run behind it.
MEASURED_LEARNING_RATE = 0.025


def _summary(rate: float, trace: list[float], wordsim: float, covered: int = 336, seeds: int = 3):
    """A :class:`RateSummary` built from a loss curve, for testing the decision rule."""
    runs = tuple(
        sweep.RateResult(
            learning_rate=rate,
            seed=seed,
            loss_trace=tuple(trace),
            epoch_loss=trace[-1],
            tokens_per_second=1.0,
            analogy_accuracy=0.0,
            analogy_covered=1,
            wordsim_spearman=wordsim,
            wordsim_covered=covered,
        )
        for seed in range(seeds)
    )
    return sweep.RateSummary(learning_rate=rate, runs=runs)


def test_the_default_learning_rate_is_the_one_the_sweep_measured():
    """Pins the rate against a silent drift back to the scaffold's value or gensim's.

    Measured on a 5,000,000-token slice of text8, four rates at three seeds each, identical
    corpus and identical vocabulary, one epoch, dimension 300, ``k=15``:

    * **0.0025** falls to a mean loss of 5.54 while the others reach 3.9 to 4.2, and scores
      -0.018 on WordSim-353, which is nothing. Too small.
    * **0.01** converges smoothly to 4.00.
    * **0.025** converges smoothly to 3.93, the lowest of the four, with the largest lurch in
      its curve at 4.3% of its total fall.
    * **0.05** reaches a worse final loss than 0.025 and gets there by lurching: its curve
      goes 10.16, 5.44, 6.29, 5.05, 5.47 over the first fifth of the run, with a single block
      rising 1.50, which is 24.9% of its total fall and 5.8x the worst lurch at 0.025.

    So the largest rate that still converges smoothly is 0.025, which is gensim's alpha and
    what the code already had. The argument for it survives being measured; the measurement
    is the reason to keep it, not the argument.
    """
    assert SGNSConfig.learning_rate == pytest.approx(MEASURED_LEARNING_RATE)
    # gensim's own default, which the gate compares against. A mismatch makes the gate a
    # comparison of two learning rates rather than two implementations.
    assert SGNSConfig.min_learning_rate == pytest.approx(1e-4)
    # The scaffold's value, which was an Adam-scale rate. It measured worst of the four.
    assert SGNSConfig.learning_rate != pytest.approx(0.0025)


def test_the_sweep_throws_out_a_rate_that_barely_moves_the_loss():
    """0.0025's failure mode: a clean descent that has not got anywhere by the end."""
    crawling = _summary(0.0025, [11.09, 11.05, 11.02, 10.99], wordsim=0.0)
    reason = sweep.stability_verdict(crawling)
    assert reason is not None
    assert "barely moved" in reason


def test_a_rate_that_blows_up_on_one_seed_in_three_is_out():
    """Stability is not a majority vote, and this is the finding that made three seeds worth it.

    Rate 0.05 survived seed 0 and blew up on seeds 1 and 2, ending at 1.2e11 and 2.5e16
    against a starting loss of 11.09. A single-seed sweep drew seed 0 and would have called it
    the winner. An overnight run gets one draw from that distribution.
    """
    surviving = sweep.RateResult(
        learning_rate=0.05,
        seed=0,
        loss_trace=(10.16, 5.44, 4.15),
        epoch_loss=4.15,
        tokens_per_second=1.0,
        analogy_accuracy=0.00176,
        analogy_covered=13_095,
        wordsim_spearman=0.1487,
        wordsim_covered=336,
    )
    exploded = replace(surviving, seed=1, loss_trace=(10.22, 8.8e12, 1.226e11), epoch_loss=1.226e11)
    assert not surviving.diverged
    # Finite, and unmistakably a blow-up. The non-finite test alone would miss it.
    assert math.isfinite(exploded.loss_trace[-1])
    assert exploded.diverged

    summary = sweep.RateSummary(learning_rate=0.05, runs=(surviving, exploded))
    reason = sweep.stability_verdict(summary)
    assert reason is not None
    assert "blew up on 1 of 2 seeds" in reason
    # The loss aggregates skip the blown-up seed, or the table shows a mean of 6e10.
    assert summary.final_loss == pytest.approx(4.15)
    assert summary.diverged_seeds == [1]


def test_the_sweep_throws_out_a_rate_that_lurches():
    """0.05's failure mode: it gets there, but not smoothly.

    The numbers are the ones measured. The lurching curve rises 1.50 in one block against a
    total fall of 6.01, which is 24.9%, over the 10% allowance. The smooth curve's worst
    block rises 0.30 against a fall of 6.94, which is 4.3%.
    """
    lurching = _summary(0.05, [10.16, 5.44, 6.94, 5.05, 4.15], wordsim=0.15)
    reason = sweep.stability_verdict(lurching)
    assert reason is not None
    assert "not smooth" in reason

    smooth = _summary(0.025, [10.87, 4.83, 4.13, 3.88, 3.93], wordsim=0.06)
    assert sweep.stability_verdict(smooth) is None


def test_the_sweep_does_not_rank_rates_on_a_difference_smaller_than_the_noise():
    """The bug this guards would have picked 0.01 over 0.025 on a 0.03 wordsim gap.

    WordSim-353 covers 336 pairs at this vocabulary, so the sampling error on a Spearman
    correlation is about ``1 / sqrt(333) = 0.055``. The measured spread between 0.01 and
    0.025 was 0.033, which is inside that band. A plain 20% shortfall test calls 0.033 out of
    0.091 a 36% shortfall and throws the better rate away on noise.
    """
    smooth = [10.87, 4.83, 4.13, 3.88, 3.93]
    better = _summary(0.01, smooth, wordsim=0.0909)
    inside_noise = _summary(0.025, smooth, wordsim=0.0577)
    assert inside_noise.task_noise == pytest.approx(1 / math.sqrt(333), rel=0.01)
    assert sweep.task_verdict(inside_noise, better) is None

    # A gap that is both over 20% and wider than the band is a real shortfall.
    outside_noise = _summary(0.0025, smooth, wordsim=-0.018)
    reason = sweep.task_verdict(outside_noise, better)
    assert reason is not None
    assert "worse vectors" in reason


def test_the_noise_band_widens_when_the_seeds_disagree():
    """A rate whose score swings with the seed has not measured anything either."""
    steady = _summary(0.025, [11.0, 5.0, 4.0], wordsim=0.30, covered=100_000)
    assert steady.wordsim_spread == pytest.approx(0.0)
    assert steady.task_noise == pytest.approx(1 / math.sqrt(99_997), abs=1e-4)

    runs = list(steady.runs)
    swinging = sweep.RateSummary(
        learning_rate=0.025,
        runs=(runs[0], replace(runs[1], wordsim_spearman=0.10)),
    )
    assert swinging.task_noise == pytest.approx(0.20, abs=1e-6)


def test_the_sweep_picks_the_largest_rate_that_survives_both_tests():
    """The rule, end to end, on the shape the measurement produced."""
    smooth = [10.87, 4.83, 4.13, 3.88, 3.93]
    summaries = [
        _summary(0.0025, [11.09, 11.05, 11.02, 10.99], wordsim=-0.018),
        _summary(0.01, smooth, wordsim=0.0909),
        _summary(0.025, smooth, wordsim=0.0577),
        _summary(0.05, [10.16, 5.44, 6.94, 5.05, 4.15], wordsim=0.1487),
    ]
    chosen, reason, rejected = sweep.pick_rate(summaries)
    assert chosen == pytest.approx(MEASURED_LEARNING_RATE)
    assert "largest" in reason
    assert "barely moved" in rejected[0.0025]
    assert "not smooth" in rejected[0.05]
    assert 0.01 not in rejected and 0.025 not in rejected


def test_a_rate_disqualified_for_lurching_does_not_set_the_bar_for_the_others():
    """0.05 scored the best wordsim and was thrown out. It must not then disqualify 0.025.

    Letting an unstable rate set the task bar would mean the rule rejects every stable rate
    and picks nothing, which is the opposite of what the sweep is for.
    """
    summaries = [
        _summary(0.025, [10.87, 4.83, 4.13, 3.88, 3.93], wordsim=0.0577),
        _summary(0.05, [10.16, 5.44, 6.94, 5.05, 4.15], wordsim=0.9),
    ]
    chosen, _, rejected = sweep.pick_rate(summaries)
    assert chosen == pytest.approx(0.025)
    assert 0.025 not in rejected


def test_the_optimiser_rate_is_the_per_example_rate_times_the_batch():
    """The arithmetic the batch-scaling measurement went and checked against a real run."""
    assert optimiser_learning_rate(0.025, 1024) == pytest.approx(25.6)
    assert optimiser_learning_rate(0.025, 1) == pytest.approx(0.025)
    # Linear in both, which is the whole claim.
    assert optimiser_learning_rate(0.05, 512) == pytest.approx(
        2 * optimiser_learning_rate(0.025, 512)
    )


def _arm(name, batch_size, rate, losses, scale_by_batch=True) -> ArmResult:
    return ArmResult(
        name=name,
        batch_size=batch_size,
        scale_by_batch=scale_by_batch,
        optimiser_rate=rate,
        epoch_losses=tuple(losses),
        steps=1,
        seconds=1.0,
        matrix=np.zeros((4, 2)),
    )


def _measured_arms() -> list[ArmResult]:
    """The four arms as ``make batch-scaling`` measured them.

    text8, 120,000 tokens, 3,023 word types, dimension 64, 2 epochs, seed 0, 83,331 training
    examples per epoch identical for every arm.
    """
    return [
        _arm("batch 1", 1, 0.025, [7.8857, 4.8183]),
        _arm("batch 32", 32, 0.8, [8.7407, 4.9987]),
        _arm("batch 1024", 1024, 25.6, [9.3678, 5.4211]),
        _arm("batch 1024, unscaled", 1024, 0.025, [11.0790, 11.0790], scale_by_batch=False),
    ]


def test_the_batch_scaling_claim_holds_on_the_numbers_it_measured():
    """The measured result, pinned. Batch 1024 at 25.6 trains; at 0.025 it does not move.

    Final losses: batch 1 at an optimiser rate of 0.025 reaches 4.8183, batch 1024 at 25.6
    reaches 5.4211, and batch 1024 at an unscaled 0.025 sits at 11.0790, which is the
    ``(1 + k) * log 2`` value a zero output matrix forces before any training. So the
    multiplication is the difference between learning and not learning.
    """
    holds, message = verdict(_measured_arms())
    assert holds
    assert "25.6" in message
    # The control is quoted, because it is what shows the multiplication is load-bearing.
    assert "11.0790" in message

    # 10.4x nearer the reference than the control is, well over the 3x the rule asks for.
    scaled_gap = abs(5.4211 - 4.8183)
    control_gap = abs(11.0790 - 4.8183)
    assert control_gap / scaled_gap == pytest.approx(10.4, abs=0.1)
    assert control_gap / scaled_gap > MINIMUM_CLOSURE_FACTOR


def test_the_residual_gap_from_batching_is_recorded_and_grows_with_the_batch():
    """The scaling is right in kind and not exact, and the residual has a measured size.

    Updates inside one batch do not see each other, where a batch-1 run's do. That shows up
    as a gap that grows with the batch: 3.7% at batch 32 and 12.5% at batch 1024, both
    against batch 1's 4.8183. It is a cost of batching, not an error in the rate, so it is
    reported rather than gated.
    """
    from hn_upvotes.embeddings.batch_scaling import _residual_gaps

    gaps = _residual_gaps(_measured_arms())
    assert gaps["batch 32"] == pytest.approx(0.037, abs=0.001)
    assert gaps["batch 1024"] == pytest.approx(MEASURED_RESIDUAL_GAP, abs=0.001)
    assert gaps["batch 32"] < gaps["batch 1024"], "the gap must grow with the batch size"


def test_the_batch_scaling_verdict_fails_when_the_large_batch_diverges():
    """If the multiplication were spurious, batch 1024 at 25.6 would blow up on step one."""
    results = [
        _arm("batch 1", 1, 0.025, [5.0, 4.2]),
        _arm("batch 1024", 1024, 25.6, [float("nan"), float("nan")]),
    ]
    holds, message = verdict(results)
    assert not holds
    assert "diverged" in message
    assert "wrong" in message


def test_the_batch_scaling_verdict_fails_when_the_multiplication_buys_nothing():
    """A multiplication that leaves the arm no nearer the reference than the control is."""
    results = [
        _arm("batch 1", 1, 0.025, [5.0, 4.0]),
        _arm("batch 1024", 1024, 25.6, [10.0, 9.5]),
        _arm("batch 1024, unscaled", 1024, 0.025, [11.0, 11.0], scale_by_batch=False),
    ]
    holds, message = verdict(results)
    assert not holds
    assert "not doing the work" in message


def test_the_batch_scaling_verdict_needs_the_control_to_have_run():
    """Without the unscaled control there is no comparison, so it is not a pass."""
    holds, message = verdict(
        [_arm("batch 1", 1, 0.025, [5.0, 4.0]), _arm("batch 1024", 1024, 25.6, [5.0, 4.2])]
    )
    assert not holds
    assert "control" in message


def test_every_batch_size_is_fed_the_same_examples_in_the_same_order():
    """The comparison is "same examples, different batch size", so this is its premise.

    The feeder subsamples and draws its window per line from a seeded generator, neither of
    which depends on the batch size, so the example stream is identical and only its grouping
    changes. If that were not true, the batch-scaling arms would differ by their training
    data as well as their batch size and the measurement would say nothing.
    """
    from hn_upvotes.embeddings.train import (
        BatchFeeder,
        encode_corpus,
        subsample_probabilities,
    )

    lines, _ = topic_corpus(topics=8, words_per_topic=12, lines_per_topic=40, line_length=10)
    vocabulary = build_vocabulary(iter(lines), min_count=1, max_size=200)
    # Subsampling off, so the check is about the batching and not about how much of a
    # 96-word synthetic vocabulary the frequent-word rule happens to throw away.
    base = replace(SGNSConfig(), dimension=8, epochs=1, seed=0, subsample_threshold=1.0)
    keep = subsample_probabilities(vocabulary.counts, base.subsample_threshold)

    streams = {}
    for batch_size in (1, 32, 1024):
        feeder = BatchFeeder(replace(base, batch_size=batch_size), keep, seed=base.seed)
        batches = list(feeder.batches(encode_corpus(iter(lines), vocabulary)))
        streams[batch_size] = (
            np.concatenate([b.centre_ids for b in batches]),
            np.concatenate([b.context_ids for b in batches]),
        )

    reference = streams[1]
    assert reference[0].size > 1000, "the fixture is too small to be a real check"
    for batch_size, stream in streams.items():
        assert np.array_equal(stream[0], reference[0]), f"batch {batch_size} centres differ"
        assert np.array_equal(stream[1], reference[1]), f"batch {batch_size} contexts differ"

    # And the count the measurement reports is that same number.
    assert count_examples(lambda: iter(lines), vocabulary, base) == reference[0].size


def test_the_unscaled_control_arm_reports_the_rate_it_actually_uses():
    """The control exists to show the multiplication matters, so it must not silently apply it."""
    assert Arm("scaled", 1024).optimiser_rate(0.025) == pytest.approx(25.6)
    assert Arm("unscaled", 1024, scale_by_batch=False).optimiser_rate(0.025) == pytest.approx(0.025)


def test_take_tokens_cuts_a_stream_at_the_budget():
    """text8 is one 100 MB line, so a budget cannot be applied by counting lines."""
    lines = [["a", "b", "c"], ["d", "e"], ["f", "g", "h", "i"]]
    assert list(take_tokens(iter(lines), None)) == lines
    assert list(take_tokens(iter(lines), 5)) == [["a", "b", "c"], ["d", "e"]]
    # A line is cut mid-way rather than dropped or overshot.
    assert list(take_tokens(iter(lines), 7)) == [["a", "b", "c"], ["d", "e"], ["f", "g"]]
    assert sum(len(line) for line in take_tokens(iter(lines), 4)) == 4
    assert list(take_tokens(iter(lines), 0)) == []


def test_the_loss_trace_records_one_point_per_block_including_the_tail():
    """The sweep needs a within-epoch curve. One number per epoch cannot see a rate lurch."""
    lines, _ = topic_corpus(topics=8, words_per_topic=12, lines_per_topic=40, line_length=10)
    vocabulary = build_vocabulary(iter(lines), min_count=1, max_size=200)
    config = replace(
        SGNSConfig(),
        dimension=8,
        epochs=1,
        batch_size=64,
        seed=0,
        subsample_threshold=1.0,
        device="cpu",
    )
    plain = train_embeddings_from_lines(lambda: iter(lines), vocabulary, config)
    traced = train_embeddings_from_lines(
        lambda: iter(lines), vocabulary, config, loss_trace_every=5
    )
    assert plain.loss_trace == ()
    assert len(traced.loss_trace) > 1
    assert all(math.isfinite(x) for x in traced.loss_trace)
    # The trace covers the same run as the epoch mean, so the two must agree on the average.
    assert np.mean(traced.loss_trace) == pytest.approx(traced.epoch_losses[0], rel=0.05)
    # The curve starts where a zero output matrix forces it to: (1 + k) * log 2.
    assert traced.loss_trace[0] == pytest.approx(
        (1 + config.negative_samples) * math.log(2), rel=0.1
    )
