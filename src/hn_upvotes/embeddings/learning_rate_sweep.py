"""Settles ``SGNSConfig.learning_rate`` by measurement rather than by argument.

The rate was changed from the scaffold's 0.0025 to gensim's 0.025 on the strength of an
argument. This measures it. Four rates over one slice of text8, the same three seeds and the
same token budget for every rate, so the only difference between the runs is the rate.

The rule the sweep answers to: **the right rate is the largest one that still converges
smoothly.** A final loss cannot say that on its own, so two more things are measured. The
within-epoch loss curve, because a rate near the edge of stability reaches a respectable
final loss by lurching and one number per epoch cannot see that. And two intrinsic task
scores on the resulting vectors, because a rate can post the best loss and still produce
worse vectors.

Every comparison is checked against a noise band before it decides anything. Three seeds
exist for that reason: the first pass separated the rates by less than the sampling error on
WordSim-353, and a difference that does not survive a change of seed is not a difference.

Run with ``make lr-sweep``. It needs text8, which it downloads once into ``data/corpora``.
Results land in ``artifacts/learning-rate-sweep.json`` and in ``docs/word2vec.md``.

Needs the ``train`` extra.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from hn_upvotes.data.preprocess import Vocabulary, build_vocabulary
from hn_upvotes.embeddings import corpora
from hn_upvotes.embeddings.evaluate import analogy_accuracy, wordsim_spearman
from hn_upvotes.embeddings.train import SGNSConfig, TrainedEmbeddings, train_embeddings_from_lines

#: The four rates, spanning the scaffold's value to twice gensim's. 0.0025 is expected to
#: barely move and 0.05 is expected to destabilise; the sweep exists to find out.
SWEEP_RATES: tuple[float, ...] = (0.0025, 0.01, 0.025, 0.05)

#: Every rate is run at each of these seeds. Three rather than one because the first pass
#: separated the rates by less than the noise on the intrinsic scores, and a difference that
#: does not survive a change of seed is not a difference. Every rate gets the same three, so
#: the seed is still held fixed across the comparison.
SWEEP_SEEDS: tuple[int, ...] = (0, 1, 2)

#: Corpus tokens each rate gets. Identical for all four, which is the whole point: a rate
#: that looks better because it saw more text has not been measured. 5M tokens is 29% of
#: text8 and about a minute per rate at the measured 77,019 tokens/s for ``k=15``.
SWEEP_TOKEN_BUDGET = 5_000_000

#: Batches per point on the loss curve. At batch 1024 over 5M tokens a run is roughly 6,000
#: batches, so 50 gives about 120 points: fine enough to see a rate blow up mid-run and
#: coarse enough that the curve is not just batch noise.
LOSS_TRACE_BLOCK = 50

#: The loss every run starts at, before any step. The output matrix is initialised to zeros,
#: so every logit is 0, every sigmoid is 0.5, and the loss is ``(1 + k) * log 2``. At
#: ``k = 15`` that is 11.0904, and it is the yardstick a fall is measured against.
INITIAL_LOSS_AT_K15 = 16 * math.log(2)

#: A rate has to move the loss by at least this fraction of where it started before it counts
#: as having converged at all. At 5% of 11.0904 that is 0.55. "Barely moves" is the predicted
#: behaviour of the smallest rate and it needs a number, not an eyeball.
MINIMUM_LOSS_FALL_FRACTION = 0.05

#: The largest single rise between neighbouring blocks of the loss curve, as a fraction of
#: the total fall. This is the "smoothly" in the rule: a rate near the edge of stability can
#: still post a decent final loss by lurching, and the final number alone cannot see that.
#:
#: **Placed from the measurement, not before it.** On the 5M-token text8 slice, over three
#: seeds each, the worst lurch was 3.1% of the fall at 0.0025, 3.0% at 0.01 and 5.8% at 0.025.
#: The one seed at 0.05 that did not blow up outright lurched by 24.9%. So the threshold sits
#: between 5.8% and 24.9%: 1.7x clear of the worst stable rate and 2.5x clear of the unstable
#: one. A first pass used 0.25, chosen blind, and the 0.05 seed cleared it by 0.0009, which is
#: a threshold deciding nothing.
#:
#: This rule is the backstop rather than the main event. Two of 0.05's three seeds diverged
#: and :attr:`RateSummary.diverged` catches those; the rule is what catches the third.
MAXIMUM_UPTICK_FRACTION = 0.10

#: How far a rate's task score may sit below the best surviving rate's before it is
#: disqualified. Same 20% the gate uses against gensim, for the same reason: a rate whose
#: loss looks fine but whose vectors are meaningfully worse has not won anything.
MAXIMUM_TASK_SHORTFALL = 0.20


def bundled_evaluation_sets() -> dict[str, Path]:
    """The Google analogy set and WordSim-353, which gensim ships inside its test data.

    Both are the full standard files, 19,544 analogy questions and 353 rated word pairs.
    Reading them out of the installed gensim package means the sweep downloads no evaluation
    set and cannot drift from a copy checked in here. gensim is already a hard dependency of
    the ``train`` extra, because the gate compares against it.
    """
    import gensim

    directory = Path(gensim.__file__).parent / "test" / "test_data"
    found = {}
    for name, filename in (("analogy", "questions-words.txt"), ("wordsim", "wordsim353.tsv")):
        path = directory / filename
        if path.exists():
            found[name] = path
    return found


@dataclass(frozen=True)
class RateResult:
    """One rate at one seed: what the loss did, and whether the vectors are any good."""

    learning_rate: float
    seed: int
    loss_trace: tuple[float, ...]
    epoch_loss: float
    tokens_per_second: float
    analogy_accuracy: float
    analogy_covered: int
    wordsim_spearman: float
    wordsim_covered: int

    @property
    def diverged(self) -> bool:
        """True when the run blew up, by either of the two ways it can.

        Non-finite is the obvious one. The other is a loss that is still a number and has
        climbed above where it started: rate 0.05 ended a run at 2.5e16 against a starting
        11.09, which is finite, is unmistakably divergence, and would otherwise be reported
        as a rate whose loss "fell" by minus ten quadrillion.
        """
        if not all(math.isfinite(x) for x in self.loss_trace) or not math.isfinite(self.epoch_loss):
            return True
        return bool(self.loss_trace) and self.loss_trace[-1] > self.loss_trace[0]

    @property
    def loss_fall(self) -> float:
        """First point of the curve minus the last. Negative means the loss went up."""
        if len(self.loss_trace) < 2 or self.diverged:
            return float("nan")
        return self.loss_trace[0] - self.loss_trace[-1]

    @property
    def upticks(self) -> int:
        """Blocks where the loss rose against the block before it.

        This is the "smoothly" in "the largest rate that still converges smoothly". A rate
        near the edge of stability reaches a decent final loss by lurching, and one number
        at the end of the run cannot tell that apart from a clean descent.
        """
        if self.diverged:
            return len(self.loss_trace)
        return sum(
            1
            for before, after in zip(self.loss_trace, self.loss_trace[1:], strict=False)
            if after > before
        )

    @property
    def largest_uptick(self) -> float:
        """The biggest single rise between neighbouring blocks."""
        if self.diverged or len(self.loss_trace) < 2:
            return float("nan")
        return max(
            (
                after - before
                for before, after in zip(self.loss_trace, self.loss_trace[1:], strict=False)
            ),
            default=0.0,
        )

    def as_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "epoch_loss": self.epoch_loss,
            "loss_first_block": self.loss_trace[0] if self.loss_trace else None,
            "loss_last_block": self.loss_trace[-1] if self.loss_trace else None,
            "loss_fall": self.loss_fall,
            "upticks": self.upticks,
            "blocks": len(self.loss_trace),
            "largest_uptick": self.largest_uptick,
            "diverged": self.diverged,
            "tokens_per_second": self.tokens_per_second,
            "analogy_accuracy": self.analogy_accuracy,
            "analogy_covered": self.analogy_covered,
            "wordsim_spearman": self.wordsim_spearman,
            "wordsim_covered": self.wordsim_covered,
            "loss_trace": list(self.loss_trace),
        }


def score_rate(
    rate: float,
    seed: int,
    vocabulary: Vocabulary,
    reopen_corpus,
    base_config: SGNSConfig,
    evaluation_sets: dict[str, Path],
) -> RateResult:
    """Train one rate at one seed over the shared corpus and score the vectors."""
    config = replace(base_config, learning_rate=rate, seed=seed)
    trained = train_embeddings_from_lines(
        reopen_corpus,
        vocabulary,
        config,
        loss_trace_every=LOSS_TRACE_BLOCK,
    )
    return RateResult(
        learning_rate=rate,
        seed=seed,
        loss_trace=trained.loss_trace,
        epoch_loss=trained.epoch_losses[0] if trained.epoch_losses else float("nan"),
        tokens_per_second=trained.tokens_per_second,
        **_task_scores(trained, evaluation_sets),
    )


@dataclass(frozen=True)
class RateSummary:
    """One rate over every seed. This is what the decision is made on, not a single run."""

    learning_rate: float
    runs: tuple[RateResult, ...]

    @property
    def diverged(self) -> bool:
        """Any seed blowing up disqualifies the rate. Stability is not a majority vote.

        Rate 0.05 is why this is not a vote. It survived seed 0 and blew up on seeds 1 and 2,
        so a single-seed sweep that happened to draw seed 0 would have called it stable.
        """
        return any(run.diverged for run in self.runs)

    @property
    def survived(self) -> tuple[RateResult, ...]:
        """The runs that did not blow up. The loss aggregates are over these only.

        Averaging a 2.5e16 loss into a mean produces a number that is not about anything. The
        rate is already rejected by :attr:`diverged`, so what its table row needs to show is
        what the seeds that stayed finite did.
        """
        return tuple(run for run in self.runs if not run.diverged)

    @property
    def diverged_seeds(self) -> list[int]:
        return [run.seed for run in self.runs if run.diverged]

    @property
    def final_loss(self) -> float:
        return self._mean([run.loss_trace[-1] for run in self.survived])

    @property
    def loss_fall(self) -> float:
        """Mean fall across the seeds that stayed finite."""
        return self._mean([run.loss_fall for run in self.survived])

    @property
    def worst_uptick(self) -> float:
        """The biggest single-block rise any surviving seed produced.

        The worst case rather than the mean, for the same reason ``diverged`` is: a rate that
        lurches on one seed in three is a rate that lurches, and an overnight run gets one
        draw from that distribution.
        """
        return max((run.largest_uptick for run in self.survived), default=nan())

    @property
    def wordsim(self) -> float:
        """Mean over **every** seed, including the ones that blew up.

        A rate whose loss exploded still leaves a matrix, and that matrix is part of what the
        rate produced. Dropping its score would flatter the rate.
        """
        return self._mean([run.wordsim_spearman for run in self.runs])

    @property
    def analogy(self) -> float:
        return self._mean([run.analogy_accuracy for run in self.runs])

    @property
    def wordsim_spread(self) -> float:
        """Largest minus smallest WordSim-353 score across the seeds."""
        scores = [run.wordsim_spearman for run in self.runs]
        return float(max(scores) - min(scores)) if scores else nan()

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else nan()

    @property
    def task_noise(self) -> float:
        """How big a WordSim-353 gap has to be before it means anything.

        Two sources, and the larger wins. The sampling error on a Spearman correlation over
        ``n`` pairs is about ``1 / sqrt(n - 3)``: at the 336 WordSim-353 pairs this vocabulary
        covers that is 0.055, so two rates 0.03 apart are the same rate. The seed spread is
        the other, measured rather than derived, and it catches noise the formula does not.
        """
        covered = max((run.wordsim_covered for run in self.runs), default=0)
        sampling = 1 / math.sqrt(covered - 3) if covered > 4 else float("inf")
        return max(sampling, self.wordsim_spread)

    def as_dict(self) -> dict:
        return {
            "learning_rate": self.learning_rate,
            "seeds": [run.seed for run in self.runs],
            "final_loss": self.final_loss,
            "loss_fall": self.loss_fall,
            "worst_uptick": self.worst_uptick,
            "worst_uptick_fraction_of_fall": (
                self.worst_uptick / self.loss_fall if self.loss_fall else nan()
            ),
            "wordsim_spearman": self.wordsim,
            "wordsim_spread_across_seeds": self.wordsim_spread,
            "wordsim_noise_band": self.task_noise,
            "analogy_accuracy": self.analogy,
            "diverged": self.diverged,
            "diverged_seeds": self.diverged_seeds,
            "runs": [run.as_dict() for run in self.runs],
        }


def nan() -> float:
    return float("nan")


def _task_scores(trained: TrainedEmbeddings, evaluation_sets: dict[str, Path]) -> dict:
    """Analogy accuracy and WordSim-353 correlation, or zeros when a set is missing.

    A matrix full of NaN scores nothing rather than raising, so a diverged run still reports
    a row in the table instead of taking the sweep down with it.
    """
    if not np.all(np.isfinite(trained.matrix)):
        return {
            "analogy_accuracy": float("nan"),
            "analogy_covered": 0,
            "wordsim_spearman": float("nan"),
            "wordsim_covered": 0,
        }
    accuracy, analogy_covered = (
        analogy_accuracy(trained, evaluation_sets["analogy"])
        if "analogy" in evaluation_sets
        else (float("nan"), 0)
    )
    correlation, wordsim_covered = (
        wordsim_spearman(trained, evaluation_sets["wordsim"])
        if "wordsim" in evaluation_sets
        else (float("nan"), 0)
    )
    return {
        "analogy_accuracy": accuracy,
        "analogy_covered": analogy_covered,
        "wordsim_spearman": correlation,
        "wordsim_covered": wordsim_covered,
    }


def stability_verdict(summary: RateSummary) -> str | None:
    """Why this rate is not a stable choice, or ``None`` if it is. Loss only, no task score.

    Two tests, run before any rate is compared against any other, because these are about
    the rate on its own:

    1. **It moved the loss, downwards.** A blow-up on any seed, or a mean fall under
       :data:`MINIMUM_LOSS_FALL_FRACTION` of the starting loss, means nothing was learned.
    2. **It moved it smoothly.** A single block rising by more than
       :data:`MAXIMUM_UPTICK_FRACTION` of the total fall is a lurch, not a descent, and the
       worst seed is what counts.
    """
    if summary.diverged:
        blown = summary.diverged_seeds
        return (
            f"blew up on {len(blown)} of {len(summary.runs)} seeds ({blown}): the loss ended "
            f"at or above where it started, or stopped being a number"
        )
    floor = MINIMUM_LOSS_FALL_FRACTION * INITIAL_LOSS_AT_K15
    if summary.loss_fall < floor:
        return f"barely moved: the loss fell {summary.loss_fall:.4f}, under the {floor:.2f} floor"
    if summary.worst_uptick > MAXIMUM_UPTICK_FRACTION * summary.loss_fall:
        return (
            f"not smooth: one block rose {summary.worst_uptick:.4f}, "
            f"{summary.worst_uptick / summary.loss_fall:.1%} of the {summary.loss_fall:.4f} "
            f"it fell, over the {MAXIMUM_UPTICK_FRACTION:.0%} allowance"
        )
    return None


def task_verdict(summary: RateSummary, best: RateSummary) -> str | None:
    """Why this rate's vectors are too far behind the best stable rate's, or ``None``.

    A gap has to clear **two** bars to count, and the second is the one that matters here:

    * more than :data:`MAXIMUM_TASK_SHORTFALL` of the best score, and
    * more than :attr:`RateSummary.task_noise`, the band inside which a WordSim-353
      difference is indistinguishable from noise.

    Without the second bar this rule decides on nothing. On the 5M-token slice the four rates
    scored between -0.018 and 0.149 with a noise band of 0.055, so three of the four gaps are
    smaller than the measurement error and a percentage test alone would rank them anyway.
    """
    if summary is best:
        return None
    gap = best.wordsim - summary.wordsim
    if best.wordsim <= 0 or gap <= 0:
        return None
    if gap <= summary.task_noise:
        return None
    if summary.wordsim >= (1 - MAXIMUM_TASK_SHORTFALL) * best.wordsim:
        return None
    return (
        f"worse vectors: wordsim {summary.wordsim:.4f} against {best.wordsim:.4f} at "
        f"rate {best.learning_rate:g}, a gap of {gap:.4f} against a noise band of "
        f"{summary.task_noise:.4f}"
    )


def pick_rate(summaries: list[RateSummary]) -> tuple[float, str, dict[float, str]]:
    """The largest rate that still converges smoothly, why, and what knocked the rest out.

    A larger rate that converges gets further in the same wall clock, and this project's
    ceiling is wall clock, so among the rates that survive both verdicts the largest wins.

    The bar for the task comparison is set by the best **stable** rate, not the best rate
    overall. A rate that was thrown out for lurching does not get to disqualify the rates
    that did not lurch.
    """
    verdicts = {s.learning_rate: stability_verdict(s) for s in summaries}
    stable = [s for s in summaries if verdicts[s.learning_rate] is None]
    if not stable:
        return (float("nan"), "no rate converged smoothly", dict(_present(verdicts)))

    best = max(stable, key=lambda s: s.wordsim)
    for summary in stable:
        verdicts[summary.learning_rate] = task_verdict(summary, best)
    survivors = [s for s in stable if verdicts[s.learning_rate] is None]
    if not survivors:
        return (float("nan"), "no rate survived", dict(_present(verdicts)))

    winner = max(survivors, key=lambda s: s.learning_rate)
    return (
        winner.learning_rate,
        f"largest of the {len(survivors)} rate(s) that converged smoothly and scored within "
        f"noise of the best stable rate's wordsim ({best.wordsim:.4f} at "
        f"{best.learning_rate:g})",
        dict(_present(verdicts)),
    )


def _present(verdicts: dict[float, str | None]):
    return ((rate, why) for rate, why in verdicts.items() if why is not None)


def run_sweep(
    corpus_directory: Path = Path("data/corpora"),
    token_budget: int = SWEEP_TOKEN_BUDGET,
    rates: tuple[float, ...] = SWEEP_RATES,
    seeds: tuple[int, ...] = SWEEP_SEEDS,
    dimension: int = 300,
) -> dict:
    """Sweep every rate at every seed over the same slice of text8 and compare."""
    text8_path = corpora.download_text8(corpus_directory)

    def reopen():
        return corpora.take_tokens(corpora.stream_text8(text8_path), token_budget)

    base_config = replace(
        SGNSConfig(),
        objective="skipgram",
        dimension=dimension,
        epochs=1,
        device="cpu",
    )
    # Built once and shared, so every run is provably over the same vocabulary and the
    # learning-rate schedule reaches min_learning_rate at the same point in all of them.
    vocabulary = build_vocabulary(
        reopen(), min_count=base_config.min_count, max_size=base_config.vocabulary_cap
    )
    evaluation_sets = bundled_evaluation_sets()
    print(
        f"text8 slice: {token_budget:,} tokens, {len(vocabulary):,} word types "
        f"at min_count={base_config.min_count}",
        flush=True,
    )
    print(f"evaluation sets: {', '.join(sorted(evaluation_sets)) or 'none found'}", flush=True)
    print(f"{len(rates)} rates x {len(seeds)} seeds = {len(rates) * len(seeds)} runs", flush=True)

    summaries = []
    for rate in rates:
        runs = []
        for seed in seeds:
            print(f"\ntraining at learning_rate={rate} seed={seed}", flush=True)
            result = score_rate(rate, seed, vocabulary, reopen, base_config, evaluation_sets)
            runs.append(result)
            print(
                f"  epoch loss {result.epoch_loss:.4f}  "
                f"largest uptick {result.largest_uptick:.4f}  "
                f"wordsim {result.wordsim_spearman:.4f}  "
                f"analogy {result.analogy_accuracy:.5f}  "
                f"{result.tokens_per_second:,.0f} tokens/s",
                flush=True,
            )
        summaries.append(RateSummary(learning_rate=rate, runs=tuple(runs)))

    chosen, reason, rejected = pick_rate(summaries)
    return {
        "corpus": str(text8_path),
        "token_budget": token_budget,
        "vocabulary": len(vocabulary),
        "dimension": dimension,
        "epochs": 1,
        "seeds": list(seeds),
        "negative_samples": base_config.negative_samples,
        "batch_size": base_config.batch_size,
        "min_learning_rate": base_config.min_learning_rate,
        "loss_trace_block": LOSS_TRACE_BLOCK,
        "maximum_uptick_fraction": MAXIMUM_UPTICK_FRACTION,
        "chosen_learning_rate": chosen,
        "chosen_because": reason,
        "rejected": {str(rate): why for rate, why in rejected.items()},
        "current_default": SGNSConfig.learning_rate,
        "rates": [summary.as_dict() for summary in summaries],
    }


def summarise(report: dict) -> str:
    """The table that goes in the write-up."""
    lines = [
        f"text8 slice {report['token_budget']:,} tokens, {report['vocabulary']:,} word types, "
        f"dimension {report['dimension']}, seeds {report['seeds']}",
        "means over seeds; worst uptick is the worst single seed",
        "",
        f"{'rate':>8} {'blew up':>8} {'final loss':>11} {'fall':>8} {'worst uptick':>13} "
        f"{'of fall':>8} {'wordsim':>9} {'+-':>7} {'analogy':>9}",
    ]
    for row in report["rates"]:
        blew = f"{len(row['diverged_seeds'])}/{len(row['seeds'])}"
        lines.append(
            f"{row['learning_rate']:>8.4f} "
            f"{blew:>8} "
            f"{row['final_loss']:>11.4f} "
            f"{row['loss_fall']:>8.4f} "
            f"{row['worst_uptick']:>13.4f} "
            f"{row['worst_uptick_fraction_of_fall']:>7.1%} "
            f"{row['wordsim_spearman']:>9.4f} "
            f"{row['wordsim_noise_band']:>7.4f} "
            f"{row['analogy_accuracy']:>9.5f}"
        )
    lines.append("")
    for rate, why in sorted(report["rejected"].items(), key=lambda kv: float(kv[0])):
        lines.append(f"out  {float(rate):.4f}: {why}")
    lines.append(f"chosen: {report['chosen_learning_rate']}  ({report['chosen_because']})")
    lines.append(f"current default: {report['current_default']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-directory", type=Path, default=Path("data/corpora"))
    parser.add_argument("--token-budget", type=int, default=SWEEP_TOKEN_BUDGET)
    parser.add_argument("--dimension", type=int, default=300)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SWEEP_SEEDS))
    parser.add_argument("--rates", type=float, nargs="+", default=list(SWEEP_RATES))
    parser.add_argument("--output", type=Path, default=Path("artifacts/learning-rate-sweep.json"))
    args = parser.parse_args(argv)

    report = run_sweep(
        corpus_directory=args.corpus_directory,
        token_budget=args.token_budget,
        rates=tuple(args.rates),
        seeds=tuple(args.seeds),
        dimension=args.dimension,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(summarise(report))
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
