"""The three Phase 1 measurement gates, and the score distribution behind them.

Each gate answers one question and each answer changes a default in
``target/normalise.BaselineConfig``. A gate that produces a plot and changes nothing
has not been finished.

1. Does the centre of ``log1p(score)`` drift by year? If it is flat, there is no centre
   for the transform to subtract.
2. Does the spread drift too? If only the centre moves, subtract a trailing median and
   do not divide.
3. How long does a score take to settle? This sets ``BaselineConfig.lag``.

Gates 1 and 2 both came back flat, and both are measured on statistics that the floor
spike pins: 57.6% of stories score 1 or 2. A flat answer here means the transform has
nothing to correct, **not** that scoring is stable. It is not. The 99th percentile of raw
score went from 38 points in 2007 to 355 in 2025, which is why
:func:`plot_score_distribution` and the per-percentile table in ``docs/design.md`` are
part of the result rather than colour. Tail drift is unhandled and is an open risk for
Phase 2.

Gate 3 needs one instant at which many posts of many ages had their scores read. The
brief's plan was to use ``committed_at`` from the dataset's ``stats.csv``, on the premise
that every score in a monthly file was read from the HN API at that instant. **That
premise does not hold for this dataset**, and it fails in exactly the months where it
would have been useful. See ``docs/design.md``. So the same cross-sectional method runs
against a live read of the HN API, where the observation instant is true by construction:
:func:`fetch_live_scores` then :func:`live_observation_ages`.

The caveat is unchanged and is stated in the README: these are different posts at
different ages, not the same post tracked over time.

Plot colours come from the categorical palette in the ``dataviz`` skill reference,
slots 1 to 3, validated for colour-vision deficiency at ``--pairs all``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: Categorical slots 1-3. Validated: worst all-pairs CVD dE 9.2, normal-vision dE 24.0.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdbd6"

#: Where the README reads its figures from.
IMAGE_DIR = Path("docs/img")


def _style(ax) -> None:
    """Recessive axes and grid, so the data is the only strong thing on the canvas."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def _figure(width: float = 7.2, height: float = 3.6):  # noqa: ANN202
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    return fig, ax


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # Small files. The README embeds these and the repo should not carry megabytes.
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.12)
    return path


# --------------------------------------------------------------------------------------
# Gates 1 and 2: does the centre drift, and does the spread drift with it
# --------------------------------------------------------------------------------------


def yearly_score_stats(frame: pd.DataFrame) -> pd.DataFrame:
    """Centre and spread of ``log1p(score)`` per calendar year.

    Both gates read this one table: gate 1 takes the median column, gate 2 takes the
    standard deviation and the interquartile range.
    """
    y = np.log1p(frame["score"].to_numpy(dtype=np.float64))
    year = pd.to_datetime(frame["time"]).dt.year
    grouped = pd.DataFrame({"year": year.to_numpy(), "log1p_score": y}).groupby("year")[
        "log1p_score"
    ]

    stats = pd.DataFrame(
        {
            "n": grouped.size(),
            "median": grouped.median(),
            "mean": grouped.mean(),
            "std": grouped.std(ddof=0),
            "q25": grouped.quantile(0.25),
            "q75": grouped.quantile(0.75),
        }
    )
    stats["iqr"] = stats["q75"] - stats["q25"]
    stats["median_raw_score"] = np.expm1(stats["median"]).round(1)
    return stats.reset_index()


def drift_ratio(stats: pd.DataFrame, column: str, first_full_year_min_n: int = 10_000) -> float:
    """Last year over first year for ``column``, ignoring thin early years.

    Years below ``first_full_year_min_n`` rows are dropped before the ratio is taken,
    because 61 posts in 2006 is not an era, it is a launch week.
    """
    usable = stats[stats["n"] >= first_full_year_min_n]
    if usable.empty:
        raise ValueError("no year clears the row threshold")
    return float(usable[column].iloc[-1] / usable[column].iloc[0])


def _note_excluded_years(ax, stats: pd.DataFrame) -> None:
    """Name any year missing from the series, so the line's gap is not read as data.

    2023 and 2026 are absent because their archived scores were captured at submission.
    A reader who meets the plot without the README should still see that.
    """
    years = set(stats["year"].astype(int))
    span = range(int(stats["year"].min()), int(stats["year"].max()) + 1)
    missing = [y for y in span if y not in years]
    if not missing:
        return
    ax.annotate(
        f"{', '.join(str(y) for y in missing)} excluded: archived scores not final",
        xy=(0.0, -0.30),
        xycoords="axes fraction",
        fontsize=8,
        color=MUTED,
    )


def plot_yearly_centre(stats: pd.DataFrame, path: Path = IMAGE_DIR / "gate1-median-by-year.png"):
    """Gate 1. Median ``log1p(score)`` per year, with the mean for reference."""
    fig, ax = _figure()
    ax.plot(stats["year"], stats["median"], color=SERIES[0], linewidth=2, marker="o", markersize=4)
    ax.plot(
        stats["year"],
        stats["mean"],
        color=SERIES[1],
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=4,
    )
    # Direct labels rather than a legend box: two series, both worth naming in place.
    ax.annotate(
        "median",
        (stats["year"].iloc[-1], stats["median"].iloc[-1]),
        textcoords="offset points",
        xytext=(6, -2),
        color=SERIES[0],
        fontsize=9,
        fontweight="bold",
    )
    ax.annotate(
        "mean",
        (stats["year"].iloc[-1], stats["mean"].iloc[-1]),
        textcoords="offset points",
        xytext=(6, -2),
        color=SERIES[1],
        fontsize=9,
        fontweight="bold",
    )
    ax.set_title("Gate 1: centre of log1p(score) by year", fontsize=11, loc="left")
    ax.set_xlabel("year of submission")
    ax.set_ylabel("log1p(score)")
    ax.set_xlim(stats["year"].min() - 0.5, stats["year"].max() + 1.6)
    _note_excluded_years(ax, stats)
    return _save(fig, path)


def plot_yearly_spread(stats: pd.DataFrame, path: Path = IMAGE_DIR / "gate2-spread-by-year.png"):
    """Gate 2. Standard deviation and interquartile range of ``log1p(score)`` per year."""
    fig, ax = _figure()
    ax.plot(stats["year"], stats["std"], color=SERIES[0], linewidth=2, marker="o", markersize=4)
    ax.plot(stats["year"], stats["iqr"], color=SERIES[2], linewidth=2, marker="o", markersize=4)
    ax.annotate(
        "std",
        (stats["year"].iloc[-1], stats["std"].iloc[-1]),
        textcoords="offset points",
        xytext=(6, -2),
        color=SERIES[0],
        fontsize=9,
        fontweight="bold",
    )
    # Direct label required: this hue sits below 3:1 contrast on the light surface.
    ax.annotate(
        "IQR",
        (stats["year"].iloc[-1], stats["iqr"].iloc[-1]),
        textcoords="offset points",
        xytext=(6, -2),
        color=SERIES[2],
        fontsize=9,
        fontweight="bold",
    )
    ax.set_title("Gate 2: spread of log1p(score) by year", fontsize=11, loc="left")
    ax.set_xlabel("year of submission")
    ax.set_ylabel("log1p(score)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(stats["year"].min() - 0.5, stats["year"].max() + 1.6)
    _note_excluded_years(ax, stats)
    return _save(fig, path)


# --------------------------------------------------------------------------------------
# Gate 3: how long a score takes to settle
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SettlingResult:
    """The settling curve and what it implies for ``BaselineConfig.lag``.

    ``flat_from_hours`` is the first bucket at or after which every later bucket's
    median stays within ``tolerance`` of the final plateau. ``None`` means the curve
    never flattens inside the observed range, which is a real answer and is reported as
    one rather than rounded into a convenient number.
    """

    curve: pd.DataFrame
    statistic: str
    plateau_median: float
    flat_from_hours: float | None
    tolerance: float
    months: tuple[str, ...]


def fetch_live_scores(
    ids, workers: int = 32, timeout: float = 20.0
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Read current scores for ``ids`` from the live HN API.

    Returns the scores and the instant the read finished. This is the observation
    timestamp gate 3 needs, and unlike the archive's ``committed_at`` it is true by
    construction: every score in the result was read within the same few minutes.

    Runs about 100 items a second at the default concurrency, measured against
    ``hacker-news.firebaseio.com``.
    """
    import concurrent.futures as futures
    import json
    import urllib.request

    def one(item_id: int):
        url = f"https://hacker-news.firebaseio.com/v0/item/{int(item_id)}.json"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as handle:
                payload = json.load(handle)
        except Exception:
            return None
        if not payload:
            return None
        return payload.get("score")

    ids = list(ids)
    with futures.ThreadPoolExecutor(workers) as pool:
        scores = list(pool.map(one, ids))
    observed_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
    frame = pd.DataFrame({"id": ids, "live_score": scores}).dropna(subset=["live_score"])
    frame["live_score"] = frame["live_score"].astype(int)
    return frame, observed_at


def live_observation_ages(
    frame: pd.DataFrame, live_scores: pd.DataFrame, observed_at: pd.Timestamp
) -> pd.DataFrame:
    """Age at observation for stories whose scores were just read from the live API.

    The brief's method, with ``now`` in place of ``committed_at``. One observation
    instant, many submission times, so a single read gives posts at every age.
    """
    merged = frame.merge(live_scores, on="id", how="inner")
    times = pd.to_datetime(merged["time"])
    out = pd.DataFrame(
        {
            "time": times,
            "score": merged["live_score"].to_numpy(),
            "month": times.dt.strftime("%Y-%m").to_numpy(),
        }
    )
    out["age_hours"] = (observed_at - out["time"]).dt.total_seconds() / 3600.0
    return out[out["age_hours"] > 0].reset_index(drop=True)


def _age_bucket_edges(max_hours: float, bucket_hours: float = 1.0) -> np.ndarray:
    """Fine buckets to 72 hours, then daily. The brief's bucketing.

    ``bucket_hours`` widens the fine part. The live sample is about 40 stories an hour,
    so hourly buckets are too thin to read and 3-hourly is the usable resolution.
    """
    fine = np.arange(0, 72 + bucket_hours, bucket_hours, dtype=np.float64)
    daily = np.arange(96, max(max_hours, 96.0) + 24, 24, dtype=np.float64)
    return np.unique(np.concatenate([fine, daily]))


def settling_curve(
    ages: pd.DataFrame,
    months: list[str] | None = None,
    min_bucket_n: int = 200,
    statistic: str = "median_score",
    plateau_from_hours: float = 14 * 24,
    tolerance_fraction: float = 0.05,
    bucket_hours: float = 1.0,
) -> SettlingResult:
    """Score against age at observation, for gate 3.

    ``ages`` comes from :func:`live_observation_ages`. It needs ``age_hours``, ``score``
    and ``month``.

    ``statistic`` picks the column the plateau test runs on. The brief specifies the
    median; the median of HN scores is pinned to the floor spike at 1 point, so
    ``mean_score`` and ``p75_score`` are computed alongside it and the README reports
    all three.

    The plateau is the median of ``statistic`` over buckets from ``plateau_from_hours``
    onward. A bucket counts as settled when it and every later bucket sit within
    ``tolerance_fraction`` of that plateau.
    """
    if months is not None:
        ages = ages[ages["month"].isin(months)]
    if ages.empty:
        raise ValueError("no rows survived the month filter")
    ages = ages.copy()

    edges = _age_bucket_edges(float(ages["age_hours"].max()), bucket_hours)
    ages["bucket"] = pd.cut(ages["age_hours"], bins=edges, right=False, labels=edges[:-1])
    grouped = ages.groupby("bucket", observed=True)["score"]
    curve = pd.DataFrame(
        {
            "age_hours": grouped.size().index.astype(float),
            "n": grouped.size().to_numpy(),
            "median_score": grouped.median().to_numpy(),
            "mean_score": grouped.mean().to_numpy(),
            # The median is pinned to the floor spike, so the readable statistics are
            # the mean and the upper quantiles. All are reported; the README says which
            # one the answer rests on and why.
            "p75_score": grouped.quantile(0.75).to_numpy(),
            "p90_score": grouped.quantile(0.90).to_numpy(),
            "mean_log1p": grouped.apply(lambda s: float(np.log1p(s).mean())).to_numpy(),
        }
    ).reset_index(drop=True)
    curve = curve[curve["n"] >= min_bucket_n].reset_index(drop=True)

    plateau_rows = curve[curve["age_hours"] >= plateau_from_hours]
    series = curve[statistic]
    plateau = float(plateau_rows[statistic].median() if not plateau_rows.empty else series.iloc[-1])
    tolerance = tolerance_fraction * plateau

    within = (series - plateau).abs() <= tolerance
    flat_from = None
    # Walk backwards: the earliest bucket from which every later bucket is inside band.
    settled_suffix = within[::-1].cummin()[::-1]
    if settled_suffix.any():
        flat_from = float(curve.loc[settled_suffix.idxmax(), "age_hours"])

    return SettlingResult(
        curve=curve,
        statistic=statistic,
        plateau_median=plateau,
        flat_from_hours=flat_from,
        tolerance=tolerance,
        months=tuple(sorted(ages["month"].unique())),
    )


#: Age bands for the balanced comparison. The last one is the settled reference.
SETTLING_BANDS: tuple[tuple[float, float], ...] = (
    (0, 12),
    (12, 24),
    (24, 48),
    (48, 72),
    (72, 168),
    (336, float("inf")),
)


def settling_bands(
    ages: pd.DataFrame,
    bands: tuple[tuple[float, float], ...] = SETTLING_BANDS,
    n_boot: int = 400,
    seed: int = 0,
) -> pd.DataFrame:
    """Mean ``log1p(score)`` per age band, balanced across UTC hour of day.

    Two corrections make this readable where the raw curve is not.

    **Hour of day.** All observations share one instant, so a story's age and its UTC
    submission hour are locked together: age modulo 24 *is* the hour of day. Hacker News
    submission volume and quality both swing hard across the day, so a raw age bucket
    measures the hour as much as the age. Each band is averaged over its hour-of-day
    cells with equal weight, which removes the composition difference.

    **The floor spike.** Over half of all stories score 1, so the median is an integer
    that flips between 2 and 3 and cannot show a trend. ``log1p`` keeps every row and is
    not dominated by the tail the way the raw mean is.

    The confidence interval is a 400-sample bootstrap of the balanced mean. A band whose
    interval overlaps the settled band's has not been shown to differ from it.
    """
    frame = ages.copy()
    frame["hod"] = pd.to_datetime(frame["time"]).dt.hour
    frame["y"] = np.log1p(frame["score"].to_numpy(dtype=np.float64))

    rng = np.random.default_rng(seed)
    rows = []
    for low, high in bands:
        band = frame[(frame["age_hours"] >= low) & (frame["age_hours"] < high)]
        if band.empty:
            continue
        balanced = float(band.groupby("hod")["y"].mean().mean())
        draws = [
            float(
                band.sample(len(band), replace=True, random_state=int(rng.integers(1 << 31)))
                .groupby("hod")["y"]
                .mean()
                .mean()
            )
            for _ in range(n_boot)
        ]
        low_ci, high_ci = np.percentile(draws, [2.5, 97.5])
        rows.append(
            {
                "band": f"{low:g}-{high:g}h" if np.isfinite(high) else f">={low / 24:g}d",
                "low_hours": float(low),
                "high_hours": float(high),
                "n": int(len(band)),
                "hours_of_day": int(band["hod"].nunique()),
                "mean_log1p": balanced,
                "ci_low": float(low_ci),
                "ci_high": float(high_ci),
            }
        )
    return pd.DataFrame(rows)


def plot_settling_bands(bands: pd.DataFrame, path: Path = IMAGE_DIR / "gate3-settling-bands.png"):
    """Gate 3, the statistical answer. Balanced mean ``log1p(score)`` per age band.

    The settled band is the reference line. A band whose interval crosses it is
    indistinguishable from settled.
    """
    fig, ax = _figure(width=7.2, height=3.6)
    settled = bands.iloc[-1]
    x = np.arange(len(bands))

    ax.axhspan(settled["ci_low"], settled["ci_high"], color=MUTED, alpha=0.12, linewidth=0)
    ax.axhline(settled["mean_log1p"], color=MUTED, linewidth=1, linestyle="--")

    # Only the *leading run* of below-reference bands is marked as still rising. A later
    # band that dips below is cohort variation, not a score climbing back down, and
    # colouring it as unsettled would be a claim the data does not support.
    below = (bands["ci_high"] < settled["ci_low"]).to_numpy()
    unsettled = np.logical_and.accumulate(below)
    for i, row in bands.iterrows():
        colour = SERIES[1] if unsettled[i] else SERIES[0]
        ax.errorbar(
            x[i],
            row["mean_log1p"],
            yerr=[[row["mean_log1p"] - row["ci_low"]], [row["ci_high"] - row["mean_log1p"]]],
            fmt="o",
            markersize=7,
            color=colour,
            elinewidth=2,
            capsize=4,
        )
        ax.annotate(
            f"n={row['n']:,}",
            (x[i], row["ci_high"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
            color=MUTED,
        )

    if unsettled.any():
        last = int(np.flatnonzero(unsettled)[-1])
        ax.annotate(
            "below the settled\nreference: still rising",
            (last, bands["mean_log1p"].iloc[last]),
            textcoords="offset points",
            xytext=(16, 0),
            ha="left",
            va="center",
            color=SERIES[1],
            fontsize=9,
            fontweight="bold",
        )
    ax.annotate(
        "settled reference",
        (len(bands) - 1.35, settled["mean_log1p"]),
        textcoords="offset points",
        xytext=(0, 7),
        ha="right",
        color=MUTED,
        fontsize=9,
    )
    ax.set_ylim(bands["ci_low"].min() - 0.08, bands["ci_high"].max() + 0.09)
    ax.set_xticks(x)
    ax.set_xticklabels(bands["band"])
    ax.set_title(
        "Gate 3: mean log1p(score) by age at observation, balanced across hour of day",
        fontsize=10.5,
        loc="left",
    )
    ax.set_xlabel("age when the score was read")
    ax.set_ylabel("mean log1p(score)")
    return _save(fig, path)


def plot_settling_curve(
    result: SettlingResult, path: Path = IMAGE_DIR / "gate3-settling-curve.png"
):
    """Gate 3. Score against age at observation, log x so the first day is legible.

    Three statistics on one linear score axis, not a dual axis. The median is the one
    the brief asks for and the one that carries no information here, so plotting it
    beside the mean and the 75th percentile is the argument, not decoration.
    """
    fig, ax = _figure(width=7.4, height=4.0)
    curve = result.curve
    tracks = [
        ("mean_score", SERIES[0], "mean"),
        ("p75_score", SERIES[2], "75th pct"),
        ("median_score", SERIES[1], "median"),
    ]
    for column, colour, label in tracks:
        ax.plot(curve["age_hours"], curve[column], color=colour, linewidth=2)
        # Direct labels: one hue sits below 3:1 contrast on this surface, and the
        # identity of three overlapping lines should not rest on colour alone.
        ax.annotate(
            label,
            (curve["age_hours"].iloc[-1], curve[column].iloc[-1]),
            textcoords="offset points",
            xytext=(7, -3),
            color=colour,
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xscale("log")
    ticks = [1, 3, 6, 12, 24, 48, 72, 168, 336, 720]
    labels = ["1h", "3h", "6h", "12h", "24h", "48h", "72h", "7d", "14d", "30d"]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(curve["age_hours"].min(), curve["age_hours"].max() * 1.35)
    ax.set_title(
        f"Gate 3: score by age at observation ({int(curve['n'].sum()):,} stories, live read)",
        fontsize=11,
        loc="left",
    )
    ax.set_xlabel("age when the score was read")
    ax.set_ylabel("score")
    ax.set_ylim(bottom=0)
    return _save(fig, path)


# --------------------------------------------------------------------------------------
# The score distribution, and the floor spike the plan asserts
# --------------------------------------------------------------------------------------


def score_floor_share(frame: pd.DataFrame) -> pd.DataFrame:
    """Share of stories at each of the first few raw scores.

    The plan asserts that a large share of posts score 1 or 2 and that the floor spike
    survives the log transform. This measures it. The answer decides whether percentile
    rank is worth reporting as a secondary readout.
    """
    scores = frame["score"].to_numpy()
    n = len(scores)
    rows = []
    for value in (1, 2, 3, 4, 5):
        count = int((scores == value).sum())
        rows.append(
            {
                "score": value,
                "n": count,
                "share": count / n,
                "log1p_score": float(np.log1p(value)),
            }
        )
    table = pd.DataFrame(rows)
    table["cumulative_share"] = table["share"].cumsum()
    return table


def plot_score_distribution(frame: pd.DataFrame, path: Path = IMAGE_DIR / "score-distribution.png"):
    """Raw and ``log1p`` score distributions side by side.

    Two panels rather than one dual-axis chart. The point is the floor spike at 1 and 2
    points and whether it survives the log transform, so both panels share the story and
    neither shares an axis.
    """
    import matplotlib.pyplot as plt

    scores = frame["score"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), dpi=110)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)

    # The x axis is windowed, not clipped. Clipping piles every high score onto the last
    # bin and invents a spike that is not in the data.
    # One bin past the window, so the last visible bin is not the closed right edge
    # catching everything at exactly 50.
    axes[0].hist(scores, bins=np.arange(0, 53, 1), color=SERIES[0], linewidth=0)
    axes[0].set_yscale("log")
    axes[0].set_xlim(0, 50)
    axes[0].set_title("Raw score, first 50 points", fontsize=10, loc="left")
    axes[0].set_xlabel(f"score (tail continues to {int(scores.max()):,}, not shown)")
    axes[0].set_ylabel("stories (log scale)")

    axes[1].hist(np.log1p(scores), bins=90, color=SERIES[1], linewidth=0)
    axes[1].set_yscale("log")
    axes[1].set_title("log1p(score), full range", fontsize=10, loc="left")
    axes[1].set_xlabel("log1p(score)")
    axes[1].set_ylabel("stories (log scale)")

    share = float((scores <= 2).mean())
    axes[1].annotate(
        f"{share:.1%} of stories score 1 or 2.\nThe floor spike survives the transform.",
        xy=(0.97, 0.94),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=9,
        color=INK,
    )
    return _save(fig, path)
