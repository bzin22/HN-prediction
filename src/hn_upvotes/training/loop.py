"""The training loop shared by all three fusion architectures.

Runs one walk-forward fold end to end: build features from the training window, fit the
poolers and encoders on that window only, train, evaluate on the next window. Every
model in the comparison goes through this same loop, so differences in the results table
come from the architecture and not from the harness.

Needs the ``train`` extra.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from hn_upvotes.data.splits import TemporalSplit
from hn_upvotes.training.metrics import MetricReport


@dataclass(frozen=True)
class TrainConfig:
    """Settings for one training run.

    loss
        Squared error is maximum likelihood under Gaussian *residuals*, not a Gaussian
        target. If the Phase 2 residual plot comes out heavy tailed, switch to Huber.
    seeds
        Every result in the README is quoted as mean plus or minus standard deviation
        across seeds. A single run is not evidence and this audience knows it.
    """

    epochs: int = 20
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    loss: Literal["mse", "huber"] = "mse"
    huber_delta: float = 1.0
    early_stopping_patience: int = 3
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class FoldResult:
    """One fold, one seed. The unit the results table aggregates over."""

    split: TemporalSplit
    seed: int
    epochs_run: int
    validation: MetricReport
    test: MetricReport


def build_dataloaders(
    frame: pd.DataFrame,
    targets: np.ndarray,
    split: TemporalSplit,
    config: TrainConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation and test loaders for one fold.

    Every fitted transform (pooler, author encoder, domain encoder) is fitted inside
    this function on the training slice only, then applied to all three. Fitting them
    outside would leak test-period statistics into training.
    """
    raise NotImplementedError


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Run one epoch and return the mean training loss."""
    raise NotImplementedError


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(predictions, actuals)`` in normalised target space."""
    raise NotImplementedError


def fit(
    model_factory: type[nn.Module],
    frame: pd.DataFrame,
    targets: np.ndarray,
    splits: Iterable[TemporalSplit],
    config: TrainConfig,
    output_dir: Path | None = None,
) -> list[FoldResult]:
    """Train and evaluate one architecture across every fold and every seed.

    Returns one :class:`FoldResult` per (fold, seed) pair. The README table aggregates
    these, and reports two models as indistinguishable where they differ by less than
    the seed noise.
    """
    raise NotImplementedError
