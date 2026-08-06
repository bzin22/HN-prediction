"""FastAPI prediction service.

Two routes: ``POST /predict`` and ``GET /health``. Phase 6 implements them.

The service holds the trained model, the vocabulary, the categorical mappings, and the
most recent trailing baseline. That last one is the part that is easy to get wrong: the
baseline has to be refreshed as time passes, or predictions drift out of date in the
same way a model trained on 2011 would. It is a served artefact, not a constant.

Needs the ``serve`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI

from hn_upvotes.serving.schemas import HealthResponse, PredictRequest, PredictResponse

app = FastAPI(
    title="HN upvote prediction",
    description="Predicts a Hacker News score from submission-time information only.",
    version="0.1.0",
)


@dataclass
class ServiceState:
    """Everything loaded at startup and reused across requests.

    Kept in one object so a reload is atomic: swap the whole state or none of it. A
    half-updated service that pairs a new model with an old vocabulary would produce
    plausible nonsense rather than an error.
    """

    model_version: str
    embedding_variant: str
    artefact_dir: Path


def load_state(artefact_dir: Path) -> ServiceState:
    """Load model, vocabulary, encoders and the current trailing baseline.

    Called once at startup. Raises rather than serving a partially loaded state.
    """
    raise NotImplementedError


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Predict a score for a post that has not been submitted yet.

    Builds the same features the training pipeline builds, runs the model, then maps the
    normalised output back to a raw score through the current trailing baseline.
    """
    raise NotImplementedError


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check, and which artefact is loaded."""
    raise NotImplementedError
