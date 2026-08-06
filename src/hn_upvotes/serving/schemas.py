"""Request and response bodies for the prediction service.

The request carries exactly the four submission-time fields on the allowlist in
``features.schema``. There is deliberately no field for score, comment count or
anything else post-hoc: the service could not use it, and accepting it would invite a
caller to think it mattered.

Needs the ``serve`` extra.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """A post as it would look at the moment of submission."""

    title: str = Field(min_length=1, description="The submitted headline.")
    author: str = Field(min_length=1, description="The submitting account, the `by` field.")
    url: str | None = Field(
        default=None,
        description="The linked address. Null for a text post, which is a valid input.",
    )
    timestamp: datetime = Field(description="Submission time. Assumed UTC if naive.")


class PredictResponse(BaseModel):
    """A prediction, given in both spaces.

    ``normalised_target`` is what the model actually produces: a position relative to
    posts from around the same time. ``predicted_score`` is that value mapped back
    through the trailing baseline, which is the number a human wants.

    ``baseline_centre`` and ``baseline_spread`` are returned because the mapping is only
    meaningful with them, and a caller reproducing the number needs both.
    """

    predicted_score: float
    normalised_target: float
    baseline_centre: float
    baseline_spread: float
    baseline_window_end: datetime
    model_version: str


class HealthResponse(BaseModel):
    """Liveness plus enough detail to tell which artefact is loaded."""

    status: str
    model_version: str
    embedding_variant: str
    baseline_window_end: datetime | None
