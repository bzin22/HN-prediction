# Training image. Carries the data and train extras, no corpora and no model artefacts.
# Corpora are mounted at run time so the image does not carry gigabytes that change
# every experiment.
#
# Note this image runs on CPU or CUDA. The MPS backend needs the host's Metal stack and
# is not available inside Docker, so local M2 training runs outside a container and this
# image exists for reproducibility and for any non-Apple machine.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so a source edit does not re-resolve the environment.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra data --extra train

COPY src/ src/
COPY configs/ configs/
RUN uv sync --frozen --extra data --extra train

ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT ["python", "-m"]
CMD ["hn_upvotes.training.loop"]
