# Training image. Carries the data and train extras, no corpora and no model artefacts.
# Corpora are mounted at run time so the image does not carry gigabytes that change
# every experiment.
#
# Note this image runs on CPU or CUDA. The MPS backend needs the host's Metal stack and
# is not available inside Docker, so local M2 training runs outside a container and this
# image exists for reproducibility and for any non-Apple machine.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a source edit does not reinstall the whole environment. The
# src/ copy is deferred, so this layer caches on pyproject.toml alone.
COPY pyproject.toml README.md ./
RUN mkdir -p src/hn_upvotes && touch src/hn_upvotes/__init__.py \
    && pip install --no-cache-dir -e ".[data,train]"

COPY src/ src/
COPY configs/ configs/

ENTRYPOINT ["python", "-m"]
CMD ["hn_upvotes.training.loop"]
