# Serving image. Multi-stage, and deliberately much smaller than the training image.
#
# It carries the model artefact, the vocabulary and the categorical mappings. It carries
# no training dependencies, no DuckDB, no gensim, no scikit-learn, and no corpus. The
# model runs forward passes only, so the whole training stack is dead weight in a
# container that has to start quickly.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Built into a venv so the runtime stage can copy one self-contained directory.
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[serve]"


FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Model artefact, vocabulary and categorical mappings. Built by the training image and
# copied in at build time, so the served version is pinned to an image tag.
COPY artifacts/serving/ /app/artifacts/

RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "hn_upvotes.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
