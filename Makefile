.DEFAULT_GOAL := help

# Plain venv and pip. PYTHON is the interpreter used to create .venv; override it to
# build against a different version, for example `make setup PYTHON=python3.12`.
PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin
PIP := $(BIN)/python -m pip

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

.PHONY: setup
setup: $(BIN)/python ## Create .venv and install the project plus dev tools
	$(PIP) install -e ".[dev]"

.PHONY: setup-all
setup-all: $(BIN)/python ## Everything, including the data, train and serve extras
	$(PIP) install -e ".[dev,data,train,serve]"

.PHONY: test
test: ## Run the test suite
	$(BIN)/pytest -v

.PHONY: lint
lint: ## Check formatting and lint rules
	$(BIN)/ruff check
	$(BIN)/ruff format --check

.PHONY: fmt
fmt: ## Apply formatting and autofixable lint rules
	$(BIN)/ruff format
	$(BIN)/ruff check --fix

.PHONY: check
check: lint test ## Everything CI runs

.PHONY: ingest
ingest: ## Phase 1. Build the stories table from the Parquet dump
	$(BIN)/python -m hn_upvotes.data.ingest

.PHONY: clean-shards
clean-shards: ## Phase 1. Delete the monthly shards. Safe once ingest reports its count
	rm -rf data/shards

# Rung 5 needs an OpenMP runtime, which macOS does not ship. The official answer is
# `brew install libomp`. scikit-learn's wheel already carries a copy inside the venv, so
# this points at that one and the target works with no system package. Empty and ignored
# on Linux, where the xgboost wheel bundles its own runtime.
OMP_DIR := $(shell ls -d $(VENV)/lib/python*/site-packages/sklearn/.dylibs 2>/dev/null | head -1)

.PHONY: baselines
baselines: ## Phase 2. Fit the five baseline rungs and print the results table
	DYLD_LIBRARY_PATH="$(OMP_DIR)" $(BIN)/python -m hn_upvotes.training.run_baselines

# Hours, not minutes: the tree count is whatever early stopping picks. Same OpenMP
# workaround as `make baselines`.
.PHONY: tune-xgboost
tune-xgboost: ## Phase 2. Early-stop rung 6 against a held-back year, and score it
	DYLD_LIBRARY_PATH="$(OMP_DIR)" $(BIN)/python -m hn_upvotes.training.tune_xgboost

.PHONY: notebook
notebook: ## Phase 1. Re-execute the EDA notebook in place, outputs and all
	$(BIN)/jupyter execute --inplace notebooks/01-eda.ipynb

.PHONY: train-embeddings
train-embeddings: ## Phase 3. Train one word2vec objective over one corpus
	$(BIN)/python -m hn_upvotes.embeddings.train

.PHONY: throughput
throughput: ## Phase 3. Measure sparse gradients and tokens/s on CPU against MPS
	$(BIN)/python -m hn_upvotes.embeddings.throughput

# Both download text8 into data/corpora on first use, about 31 MB zipped.
.PHONY: lr-sweep
lr-sweep: ## Phase 3. Sweep the learning rate on text8 and pick one, about 25 minutes
	$(BIN)/python -m hn_upvotes.embeddings.learning_rate_sweep

.PHONY: batch-scaling
batch-scaling: ## Phase 3. Check the batch-size rate multiplication against batch 1, minutes
	$(BIN)/python -m hn_upvotes.embeddings.batch_scaling

.PHONY: hn-token-count
hn-token-count: ## Phase 3. Count the Hacker News corpus for runtime estimates
	$(BIN)/python -m hn_upvotes.embeddings.corpora

.PHONY: chain-dry-run
chain-dry-run: ## Phase 3. Walk every stage of both objectives on synthetic corpora, seconds
	$(BIN)/python -m hn_upvotes.embeddings.chain --dry-run \
		--output-directory artifacts/embeddings-dry-run

# The full training run, without time limits. Both objectives, four stages each, six variants. Detached, so it
# survives the terminal closing, and it writes artifacts/embeddings/run-manifest.json as it
# goes. Read that file in the morning, not the log. Add --resume to carry on from the newest
# checkpoint of each stage, which does not restart an objective that already finished.
.PHONY: chain
chain: ## Phase 3. Launch the training chain for both objectives, detached
	$(BIN)/python -m hn_upvotes.embeddings.chain --detach

.PHONY: serve
serve: ## Phase 6. Run the prediction service locally
	$(BIN)/uvicorn hn_upvotes.serving.app:app --reload --port 8000

.PHONY: docker-train
docker-train: ## Build the training image
	docker build -f docker/train.Dockerfile -t hn-upvotes-train .

.PHONY: docker-serve
docker-serve: ## Build the serving image
	docker build -f docker/serve.Dockerfile -t hn-upvotes-serve .

.PHONY: clean
clean: ## Remove caches and build output
	rm -rf .pytest_cache .ruff_cache build dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: clean-venv
clean-venv: ## Remove the virtual environment
	rm -rf $(VENV)
