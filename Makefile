.DEFAULT_GOAL := help
UV ?= uv

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Create the environment on Python 3.12
	$(UV) sync

.PHONY: setup-all
setup-all: ## Environment plus the data, train and serve extras
	$(UV) sync --extra data --extra train --extra serve

.PHONY: test
test: ## Run the test suite
	$(UV) run pytest -v

.PHONY: lint
lint: ## Check formatting and lint rules
	$(UV) run ruff check
	$(UV) run ruff format --check

.PHONY: fmt
fmt: ## Apply formatting and autofixable lint rules
	$(UV) run ruff format
	$(UV) run ruff check --fix

.PHONY: check
check: lint test ## Everything CI runs

.PHONY: ingest
ingest: ## Phase 1. Build the stories table from the Parquet dump
	$(UV) run --extra data python -m hn_upvotes.data.ingest

.PHONY: clean-shards
clean-shards: ## Phase 1. Delete the monthly shards. Safe once ingest reports its count
	rm -rf data/shards

.PHONY: notebook
notebook: ## Phase 1. Re-execute the EDA notebook in place, outputs and all
	$(UV) run --extra data jupyter execute --inplace notebooks/01-eda.ipynb

.PHONY: train-embeddings
train-embeddings: ## Phase 3. Train one word2vec objective
	$(UV) run --extra train python -m hn_upvotes.embeddings.train

.PHONY: serve
serve: ## Phase 6. Run the prediction service locally
	$(UV) run --extra serve uvicorn hn_upvotes.serving.app:app --reload --port 8000

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
