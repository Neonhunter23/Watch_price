# ── Makefile (Linux / Docker / WSL) ──────────────────────────
# On Windows, use run.ps1 instead: .\run.ps1 help
.PHONY: setup train evaluate eda explainability docker docker-train clean test lint help

PYTHON := python3
CONFIG ?= configs/base.yaml
DOCKER_IMAGE := watch-price-cnn
DOCKER_TAG := latest

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────

setup: ## Create venv and install dependencies
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"
	.venv/bin/python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
	@echo "\n✅ Setup complete. Activate with: source .venv/bin/activate"

# ── Training & Evaluation ───────────────────────────────────

train: ## Train model (CONFIG=configs/base.yaml)
	$(PYTHON) scripts/train.py --config $(CONFIG)

evaluate: ## Evaluate trained model
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

predict: ## Run predictions on new images
	$(PYTHON) scripts/predict.py --config $(CONFIG)

# ── Notebooks ────────────────────────────────────────────────

eda: ## Launch EDA notebook
	jupyter notebook notebooks/01_eda.ipynb

# ── Explainability ───────────────────────────────────────────

explainability: ## Generate Grad-CAM visualizations
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --gradcam

# ── Docker (GPU) ─────────────────────────────────────────────

docker: ## Build Docker image
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

docker-train: docker ## Train inside Docker with GPU
	docker compose -f docker/docker-compose.yaml run --rm train

docker-shell: docker ## Open shell in Docker container with GPU
	docker compose -f docker/docker-compose.yaml run --rm --entrypoint bash train

# ── Quality ──────────────────────────────────────────────────

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

lint: ## Lint and format code
	$(PYTHON) -m ruff check src/ scripts/ tests/
	$(PYTHON) -m black --check src/ scripts/ tests/

format: ## Auto-format code
	$(PYTHON) -m ruff check --fix src/ scripts/ tests/
	$(PYTHON) -m black src/ scripts/ tests/

# ── Cleanup ──────────────────────────────────────────────────

clean: ## Remove outputs and cache
	rm -rf outputs/checkpoints/* outputs/logs/* outputs/results/*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned outputs and caches"

clean-all: clean ## Remove outputs, caches, and venv
	rm -rf .venv
	@echo "✅ Cleaned everything including venv"
