# hybrid-uav-grid — local dev and Docker shortcuts
#
# Override tools if needed: make run PYTHON=python3.11

.DEFAULT_GOAL := help

PYTHON        ?= python3
PIP           ?= $(PYTHON) -m pip
COMPOSE       ?= docker compose
IMAGE          ?= multi-uav-grid:latest
IMAGE_GPU      ?= multi-uav-grid:gpu
DOCKERFILE     ?= Dockerfile
DOCKERFILE_GPU ?= Dockerfile.gpu
TRAIN_EP      ?= 5000
TEST_EP       ?= 1000

.PHONY: help install install-editable run run-smoke run-full docker-build docker-build-gpu docker-up docker-up-gpu docker-run docker-run-smoke docker-run-gpu-smoke clean

help: ## Show available targets
	@echo "hybrid-uav-grid"
	@echo ""
	@grep -E '^[-a-zA-Z0-9_]+:.*?##' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install dependencies from requirements.txt (uses your default torch)
	$(PIP) install -r requirements.txt

install-editable: ## Editable install of the package (pip install -e .)
	$(PIP) install -e .

run: ## Train + eval locally (default episode counts from CLI)
	$(PYTHON) -m multi_uav_grid

run-smoke: ## Short local run for quick verification
	$(PYTHON) -m multi_uav_grid --train-episodes 20 --test-episodes 10 --log-every 5 --log-level INFO

run-full: ## Local run with TRAIN_EP and TEST_EP (default 5000 / 1000); e.g. make run-full TRAIN_EP=200
	$(PYTHON) -m multi_uav_grid --train-episodes $(TRAIN_EP) --test-episodes $(TEST_EP) --log-every 100 --log-level INFO

docker-build: ## Build the CPU Docker image (default Dockerfile)
	docker build -f $(DOCKERFILE) -t $(IMAGE) .

docker-build-gpu: ## Build the CUDA Docker image (Dockerfile.gpu)
	docker build -f $(DOCKERFILE_GPU) -t $(IMAGE_GPU) .

docker-up: ## Build and run CPU service via docker compose
	$(COMPOSE) up --build multi_uav_grid

docker-up-gpu: ## Build and run GPU service (needs nvidia-container-toolkit)
	$(COMPOSE) --profile gpu up --build multi_uav_grid_gpu

docker-run: ## Run the CPU image with default CMD
	docker run --rm $(IMAGE)

docker-run-smoke: ## Short job in the CPU container
	docker run --rm $(IMAGE) $(PYTHON) -m multi_uav_grid --train-episodes 20 --test-episodes 10 --log-every 5

docker-run-gpu-smoke: ## Short job with GPU (--gpus all)
	docker run --rm --gpus all $(IMAGE_GPU) $(PYTHON) -m multi_uav_grid --device cuda --train-episodes 20 --test-episodes 10 --log-every 5

clean: ## Remove local Python caches and egg-info
	rm -rf build dist *.egg-info .eggs
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
