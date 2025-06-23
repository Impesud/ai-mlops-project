# AI MLOps Project Makefile

.PHONY: help pipeline-dev pipeline-prod ingest-dev ingest-prod train-dev train-prod full-dev full-prod mlflow sync-s3 docker-build docker-push clean

help:
	@echo ""
	@echo "Available commands:"
	@echo "  make pipeline-dev     # Run full pipeline (Python orchestrator) in DEV"
	@echo "  make pipeline-prod    # Run full pipeline (Python orchestrator) in PROD"
	@echo "  make train-dev        # Train model (sklearn) on DEV"
	@echo "  make train-prod       # Train model (Spark) on PROD"
	@echo "  make ingest-dev       # Run ingestion only on DEV"
	@echo "  make ingest-prod      # Run ingestion only on PROD"
	@echo "  make full-dev         # Full pipeline via bash (legacy) DEV"
	@echo "  make full-prod        # Full pipeline via bash (legacy) PROD"
	@echo "  make mlflow           # Start MLflow UI"
	@echo "  make sync-s3          # Sync data to S3 manually"
	@echo "  make docker-build     # Build Docker image"
	@echo "  make docker-push      # Push Docker image to registry"
	@echo "  make clean            # Clean temporary files (mlruns, logs, artifacts)"
	@echo ""

# ----------------------------------------------------------
# Full orchestrator pipeline (Python)

pipeline-dev:
	python scripts/pipeline.py --env dev

pipeline-prod:
	python scripts/pipeline.py --env prod

# ----------------------------------------------------------
# Direct training

train-dev:
	python models/train.py --env dev

train-prod:
	python models/train.py --env prod

# ----------------------------------------------------------
# Ingestion steps

ingest-dev:
	./scripts/run_ingest.sh dev

ingest-prod:
	./scripts/run_ingest.sh prod

# ----------------------------------------------------------
# Full legacy pipelines via bash (still supported)

full-dev:
	./scripts/run_pipeline.sh dev

full-prod:
	./scripts/run_pipeline.sh prod

# ----------------------------------------------------------
# MLflow UI

mlflow:
	mlflow ui --backend-store-uri ./mlruns --port 5000

# ----------------------------------------------------------
# S3 Sync (manual)

sync-s3:
	bash scripts/sync_s3.sh dev

# ----------------------------------------------------------
# Docker Build & Push

docker-build:
	docker build -t ${DOCKER_USERNAME}/ai-mlops-project:latest .

docker-push:
	echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin
	docker push ${DOCKER_USERNAME}/ai-mlops-project:latest

# ----------------------------------------------------------
# Cleanup

clean:
	rm -rf mlruns logs __pycache__ data/* .pytest_cache

# ----------------------------------------------------------
# Docker Compose Commands

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

shell:
	docker exec -it ai-mlops-platform bash

# ----------------------------------------------------------
# Pipelines inside Docker

pipeline-dev:
	docker exec ai-mlops-platform make internal-pipeline-dev

pipeline-prod:
	docker exec ai-mlops-platform make internal-pipeline-prod

train-dev:
	docker exec ai-mlops-platform make internal-train-dev

train-prod:
	docker exec ai-mlops-platform make internal-train-prod

# ----------------------------------------------------------
# Standalone MLflow UI (external service)

mlflow:
	docker-compose up -d mlflow


