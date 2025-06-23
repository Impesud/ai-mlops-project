# AI MLOps Project Makefile (Hybrid: Local & Dockerized)

.PHONY: help pipeline-dev pipeline-prod ingest-dev ingest-prod train-dev train-prod full-dev full-prod mlflow sync-s3 docker-build docker-push clean build up down logs shell

export PYTHONPATH := $(shell pwd)

help:
	@echo ""
	@echo "Available commands:"
	@echo "=== Local runs ==="
	@echo "  make pipeline-dev-local    # Full pipeline locally (dev)"
	@echo "  make pipeline-prod-local   # Full pipeline locally (prod)"
	@echo "  make train-dev-local       # Train model locally (sklearn dev)"
	@echo "  make train-prod-local      # Train model locally (Spark prod)"
	@echo "  make ingest-dev-local      # Ingest data locally (dev)"
	@echo "  make ingest-prod-local     # Ingest data locally (prod)"
	@echo ""
	@echo "=== Dockerized runs ==="
	@echo "  make build           # Build docker-compose stack"
	@echo "  make up              # Start full stack"
	@echo "  make down            # Stop stack"
	@echo "  make logs            # Show logs"
	@echo "  make shell           # Enter mlops container"
	@echo "  make pipeline-dev    # Run pipeline inside docker (dev)"
	@echo "  make pipeline-prod   # Run pipeline inside docker (prod)"
	@echo "  make train-dev       # Train inside docker (dev)"
	@echo "  make train-prod      # Train inside docker (prod)"
	@echo ""
	@echo "=== MLflow UI ==="
	@echo "  make mlflow          # Start MLflow UI standalone (docker)"
	@echo ""
	@echo "=== S3 Sync ==="
	@echo "  make sync-s3         # Manual sync of data to S3"
	@echo ""
	@echo "=== Docker image ==="
	@echo "  make docker-build    # Build Docker image standalone"
	@echo "  make docker-push     # Push Docker image to registry"
	@echo ""
	@echo "=== Maintenance ==="
	@echo "  make clean           # Clean local artifacts"
	@echo ""

# ----------------------------------------------------------
# Local (non dockerized) execution - for pure local testing

pipeline-dev-local:
	python scripts/pipeline.py --env dev

pipeline-prod-local:
	python scripts/pipeline.py --env prod

train-dev-local:
	python models/train.py --env dev

train-prod-local:
	python models/train.py --env prod

ingest-dev-local:
	./scripts/run_ingest.sh dev

ingest-prod-local:
	./scripts/run_ingest.sh prod

process-dev-local:
	./scripts/run_process.sh dev

process-prod-local:
	./scripts/run_process.sh prod

full-dev-local:
	./scripts/run_pipeline.sh dev

full-prod-local:
	./scripts/run_pipeline.sh prod

ml-flow-local:
	mlflow ui --backend-store-uri ./mlruns --port 5000

# ----------------------------------------------------------
# Docker Compose orchestration

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
# Dockerized pipeline execution

pipeline-dev:
	docker exec ai-mlops-platform python scripts/pipeline.py --env dev

pipeline-prod:
	docker exec ai-mlops-platform python scripts/pipeline.py --env prod

train-dev:
	docker exec ai-mlops-platform python models/train.py --env dev

train-prod:
	docker exec ai-mlops-platform python models/train.py --env prod

mlflow-docker:
	docker-compose up -d mlflow-ui

# ----------------------------------------------------------
# Manual S3 Sync

sync-s3:
	bash scripts/sync_s3.sh dev

# ----------------------------------------------------------
# Docker image build & push (standalone)

docker-build:
	docker build -t ${DOCKER_USERNAME}/ai-mlops-project:latest .

docker-push:
	echo "${DOCKER_PASSWORD}" | docker login --username "${DOCKER_USERNAME}" --password-stdin
	docker push ${DOCKER_USERNAME}/ai-mlops-project:latest

# ----------------------------------------------------------
# Cleanup

clean:
	rm -rf mlruns logs __pycache__ data/* .pytest_cache
