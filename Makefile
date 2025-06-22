# AI MLOps Project Makefile

.PHONY: help pipeline-dev pipeline-prod ingest-dev ingest-prod train-dev train-prod full-dev full-prod mlflow-ui sync-s3

help:
	@echo ""
	@echo "Available commands:"
	@echo "  make pipeline-dev     # Run full pipeline script (scripts/pipeline.py) in dev mode"
	@echo "  make pipeline-prod    # Run full pipeline script (scripts/pipeline.py) in prod mode"
	@echo "  make ingest-dev       # Run ingestion only (scripts/run_ingest.sh dev)"
	@echo "  make ingest-prod      # Run ingestion only (scripts/run_ingest.sh prod)"
	@echo "  make train-dev        # Run training only (scripts/run_train.sh dev)"
	@echo "  make train-prod       # Run training only (scripts/run_train.sh prod)"
	@echo "  make full-dev         # Run full bash pipeline (scripts/run_pipeline.sh dev)"
	@echo "  make full-prod        # Run full bash pipeline (scripts/run_pipeline.sh prod)"
	@echo "  make mlflow-ui        # Start MLflow UI"
	@echo "  make sync-s3          # Sync data to S3 manually"
	@echo ""

# ----------------------------------------------------------
# Full orchestrator via pipeline.py
pipeline-dev:
	python scripts/pipeline.py --env dev

pipeline-prod:
	python scripts/pipeline.py --env prod

# ----------------------------------------------------------
# Bash pipeline
full-dev:
	./scripts/run_pipeline.sh dev

full-prod:
	./scripts/run_pipeline.sh prod

# ----------------------------------------------------------
# Separate steps
ingest-dev:
	./scripts/run_ingest.sh dev

ingest-prod:
	./scripts/run_ingest.sh prod

train-dev:
	./scripts/run_train.sh dev

train-prod:
	./scripts/run_train.sh prod

# ----------------------------------------------------------
# MLflow UI
mlflow-ui:
	mlflow ui --backend-store-uri ./mlruns --port 5000

# ----------------------------------------------------------
# Manual S3 Sync
sync-s3:
	bash scripts/sync_s3.sh dev
