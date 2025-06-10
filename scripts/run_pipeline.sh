#!/bin/bash

set -e  # Stop execution if any command fails

MODE=$1
MLFLOW_UI=$2  # Optional: --mlflow-ui

if [[ "$MODE" != "dev" && "$MODE" != "prod" ]]; then
  echo "❌ Usage: ./scripts/run_pipeline.sh [dev|prod] [--mlflow-ui]"
  echo "Example: ./scripts/run_pipeline.sh dev --mlflow-ui"
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 Starting pipeline in '$MODE' mode..."

# Export mode for subprocesses
export ENV_MODE="$MODE"

# Build python command
CMD=(python3 -m scripts.pipeline --env "$MODE")

if [[ "$MLFLOW_UI" == "--mlflow-ui" ]]; then
  CMD+=("--mlflow-ui")
  echo "📡 MLflow UI will be started at the end of training."
fi

# Run the pipeline with or without --mlflow-ui
"${CMD[@]}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Pipeline completed for '$MODE'."








