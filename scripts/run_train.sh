#!/bin/bash

MODE=$1
UI_FLAG=$2  # opzionale: --mlflow-ui

if [[ "$MODE" != "dev" && "$MODE" != "prod" ]]; then
  echo "❌ Usage: ./scripts/run_train.sh [dev|prod] [--mlflow-ui]"
  echo "Example: ./scripts/run_train.sh dev --mlflow-ui"
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🧠 Starting model training in '$MODE' mode..."

# Export environment variable
export ENV_MODE=$MODE

# Determine the module path
if [[ "$MODE" == "dev" ]]; then
  echo "🧪 Using Scikit-learn training pipeline..."
  python3 -m models.train_sklearn --env "$MODE" $UI_FLAG
else
  echo "⚙️ Using Spark MLlib training pipeline..."
  python3 -m models.train_spark --env "$MODE" $UI_FLAG
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Training completed for '$MODE'."


