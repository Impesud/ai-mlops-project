#!/bin/bash

MODE=$1

if [[ "$MODE" != "dev" && "$MODE" != "prod" ]]; then
  echo "❌ Usage: ./run_pipeline.sh [dev|prod]"
  exit 1
fi

echo "🚀 Switching to $MODE mode..."

# Update the "mode" field in config.yaml
CONFIG_FILE="data_ingestion/config.yaml"
TMP_FILE="${CONFIG_FILE}.tmp"

# Replace the mode: line with the correct one
sed "s/^mode:.*/mode: $MODE/" "$CONFIG_FILE" > "$TMP_FILE" && mv "$TMP_FILE" "$CONFIG_FILE"

echo "✅ Updated mode in config.yaml to '$MODE'"

# Run the pipeline
echo "🚀 Launching pipeline..."
python3 scripts/pipeline.py
