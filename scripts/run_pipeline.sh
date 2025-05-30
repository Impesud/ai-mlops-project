#!/bin/bash

MODE=$1

if [[ "$MODE" != "dev" && "$MODE" != "prod" ]]; then
  echo "❌ Usage: ./run_pipeline.sh [dev|prod]"
  exit 1
fi

echo "🚀 Switching to $MODE mode..."

# Funzione per aggiornare il campo "mode" in un file YAML
update_mode_field() {
  local file=$1
  local tmp="${file}.tmp"
  if [[ -f "$file" ]]; then
    sed "s/^mode:.*/mode: $MODE/" "$file" > "$tmp" && mv "$tmp" "$file"
    echo "✅ Updated mode in $file to '$MODE'"
  else
    echo "⚠️ File not found: $file"
  fi
}

# Aggiorna entrambi i config.yaml
update_mode_field "data_ingestion/config.yaml"
update_mode_field "models/config.yaml"

# Avvia la pipeline
echo "🚀 Launching pipeline..."
python3 scripts/pipeline.py

