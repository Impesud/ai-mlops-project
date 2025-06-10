#!/bin/bash

MODE=$1
PROMPT=$2
DEV_MODE=$3

if [[ "$MODE" != "dev" && "$MODE" != "prod" ]]; then
  echo "❌ Usage: ./run_generate.sh [dev|prod] \"Your prompt here\" [--dev-mode]"
  exit 1
fi

if [[ -z "$PROMPT" ]]; then
  PROMPT="Data analysis"
fi

echo "🧠 Generating report for '$MODE' mode with prompt: \"$PROMPT\""

# Build base command
CMD="python3 -m generative_ai.generate --prompt \"$PROMPT\" --env $MODE"

# Optional: dev-mode flag
if [[ "$DEV_MODE" == "--dev-mode" ]]; then
  CMD="$CMD --dev-mode"
fi

# Run command
eval $CMD

