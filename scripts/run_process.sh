#!/bin/bash

ENV=$1

if [[ "$ENV" != "dev" && "$ENV" != "prod" ]]; then
  echo "❌ Usage: ./scripts/run_process.sh [dev|prod]"
  exit 1
fi

echo "🧠 Running processing for '$ENV' environment..."
python3 -m data_processing.process --env $ENV
