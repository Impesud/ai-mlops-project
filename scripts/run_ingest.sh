#!/bin/bash

ENV=$1

if [[ "$ENV" != "dev" && "$ENV" != "prod" ]]; then
  echo "❌ Usage: ./scripts/run_ingest.sh [dev|prod]"
  exit 1
fi

echo "🚀 Running ingestion for '$ENV' environment..."
python3 -m data_ingestion.ingest_spark --env $ENV

