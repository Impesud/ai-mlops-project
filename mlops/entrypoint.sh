#!/bin/bash

set -e

# Configurazione
BACKEND_STORE_URI="sqlite:///mlflow.db"
ARTIFACT_ROOT="./mlruns"
PORT=5000
HOST="0.0.0.0"

# Crea cartelle se non esistono
mkdir -p mlruns

# Messaggio prima dell'avvio
echo "==============================="
echo "🚀 MLflow server in partenza..."
echo "📂 Artifact root: $ARTIFACT_ROOT"
echo "🧠 Backend: $BACKEND_STORE_URI"
echo "🌐 Accesso via browser su:"
echo "👉 http://localhost:$PORT"
echo "==============================="

# Avvio del server MLflow
mlflow server \
  --backend-store-uri "$BACKEND_STORE_URI" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --host "$HOST" \
  --port "$PORT"