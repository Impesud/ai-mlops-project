#!/bin/bash

# === Setup ===
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/sync_s3.log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# === Ensure log directory exists ===
mkdir -p "$LOG_DIR"

log() {
  echo "[$TIMESTAMP] $1" | tee -a "$LOG_FILE"
}

# === Sync configuration ===
declare -A SYNC_PATHS=(
  ["data/raw-data/"]="s3://my-mlops-raw-data/"
  ["data/intermediate/"]="s3://my-mlops-intermediate-data/"
  ["data/processed/"]="s3://my-mlops-processed-data/"
)

log "🚀 Starting S3 sync job..."

for LOCAL_DIR in "${!SYNC_PATHS[@]}"; do
  S3_DEST=${SYNC_PATHS[$LOCAL_DIR]}
  
  if [[ -d "$LOCAL_DIR" ]]; then
    log "📤 Syncing '$LOCAL_DIR' to '$S3_DEST'..."
    aws s3 sync "$LOCAL_DIR" "$S3_DEST" --delete >> "$LOG_FILE" 2>&1
    if [[ $? -eq 0 ]]; then
      log "✅ Successfully synced '$LOCAL_DIR'."
    else
      log "❌ Failed to sync '$LOCAL_DIR'."
    fi
  else
    log "⚠️ Local directory '$LOCAL_DIR' does not exist. Skipping..."
  fi
done

log "🎉 Sync job completed."
