# test/test_ingest.py

import os
import pytest
from pathlib import Path
import yaml

# Carica la configurazione
def get_output_path():
    config_path = "data_ingestion/config.yaml"
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    mode = cfg.get("mode", "dev")
    local_output = cfg.get("local_output_path", "data/processed")
    return Path(local_output) / mode

def test_parquet_created():
    output_path = get_output_path()
    assert output_path.exists(), f"❌ Output path '{output_path}' does not exist"
    parquet_files = list(output_path.glob("*.parquet"))
    assert len(parquet_files) > 0, f"❌ No Parquet files found in '{output_path}'"

    
