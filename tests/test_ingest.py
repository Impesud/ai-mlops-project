# Test for ingest_spark
import os
import pytest
from pathlib import Path

@pytest.mark.parametrize("output_path", ["data/processed"])
def test_parquet_created(output_path):
    assert Path(output_path).exists(), f"❌ Output path '{output_path}' non esiste"
    files = list(Path(output_path).glob("*.parquet"))
    assert len(files) > 0, "❌ Nessun file Parquet trovato dopo l’ingestion"
    
