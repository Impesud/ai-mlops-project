# Data processing script
import pandas as pd

def process(data_path: str, output_path: str):
    df = pd.read_parquet(data_path)
    # Add processing steps here
    df.to_parquet(output_path)

if __name__ == "__main__":
    process('data_ingestion/output.parquet', 'data/processed/data.parquet')