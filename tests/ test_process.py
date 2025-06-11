import os
import logging
from pathlib import Path
from utils.io import load_env_config
from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_processed_output_path(env="dev"):
    cfg = load_env_config(env)
    return Path(cfg["data"]["local_processed_path"])

def test_parquet_written_to_processed():
    """
    Test that at least one .parquet file was created in the processed output folder.
    """
    env = os.getenv("TEST_ENV", "dev")
    processed_path = get_processed_output_path(env)

    logger.info(f"📁 Checking processed output path: {processed_path}")
    assert processed_path.exists(), f"❌ Folder '{processed_path}' does not exist."

    parquet_files = list(processed_path.glob("*.parquet"))
    assert len(parquet_files) > 0, f"❌ No .parquet files found in '{processed_path}'."
    logger.info(f"✅ Found {len(parquet_files)} .parquet file(s).")

def test_feature_columns_present():
    """
    Test that all expected feature columns are present in the processed parquet files.
    """
    env = os.getenv("TEST_ENV", "dev")
    processed_path = str(get_processed_output_path(env))

    spark = SparkSession.builder.appName("TestFeatures").getOrCreate()
    df = spark.read.parquet(processed_path)

    expected = {
        "hour", "day_of_week", "day_of_month", "week_of_year",
        "month", "event_timestamp", "is_weekend"
    }

    missing = expected - set(df.columns)
    assert not missing, f"❌ Missing feature columns: {missing}"
    logger.info(f"✅ All expected feature columns are present.")
    spark.stop()

def test_feature_values_valid():
    """
    Test that feature columns have valid ranges and no nulls in the processed parquet files.
    """
    env = os.getenv("TEST_ENV", "dev")
    processed_path = str(get_processed_output_path(env))

    spark = SparkSession.builder.appName("TestFeatureValues").getOrCreate()
    df = spark.read.parquet(processed_path)

    checks = {
        "hour": (0, 23),
        "day_of_week": (1, 7),
        "day_of_month": (1, 31),
        "week_of_year": (1, 53),
        "month": (1, 12),
        "is_weekend": (0, 1)
    }

    for col_name, (min_val, max_val) in checks.items():
        null_count = df.filter(df[col_name].isNull()).count()
        assert null_count == 0, f"❌ Nulls in column '{col_name}'"

        invalid_count = df.filter((df[col_name] < min_val) | (df[col_name] > max_val)).count()
        assert invalid_count == 0, f"❌ Invalid values in '{col_name}': must be in [{min_val}, {max_val}]"

        logger.info(f"✅ Column '{col_name}' passed null and range checks.")

    spark.stop()
