import logging
from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger("spark_io")


def read_parquet(spark: SparkSession, path: str) -> DataFrame:
    """
    Read parquet file from given path into Spark DataFrame.
    """
    try:
        logger.info(f"Reading Parquet file from: {path}")
        df = spark.read.parquet(path)
        logger.info(f"✅ Loaded dataframe with shape: {df.count()} rows, {len(df.columns)} columns.")
        return df
    except Exception as e:
        logger.error(f"Failed to read parquet file at {path}: {e}")
        raise


def write_parquet(df: DataFrame, path: str, mode: str = "overwrite"):
    """
    Write Spark DataFrame to parquet file.
    """
    try:
        logger.info(f"Writing DataFrame to: {path} with mode={mode}")
        df.write.mode(mode).parquet(path)
        logger.info(f"✅ Successfully wrote data to: {path}")
    except Exception as e:
        logger.error(f"Failed to write parquet file to {path}: {e}")
        raise
