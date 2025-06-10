# data_processing/process.py

import os
import sys
import argparse
from pyspark.sql import SparkSession
from utils.io import load_env_config
from utils.logging_utils import setup_logger
from data_processing.features import basic_cleaning, advanced_feature_engineering


def create_spark_session():
    return (
        SparkSession.builder
        .appName("DataProcessingPipeline")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", os.environ.get("AWS_ACCESS_KEY_ID", ""))
        .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("AWS_SECRET_ACCESS_KEY", ""))
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{os.environ.get('AWS_REGION', 'us-east-1')}.amazonaws.com")
        .getOrCreate()
    )


def main(env: str):
    logger = setup_logger("process_data", env)
    cfg = load_env_config(env)
    data_cfg = cfg["data"]

    spark = create_spark_session()
    logger.info("✨ Spark session created.")

    input_path = data_cfg["local_intermediate_path"]
    output_path = data_cfg["local_processed_path"]

    logger.info(f"✅ Reading intermediate data from: {input_path}")
    df = spark.read.parquet(input_path)

    logger.info("🧹 Performing basic cleaning...")
    df = basic_cleaning(df, handle_outliers=True)

    logger.info("🧠 Applying advanced feature engineering...")
    df = advanced_feature_engineering(df)

    logger.info(f"✅ Saving processed data to: {output_path}")
    df.write.mode("overwrite").parquet(output_path)

    spark.stop()
    logger.info("🎉 Data processing pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dev", help="Environment to use: dev or prod")
    args = parser.parse_args()
    main(args.env)


