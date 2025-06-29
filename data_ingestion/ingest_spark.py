# data_ingestion/ingest_spark.py
import argparse
import os

from pyspark.sql import SparkSession

from utils.io import load_env_config
from utils.logging_utils import setup_logger


def create_spark_session():
    return (
        SparkSession.builder.appName("Batch Ingestion Job")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"])
        .config(
            "spark.hadoop.fs.s3a.endpoint",
            f"s3.{os.environ['AWS_DEFAULT_REGION']}.amazonaws.com",
        )
        .getOrCreate()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", help="Environment: dev or prod")
    args = parser.parse_args()
    logger = setup_logger("ingest_spark", args.env)

    spark = None

    try:
        cfg = load_env_config(args.env)
        data_cfg = cfg.get("data", {})

        if "local_input_path" not in data_cfg:
            raise ValueError("Missing data config key: local_input_path")

        logger.info(f"🔧 Environment: {args.env}")
        logger.info(f"📁 Input path: {data_cfg['local_input_path']}")

        spark = create_spark_session()
        logger.info("✅ Spark session started.")

        df = (
            spark.read.format(data_cfg.get("format", "csv"))
            .option("header", "true")
            .option("inferSchema", "true")
            .option("sep", ",")
            .load(data_cfg["local_input_path"])
        )

        logger.info(f"📊 Ingested {df.count()} rows")
        df.write.mode("overwrite").parquet(f"data/intermediate/{args.env}")
        logger.info(f"💾 Saved intermediate data to data/intermediate/{args.env}")

    except Exception as e:
        logger.exception(f"❌ Ingestion failed: {e}")
        raise

    finally:
        if spark is not None:
            spark.stop()
            logger.info("🛑 Spark session stopped.")
