import argparse

from data_processing.features import advanced_feature_engineering, basic_cleaning
from utils.io import load_env_config
from utils.logging_utils import setup_logger
from utils.spark_io import read_parquet, write_parquet
from utils.spark_utils import create_spark_session


def main(env: str):
    logger = setup_logger("process_data", env)
    cfg = load_env_config(env)
    data_cfg = cfg["data"]

    spark = create_spark_session(app_name="AI-MLOps Data Processing")
    logger.info("✨ Spark session created.")

    input_path = data_cfg["local_intermediate_path"]
    output_path = data_cfg["local_processed_path"]

    logger.info(f"📥 Reading intermediate data from: {input_path}")
    df = read_parquet(spark, input_path)

    logger.info("🧹 Performing basic cleaning...")
    df = basic_cleaning(df, handle_outliers=True)

    logger.info("🧠 Applying advanced feature engineering...")
    df = advanced_feature_engineering(df)

    logger.info(f"📤 Saving processed data to: {output_path}")
    write_parquet(df, output_path)

    spark.stop()
    logger.info("🎉 Data processing pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dev", help="Environment to use: dev or prod")
    args = parser.parse_args()
    main(args.env)
