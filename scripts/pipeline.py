import os
import sys
import subprocess
import boto3
import argparse
from utils.io import load_env_config
from utils.logging_utils import setup_logger
from datetime import datetime

def ensure_bucket(s3_client, bucket_name, region, logger):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket '{bucket_name}' already exists.")
    except s3_client.exceptions.NoSuchBucket:
        logger.info(f"Creating bucket '{bucket_name}' in region '{region}'.")
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": region}
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", help="Environment: dev or prod")
    parser.add_argument("--mlflow-ui", action="store_true", help="Start MLflow UI after training")
    args = parser.parse_args()

    logger = setup_logger("pipeline", args.env)

    # Load configuration
    cfg = load_env_config(args.env)
    data_cfg = cfg["data"]
    aws_cfg = cfg["aws"]
    cloud_cfg = cfg["cloud"]

    region = aws_cfg["region"]
    access_key = aws_cfg["access_key_id"]
    secret_key = aws_cfg["secret_access_key"]

    # Set AWS credentials in environment
    os.environ["AWS_ACCESS_KEY_ID"] = access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
    os.environ["AWS_REGION"] = region

    input_csv = data_cfg["local_input_path"]
    raw_bucket = cloud_cfg["s3_bucket_input"].replace("s3a://", "").rstrip("/")
    proc_bucket = cloud_cfg["s3_bucket_output"].replace("s3a://", "").rstrip("/")

    # Check if local input exists
    if not os.path.exists(input_csv):
        logger.error(f"CSV input not found at: {input_csv}")
        sys.exit(1)
    else:
        logger.info(f"Found input CSV: {input_csv}")

    # Initialize boto3 session
    session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"), region_name=region)
    s3 = session.client("s3")

    # Ensure buckets exist
    ensure_bucket(s3, raw_bucket, region, logger)
    ensure_bucket(s3, proc_bucket, region, logger)

    # Start ingestion
    env_vars = os.environ.copy()
    env_vars["PYSPARK_PYTHON"] = sys.executable
    env_vars["PYSPARK_DRIVER_PYTHON"] = sys.executable

    logger.info("🚚 Starting Spark ingestion job...")
    try:
        subprocess.run([
            sys.executable, "-m", "data_ingestion.ingest_spark", "--env", args.env
        ], check=True, env=env_vars)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Spark ingestion failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    logger.info("🧹 Starting data processing job...")
    try:
        subprocess.run([
            sys.executable, "-m", "data_processing.process", "--env", args.env
        ], check=True, env=env_vars)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Data processing failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    logger.info("🧠 Starting model training job...")
    try:
        command = [sys.executable, "-m", "models.train", "--env", args.env]
        if args.mlflow_ui:
            command.append("--mlflow-ui")
        subprocess.run(command, check=True, env=env_vars)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    logger.info("🧾 Generating AI report...")
    try:
        prompt = os.environ.get("AI_PROMPT", "Data pipeline report")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"docs/reports/report_{args.env}_pipeline_{timestamp}.txt"
        subprocess.run([
            sys.executable, "-m", "generative_ai.generate", "--env", args.env, "--prompt", prompt, "--output", report_file
        ], check=True, env=env_vars)
        logger.info(f"✅ Report saved to {report_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Report generation failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    logger.info("☁️ Syncing local folders to S3...")
    try:
        subprocess.run(["bash", "scripts/sync_s3.sh", args.env], check=True, env=env_vars)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ S3 sync failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    logger.info("🎉 Pipeline completed successfully.")







