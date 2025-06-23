import os
from pyspark.sql import SparkSession

def create_spark_session(app_name="DataProcessingPipeline") -> SparkSession:
    """
    Create a SparkSession configured for AWS S3 (s3a) and local environments.
    AWS credentials and region are read from environment variables.
    """
    aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
    aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
    aws_region = os.environ.get("AWS_DEFAULT_REGION", "eu-central-1")
    s3_endpoint = f"s3.{aws_region}.amazonaws.com"

    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", aws_access_key)
        .config("spark.hadoop.fs.s3a.secret.key", aws_secret_key)
        .config("spark.hadoop.fs.s3a.endpoint", s3_endpoint)
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .getOrCreate()
    )

    return spark
