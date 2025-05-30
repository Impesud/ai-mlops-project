from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
import yaml, os

# 1) Definizione manuale dello schema
schema = StructType([
    StructField("user_id", StringType(), True),
    StructField("event_time", TimestampType(), True),
    StructField("action", StringType(), True),
    StructField("value", DoubleType(), True),
])

def load_config(path: str):
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, path)
    with open(full_path) as f:
        return yaml.safe_load(f)
    
os.environ["fs.s3a.connection.timeout"] = os.getenv("FS_S3A_CONNECTION_TIMEOUT", "60000")

if __name__ == "__main__":
    config = load_config("config.yaml")
    mode = config.get("mode", "dev")
    data_path = config[mode]["path"]
    output_path = config[mode]["output_path"]
    local_output_path = config[mode]["local_output_path"]
     
    print(f"Mode attivo: {mode}")
    print(f"File di input: {data_path}")

    spark = (
        SparkSession.builder
        .appName("ModelTraining")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", os.environ['AWS_ACCESS_KEY_ID'])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ['AWS_SECRET_ACCESS_KEY'])
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{os.environ['AWS_REGION']}.amazonaws.com")
        .getOrCreate()
    )
    
    # 2) Batch ingestion
    print("Avvio batch ingestion Spark...")
    df = (
        spark.read
            .format(config["format"])
            .option("header", "true")
            .option("inferSchema", "true")
            .option("sep", ",") 
            .schema(schema)
            .load(data_path)
    )
    cleaned = df.filter(col("event_time").isNotNull() & col("value").isNotNull())

    # 3) Scrittura batch
    print(f"Scrivo Parquet in {output_path}")
    cleaned.write.mode("overwrite").parquet(output_path)
    # Scrittura anche in locale (local_output_path)
    print(f"Scrivo anche Parquet localmente in {local_output_path}")
    cleaned.write.mode("overwrite").parquet(local_output_path)

    spark.stop()
    print("Ingestione batch completata")


