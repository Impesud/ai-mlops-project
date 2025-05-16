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
    # … aggiungi gli altri campi
])

def load_config(path: str):
    base_dir = os.path.dirname(__file__)
    full_path = os.path.join(base_dir, path)
    with open(full_path) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    config = load_config("config.yaml")

    spark = (
        SparkSession.builder
        .appName("IngestioneBigData")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"])
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{config['aws']['region']}.amazonaws.com")
        .getOrCreate()
    )

    # 2) Lettura streaming con schema
    streaming_df = (
        spark.readStream
            .format(config["format"])
            .option("header", "true")          # riconosce il primo header
            .option("inferSchema", "true")     # inferisce i tipi, poi applica il schema
            .schema(schema)                    # lo schema definito in partenza
            .load(config["path"])
            # filtra righe con event_time o value mancanti
            .filter(col("event_time").isNotNull() & col("value").isNotNull())
    )
    
    # 3) Scrittura streaming
    query = (
        streaming_df.writeStream
        .format("parquet")
        .option("path", config["output_path"])
        .option("checkpointLocation", config["checkpoint_location"])
        .outputMode("append")
        .trigger(once=True) # ← esegue un solo micro-batch e poi si ferma
        .start()
    )

    query.awaitTermination()
    spark.stop()
    print("Ingestione streaming completata")

