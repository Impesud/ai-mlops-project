from pyspark.sql.functions import (
    col, hour, dayofweek, dayofmonth, weekofyear, month,
    unix_timestamp, when, trim, lower as spark_lower, current_timestamp,
    count, sum as spark_sum, max as spark_max, to_date
)
from pyspark.sql.window import Window as SparkWindow
from pyspark.sql.types import StringType, TimestampType, DoubleType, DataType
from pyspark.sql import DataFrame, Window
from typing import Dict
import logging

# Required columns with their expected types
REQUIRED_COLUMNS: Dict[str, DataType] = {
    "user_id": StringType(),
    "event_time": TimestampType(),
    "action": StringType(),
    "value": DoubleType()
}

logger = logging.getLogger("features")

def validate_schema(df: DataFrame) -> DataFrame:
    logger.info("🔍 Validating and casting schema if needed...")
    for col_name, expected_type in REQUIRED_COLUMNS.items():
        if col_name not in df.columns:
            logger.error(f"❌ Missing required column: {col_name}")
            raise ValueError(f"Missing required column: {col_name}")
        actual_type = df.schema[col_name].dataType
        if type(actual_type) != type(expected_type):
            logger.warning(f"⚠️ Column '{col_name}' has type {actual_type}, casting to {expected_type}")
            df = df.withColumn(col_name, col(col_name).cast(expected_type))
    logger.info("✅ Schema validated and casted where needed.")
    return df

def basic_cleaning(df: DataFrame, handle_outliers: bool = False) -> DataFrame:
    logger.info("🧹 Performing basic cleaning...")
    df = validate_schema(df)
    df = df.dropna(subset=["user_id", "event_time", "action", "value"])
    df = df.withColumn("user_id", trim(spark_lower(col("user_id"))))
    df = df.withColumn("action", trim(spark_lower(col("action"))))
    df = df.fillna({"value": 0.0, "action": "unknown"})
    df = df.filter(col("value") >= 0)

    if handle_outliers:
        q1 = df.approxQuantile("value", [0.25], 0.05)[0]
        q3 = df.approxQuantile("value", [0.75], 0.05)[0]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        logger.info(f"📊 Outlier thresholds: lower={lower_bound:.2f}, upper={upper_bound:.2f}")
        df = df.filter((col("value") >= lower_bound) & (col("value") <= upper_bound))

    df = df.dropDuplicates(["user_id", "event_time", "action"])
    df = df.filter(col("event_time") <= current_timestamp())
    logger.info("✅ Basic cleaning completed.")
    return df

def advanced_feature_engineering(df: DataFrame) -> DataFrame:
    logger.info("🧠 Starting advanced feature engineering...")

    # Temporal features
    df = df.withColumn("hour", hour(col("event_time")))
    df = df.withColumn("day_of_week", dayofweek(col("event_time")))
    df = df.withColumn("day_of_month", dayofmonth(col("event_time")))
    df = df.withColumn("week_of_year", weekofyear(col("event_time")))
    df = df.withColumn("month", month(col("event_time")))
    df = df.withColumn("event_timestamp", unix_timestamp(col("event_time")))
    df = df.withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))

    # Behavioral features (user-level aggregation)
    user_window = Window.partitionBy("user_id")

    df = df.withColumn("total_value", spark_sum("value").over(user_window))
    df = df.withColumn("total_events", count("event_time").over(user_window))
    df = df.withColumn("purchase_events", count(when(col("action") == "purchase", 1)).over(user_window))
    df = df.withColumn("add_to_cart_events", count(when(col("action") == "add_to_cart", 1)).over(user_window))
    df = df.withColumn("purchase_ratio", col("purchase_events") / col("total_events"))
    df = df.withColumn("add_to_cart_ratio", col("add_to_cart_events") / col("total_events"))
    
    # Rolling window: last 7 days for user

    window_7_days = SparkWindow.partitionBy("user_id").orderBy(col("event_time").cast("long")).rangeBetween(-7*86400, 0)
    df = df.withColumn("rolling_purchase_7d", spark_sum(when(col("action") == "purchase", 1).otherwise(0)).over(window_7_days))
    df = df.withColumn("rolling_value_7d", spark_sum("value").over(window_7_days))
    df = df.withColumn("rolling_events_7d", count("event_time").over(window_7_days))
    df = df.withColumn("rolling_avg_value_7d", (col("rolling_value_7d") / col("rolling_events_7d")).cast("double"))

    # Active days and average events per day
    active_days_df = df.select("user_id", to_date("event_time").alias("event_date")).distinct()
    active_days_count = active_days_df.groupBy("user_id").count().withColumnRenamed("count", "active_days")
    df = df.join(active_days_count, on="user_id", how="left")
    df = df.withColumn("avg_events_per_day", col("total_events") / col("active_days"))

    # Recency: days since last event
    df = df.withColumn("max_event_time", spark_max("event_time").over(user_window))
    df = df.withColumn("recency_days", (unix_timestamp(current_timestamp()) - unix_timestamp(col("max_event_time"))) / 86400)

    # User segmentation by frequency
    df = df.withColumn(
        "user_segment",
        when(col("total_events") >= 100, 2)
        .when(col("total_events") >= 20, 1)
        .otherwise(0)
    )

    logger.info("✅ Advanced feature engineering completed.")
    return df



