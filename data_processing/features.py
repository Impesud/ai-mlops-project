# data_processing/features.py

from pyspark.sql.functions import (
    col, hour, dayofweek, dayofmonth, weekofyear,
    month, year, unix_timestamp, when, trim, lower as spark_lower, current_timestamp
)
from pyspark.sql.types import StringType, TimestampType, DoubleType, DataType
from pyspark.sql import DataFrame
from typing import Dict
import logging

REQUIRED_COLUMNS: Dict[str, DataType] = {
    "user_id": StringType(),
    "event_time": TimestampType(),
    "action": StringType(),
    "value": DoubleType()
}

logger = logging.getLogger("features")


def validate_schema(df: DataFrame) -> DataFrame:
    """
    Ensures the DataFrame has all required columns with correct types.
    If a column has a mismatched type, it will be cast to the expected type.
    """
    logger.info("🔍 Validating and casting schema if needed...")
    for col_name, expected_type in REQUIRED_COLUMNS.items():
        if col_name not in df.columns:
            logger.error(f"❌ Missing required column: {col_name}")
            raise ValueError(f"Missing required column: {col_name}")

        actual_type = df.schema[col_name].dataType
        if type(actual_type) != type(expected_type):
            logger.warning(
                f"⚠️ Column '{col_name}' has type {actual_type}, casting to {expected_type}"
            )
            df = df.withColumn(col_name, col(col_name).cast(expected_type))

    logger.info("✅ Schema validated and casted where needed.")
    return df


def basic_cleaning(df: DataFrame, handle_outliers: bool = False) -> DataFrame:
    """
    Basic data cleaning with:
    - Schema validation
    - Null handling (with fallback)
    - String trimming
    - Optional outlier filtering
    - Duplicate & future filtering
    """
    logger.info("🧹 Performing basic cleaning...")
    df = validate_schema(df)

    # 1. Drop rows with missing key fields
    df = df.dropna(subset=["user_id", "event_time", "action", "value"])

    # 2. Standardize string fields
    df = df.withColumn("user_id", trim(spark_lower(col("user_id"))))
    df = df.withColumn("action", trim(spark_lower(col("action"))))

    # 3. Fill missing values with fallback defaults
    df = df.fillna({
        "value": 0.0,
        "action": "unknown"
    })

    # 4. Remove negative values
    df = df.filter(col("value") >= 0)

    # 5. Optionally handle outliers with IQR method
    if handle_outliers:
        q1 = df.approxQuantile("value", [0.25], 0.05)[0]
        q3 = df.approxQuantile("value", [0.75], 0.05)[0]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        logger.info(f"📊 Outlier thresholds: lower={lower_bound:.2f}, upper={upper_bound:.2f}")
        df = df.filter((col("value") >= lower_bound) & (col("value") <= upper_bound))

    # 6. Remove duplicates
    df = df.dropDuplicates(["user_id", "event_time", "action"])

    # 7. Remove future timestamps
    df = df.filter(col("event_time") <= current_timestamp())

    logger.info("✅ Basic cleaning completed.")
    return df


def advanced_feature_engineering(df: DataFrame) -> DataFrame:
    """
    Extracts time-based features and additional metadata.
    """
    logger.info("🧠 Starting advanced feature engineering...")
    df = df.withColumn("hour", hour(col("event_time")))
    df = df.withColumn("day_of_week", dayofweek(col("event_time")))
    df = df.withColumn("day_of_month", dayofmonth(col("event_time")))
    df = df.withColumn("week_of_year", weekofyear(col("event_time")))
    df = df.withColumn("month", month(col("event_time")))
    #df = df.withColumn("year", year(col("event_time")))
    df = df.withColumn("event_timestamp", unix_timestamp(col("event_time")))
    df = df.withColumn("is_weekend", when(col("day_of_week").isin([1, 7]), 1).otherwise(0))
    logger.info("✅ Advanced feature engineering completed.")

    return df



