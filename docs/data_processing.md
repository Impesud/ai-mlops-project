# AI MLOps Project - Data Processing & Feature Engineering Documentation

---

## 🧶 Overview

This document provides complete technical documentation of the data processing pipeline and feature engineering used to build the behavioral purchase prediction model.

The entire pipeline transforms raw user event logs into a fully cleaned, enriched dataset ready for machine learning training.

---

## 🔎 Raw Input Data Schema

| Column Name  | Type      | Description                                                   |
| ------------ | --------- | ------------------------------------------------------------- |
| `user_id`    | String    | Unique user identifier                                        |
| `event_time` | Timestamp | Datetime of event                                             |
| `action`     | String    | Type of user action (`purchase`, `add_to_cart`, `view`, etc.) |
| `value`      | Double    | Monetary value of transaction                                 |

---

## 🥹 Data Cleaning Pipeline

### 1️⃣ Schema Validation

- Ensures all required columns exist.
- Casts columns to expected types if mismatches detected.
- Strict error raised if critical columns missing.

### 2️⃣ Cleaning Steps

| Step | Description |
| ---- | ----------- |
|      |             |

| **Null Removal**          | Drop rows where any of `user_id`, `event_time`, `action`, `value` are missing |
| ------------------------- | ----------------------------------------------------------------------------- |
| **String Normalization**  | Lowercase + trim `user_id` and `action` fields                                |
| **Missing Defaults**      | Fill nulls: `value = 0.0`, `action = unknown`                                 |
| **Negative Value Filter** | Remove rows where `value < 0`                                                 |
| **Outlier Removal**       | Optional: Filter extreme values using IQR method                              |
| **Duplicate Removal**     | Drop duplicates on `(user_id, event_time, action)`                            |
| **Future Filtering**      | Discard events with `event_time` after current timestamp                      |

---

## 🧠 Feature Engineering

### Complete Feature Set (v3)

| Feature Name                      | Description                               | Technical Details                                  |
| --------------------------------- | ----------------------------------------- | -------------------------------------------------- |
| `user_id`                         | User identifier                           | Normalized string (lowercase, trimmed)             |
| `event_time`                      | Original timestamp                        | Cleaned and validated timestamp                    |
| `action`                          | Event type                                | Normalized action string                           |
| `value`                           | Transaction value                         | Original event value                               |
| `hour`                            | Hour of event                             | Extracted from `event_time` (0-23)                 |
| `day_of_week`                     | Day of week                               | Extracted from `event_time` (1=Sunday, 7=Saturday) |
| `day_of_month`                    | Day of month                              | Extracted from `event_time`                        |
| `week_of_year`                    | Week number of year                       | Extracted from `event_time`                        |
| `month`                           | Month of year                             | Extracted from `event_time`                        |
| `event_timestamp`                 | Unix timestamp                            | Conversion of `event_time`                         |
| `is_weekend`                      | Weekend flag                              | 1 if Saturday/Sunday, else 0                       |
| `total_value`                     | Total monetary value per user             | Sum of `value` for the user                        |
| `total_events`                    | Total number of events per user           | Count of events                                    |
| `purchase_events`                 | Total purchase events per user            | Count where `action == purchase`                   |
| `add_to_cart_events`              | Total add-to-cart events per user         | Count where `action == add_to_cart`                |
| `purchase_ratio`                  | Purchase event ratio                      | `purchase_events / total_events`                   |
| `add_to_cart_ratio`               | Add-to-cart event ratio                   | `add_to_cart_events / total_events`                |
| `recency_days`                    | Days since last event                     | `(current_date - max(event_time)) in days`         |
| `active_days`                     | Distinct active days per user             | Count of unique dates                              |
| `avg_events_per_day`              | Average events per active day             | `total_events / active_days`                       |
| `avg_value_per_event`             | Average transaction value per event       | `total_value / total_events`                       |
| `purchase_conversion_value_ratio` | Purchase-to-value ratio                   | `purchase_events / total_value`                    |
| `cart_conversion_value_ratio`     | Add-to-cart-to-value ratio                | `add_to_cart_events / total_value`                 |
| `avg_days_between_events`         | Average days between events               | `active_days / total_events`                       |
| `rolling_purchase_7d`             | Total purchases in last 7 days (per user) | Calculated over 7-day rolling window               |
| `rolling_value_7d`                | Total value in last 7 days (per user)     | 7-day rolling sum of transaction value             |
| `rolling_events_7d`               | Total events in last 7 days               | 7-day event count window                           |
| `rolling_avg_value_7d`            | Average value per event (last 7d)         | `rolling_value_7d / rolling_events_7d`             |
| `user_segment`                    | User segmentation based on activity       | Segment users as: `frequent` vs `occasional`       |

---

## 🛠️ Pipeline Processing Flow

```mermaid
flowchart TD
  A[Raw Input Data] --> B(Basic Cleaning)
  B --> C(Advanced Feature Engineering)
  C --> D(Rolling Windows & Segmentation)
  D --> E[Processed Dataset (Parquet)]
  E --> F[Training Pipeline]
```

---

## ⚙️ Technical Implementation Notes

- Full pipeline implemented in PySpark for scalable distributed processing.
- All aggregations partitioned by `user_id` using Spark Window functions.
- Rolling features leverage time-based filtering via `event_time`.
- Segmentations applied after all feature generation for optimal training balance.
- Processed data outputted in Parquet format for efficient ML pipeline consumption.
- MLflow used for full experiment tracking and model versioning.

---

## 🚀 Possible Future Extensions

| Extension                         | Description                                          |
| --------------------------------- | ---------------------------------------------------- |
| **Session Features**              | Extract session gaps, duration, click path sequences |
| **Customer Lifetime Value (CLV)** | Lifetime revenue projection                          |
| **Time Since Last Purchase**      | Recency relative to purchase only                    |
| **Seasonality Features**          | Add holiday flags, special sale periods              |
| **User Embeddings**               | Representation learning via product sequences        |
| **Anomaly Detection**             | Detect abnormal purchase spikes                      |

---

## 🕺 File Structure

- Codebase: `data_processing/`
- Main Pipeline Entry: `process.py`
- Feature Logic: `features.py`
- Spark Session Utilities: `utils/spark_utils.py`
- Spark I/O Utilities: `utils/spark_io.py`
- Config Loader: `utils/io.py`
- Full ML Pipelines (training): `models/train_sklearn.py` & `models/train_spark.py`

---

*Last updated: June 2025 (Feature Set v3 with Rolling Windows & Segmentation Complete)*

