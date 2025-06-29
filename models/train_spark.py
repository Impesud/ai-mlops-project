import argparse
import io
import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.spark
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, when
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

from utils.io import load_env_config
from utils.logging_utils import setup_logger


# ------------------------------------------------------------------------
def start_spark():
    return (
        SparkSession.builder.appName("SparkModelTraining")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", os.environ["AWS_ACCESS_KEY_ID"])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ["AWS_SECRET_ACCESS_KEY"])
        .config(
            "spark.hadoop.fs.s3a.endpoint",
            f"s3.{os.environ['AWS_DEFAULT_REGION']}.amazonaws.com",
        )
        .config("spark.driver.memory", "16g")
        .config("spark.executor.memory", "16g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.cores", "2")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "60s")
        # For cluster/cloud environments, you might need to adjust the following:
        # .config("spark.sql.shuffle.partitions", "200")
        # .config("spark.default.parallelism", "200")
        # .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # .config("spark.sql.adaptive.enabled", "true")
        # .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # .config("spark.dynamicAllocation.enabled", "true")
        .getOrCreate()
    )


# ------------------------------------------------------------------------
def create_labels(df):
    if "label" not in df.columns:
        df = df.withColumn("label", when(col("action") == "purchase", 1.0).otherwise(0.0))
    return df


# ------------------------------------------------------------------------
def compute_class_weights(df):
    counts = df.groupBy("label").count().toPandas()
    total = counts["count"].sum()
    weights = {row["label"]: total / row["count"] for _, row in counts.iterrows()}
    return weights


# ------------------------------------------------------------------------
def apply_class_weights(df, weights):
    expr = lit(1.0)
    for label, weight in weights.items():
        expr = when(col("label") == label, lit(weight)).otherwise(expr)
    return df.withColumn("class_weight", expr)


# ------------------------------------------------------------------------
def build_pipeline(features, model_params):
    assembler = VectorAssembler(inputCols=features, outputCol="features")
    gbt = GBTClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="class_weight",
        seed=model_params.get("random_seed", 42),
        maxIter=model_params.get("maxIter", 40),
        maxDepth=model_params.get("maxDepth", 10),
        stepSize=model_params.get("stepSize", 0.05),
        subsamplingRate=model_params.get("subsamplingRate", 1.0),
        minInstancesPerNode=model_params.get("minInstancesPerNode", 1),
        minInfoGain=model_params.get("minInfoGain", 0.0),
    )
    return Pipeline(stages=[assembler, gbt]), gbt


# ------------------------------------------------------------------------
def build_param_grid(gbt, grid_cfg):
    builder = ParamGridBuilder()
    if grid_cfg:
        for param, values in grid_cfg.items():
            spark_param = getattr(gbt, param)
            builder = builder.addGrid(spark_param, values)
    return builder.build()


# ------------------------------------------------------------------------
def plot_conf_matrix(y_true, y_pred, logger):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    image = Image.open(buf)
    mlflow.log_image(image, "confusion_matrix.png")
    logger.info("📊 Confusion matrix logged.")


# ------------------------------------------------------------------------
def plot_roc_curve(y_true, y_scores, logger):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)

    image = Image.open(buf)

    mlflow.log_image(image, "roc_curve.png")
    logger.info("📈 ROC curve logged.")

    return fpr, tpr, thresholds, roc_auc


# ------------------------------------------------------------------------
def find_best_threshold(fpr, tpr, thresholds):
    youden_index = tpr - fpr
    best_idx = np.argmax(youden_index)
    best_threshold = thresholds[best_idx]
    mlflow.log_param("best_threshold", best_threshold)
    return best_threshold


# ------------------------------------------------------------------------
def compute_metrics_at_threshold(y_true, y_scores, threshold):
    y_pred_adjusted = (y_scores >= threshold).astype(int)
    precision = precision_score(y_true, y_pred_adjusted, zero_division=0)
    recall = recall_score(y_true, y_pred_adjusted, zero_division=0)
    f1_adj = f1_score(y_true, y_pred_adjusted, zero_division=0)
    mlflow.log_metric("precision_adj", precision)
    mlflow.log_metric("recall_adj", recall)
    mlflow.log_metric("f1_adj", f1_adj)
    return y_pred_adjusted


# ------------------------------------------------------------------------
def rolling_evaluation(pred_pd, threshold):
    pred_pd["event_date"] = pd.to_datetime(pred_pd["event_time"]).dt.floor("D")
    daily = pred_pd.groupby("event_date").agg(list).reset_index()
    roll_metrics = []

    for i in range(6, len(daily)):
        window = daily.iloc[i - 6 : i + 1]
        yt = np.concatenate(window["label"].to_list())
        pr = np.concatenate(window["probability"].apply(lambda x: [p[1] for p in x]).to_list())
        yp = (pr >= threshold).astype(int)
        roll_metrics.append(
            {
                "period": f"{window.iloc[0]['event_date'].date()} to {window.iloc[-1]['event_date'].date()}",
                "precision": precision_score(yt, yp, zero_division=0),
                "recall": recall_score(yt, yp, zero_division=0),
                "f1": f1_score(yt, yp, zero_division=0),
                "roc_auc": roc_auc_score(yt, pr) if len(np.unique(yt)) > 1 else np.nan,
            }
        )

    roll_df = pd.DataFrame(roll_metrics)
    mlflow.log_table(roll_df, "rolling_metrics.parquet")


# ------------------------------------------------------------------------
def main(env):
    logger = setup_logger("train_spark", env)
    cfg = load_env_config(env)
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]

    input_path = data_cfg["local_processed_path"]
    features = model_cfg["features"]
    test_size = model_cfg["test_size"]
    seed = model_cfg["random_seed"]
    model_params = model_cfg["model_params"]
    grid_cfg = model_cfg.get("grid", {})
    numFolds = model_cfg.get("numFolds", 2)
    parallelism = model_cfg.get("parallelism", 2)

    logger.info(f"Loaded config for environment: {env}")

    spark = start_spark()
    df = spark.read.parquet(input_path)
    df = create_labels(df)

    num_cores = int(os.environ.get("NUM_CORES", 8))
    num_partitions = num_cores * 2
    df = df.repartition(num_partitions).persist()

    weights = compute_class_weights(df)
    df = apply_class_weights(df, weights)

    train, test = df.randomSplit([1 - test_size, test_size], seed=seed)

    pipeline, gbt = build_pipeline(features, model_params)
    param_grid = build_param_grid(gbt, grid_cfg)

    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=numFolds,
        parallelism=parallelism,
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("spark-experiment")

    with mlflow.start_run():
        logger.info("🚀 Training with full evaluation...")
        model = crossval.fit(train).bestModel

        preds = model.transform(test).cache()
        pred_pd = preds.select("label", "probability", "event_time").toPandas()
        y_true = pred_pd["label"].astype(int)
        y_scores = pred_pd["probability"].apply(lambda x: float(x[1])).values

        auc_value = evaluator.evaluate(preds)
        acc = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="accuracy"
        ).evaluate(preds)
        f1_spark = MulticlassClassificationEvaluator(
            labelCol="label", predictionCol="prediction", metricName="f1"
        ).evaluate(preds)

        mlflow.log_metrics({"roc_auc": auc_value, "accuracy": acc, "f1_score": f1_spark})

        # ROC + threshold tuning
        fpr, tpr, thresholds, roc_auc_final = plot_roc_curve(y_true, y_scores, logger)
        best_threshold = find_best_threshold(fpr, tpr, thresholds)
        y_pred_adjusted = compute_metrics_at_threshold(y_true, y_scores, best_threshold)
        plot_conf_matrix(y_true, y_pred_adjusted, logger)

        # Rolling evaluation
        rolling_evaluation(pred_pd, best_threshold)

        mlflow.log_params(model_params)
        mlflow.log_param("train_rows", train.count())
        mlflow.log_param("test_rows", test.count())

        mlflow.spark.log_model(model, artifact_path="model")
        logger.info("✅ Model fully logged to MLflow.")

    spark.stop()


# ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="prod", help="Environment: dev or prod")
    args = parser.parse_args()
    main(args.env)
