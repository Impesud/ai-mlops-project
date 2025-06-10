import os
import sys
import argparse
import subprocess
import time
from pyspark.sql.functions import when, lit, col
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import mlflow
import mlflow.spark
from utils.io import load_env_config
from utils.logging_utils import setup_logger
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def start_mlflow_ui(logger):
    logger.info("🔪 Starting MLflow UI in background…")
    try:
        subprocess.Popen([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", "./mlruns",
            "--port", "5000"
        ])
        if sys.stdin.isatty():
            input("Press Enter to close MLflow UI and exit…")
        else:
            logger.info("📱 MLflow UI started in background.")
            time.sleep(3)
    except Exception as e:
        logger.warning(f"⚠️ Could not start MLflow UI: {e}")

def plot_confusion_matrix(cm, classes, logger):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    fig_path = "confusion_matrix.png"
    plt.savefig(fig_path)
    plt.close()
    mlflow.log_artifact(fig_path)
    logger.info(f"📊 Confusion matrix saved and logged to MLflow: {fig_path}")

def main(mode: str, data_cfg=None, model_cfg=None, start_ui: bool = False):
    logger = setup_logger("train_spark", mode)

    # Load config from train.py or from environment
    if data_cfg is None or model_cfg is None:
        cfg = load_env_config(mode)
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]

    input_path = data_cfg["local_processed_path"]
    test_ratio = model_cfg.get("test_size", 0.2)
    model_params = model_cfg.get("model_params", {})

    # Map config parameters to Spark parameters
    if "n_estimators" in model_params:
        model_params["maxIter"] = model_params.pop("n_estimators")
    if "max_depth" in model_params:
        model_params["maxDepth"] = model_params.pop("max_depth")

    logger.info(f"Active mode: {mode}")
    logger.info(f"Input file: {input_path}")

    spark = (
        SparkSession.builder
        .appName("SparkModelTraining")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", os.environ['AWS_ACCESS_KEY_ID'])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ['AWS_SECRET_ACCESS_KEY'])
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{os.environ['AWS_REGION']}.amazonaws.com")
        .config("spark.driver.memory", "16g")
        .config("spark.executor.memory", "16g")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.executor.cores", "2")
        .config("spark.network.timeout", "600s")
        .config("spark.executor.heartbeatInterval", "60s")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", "./logs/spark-events")
        .getOrCreate()
    )
    logger.info(f"✨ Spark session started. Version: {spark.version}")

    try:
        df = spark.read.parquet(input_path)
        logger.info(f"✅ Loaded processed data with {df.count()} rows.")

        # 1. Create binary label: 1 only for "purchase", 0 for everything else
        df = df.withColumn(
            "label",
            when(col("action") == "purchase", 1.0).otherwise(0.0)
        )

        # 2. Log class distribution
        class_dist = df.groupBy("label").count().orderBy("label").toPandas()
        logger.info(f"📊 Class distribution:\n{class_dist.to_string(index=False)}")

        total = df.count()
        class_counts = df.groupBy("label").count().collect()
        class_weights = {row['label']: total / row['count'] for row in class_counts}
        logger.info(f"⚖️ Class weights: {class_weights}")

        # 3. Check for binary labels only (GBTClassifier requirement)
        unique_labels = sorted(class_weights.keys())
        if set(unique_labels) - {0.0, 1.0}:
            logger.error(f"❌ GBTClassifier supports only binary labels (0.0, 1.0), found: {unique_labels}")
            raise ValueError("Non-binary labels found! Please check your label creation logic.")

        # 4. Build class_weight column only if classes are imbalanced
        min_count = min([row['count'] for row in class_counts])
        max_count = max([row['count'] for row in class_counts])
        imbalance_ratio = max_count / min_count if min_count > 0 else 1
        use_class_weight = imbalance_ratio > 1.5  # threshold can be tuned

        if use_class_weight:
            expr = lit(1.0)
            for label, weight in class_weights.items():
                expr = when(col("label") == float(label), lit(weight)).otherwise(expr)
            df = df.withColumn("class_weight", expr)
            logger.info("⚖️ Using class weights for training.")
        else:
            df = df.withColumn("class_weight", lit(1.0))
            logger.info("ℹ️ Skipping class weights: classes are balanced.")

       # 5. Define features and pipeline (features from model config)
        random_seed = model_cfg.get("random_seed", 42)
        feature_cols = model_cfg.get("features", ["value", "hour", "day_of_week", "month"])
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        classifier = GBTClassifier(
            featuresCol="features",
            labelCol="label",
            weightCol="class_weight",
            maxIter=model_params.get("maxIter", 40),
            maxDepth=model_params.get("maxDepth", 10),
            stepSize=model_params.get("stepSize", 0.05),
            seed=random_seed
        )
        pipeline = Pipeline(stages=[assembler, classifier])

        # 6. Train/test split
        test_ratio = model_cfg.get("test_size", 0.2)
        train_data, test_data = df.randomSplit([1 - test_ratio, test_ratio], seed=random_seed)

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("spark-experiment")

        with mlflow.start_run() as run:
            # 7. Hyperparameter grid for cross-validation (from model config if present)
            grid_cfg = model_cfg.get("grid", {})
            paramGridBuilder = ParamGridBuilder()
            if "maxIter" in grid_cfg:
                paramGridBuilder = paramGridBuilder.addGrid(classifier.maxIter, grid_cfg["maxIter"])
            if "maxDepth" in grid_cfg:
                paramGridBuilder = paramGridBuilder.addGrid(classifier.maxDepth, grid_cfg["maxDepth"])
            if "stepSize" in grid_cfg:
                paramGridBuilder = paramGridBuilder.addGrid(classifier.stepSize, grid_cfg["stepSize"])
            paramGrid = paramGridBuilder.build()

            evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")

            crossval = CrossValidator(
                estimator=pipeline,
                estimatorParamMaps=paramGrid,
                evaluator=evaluator,
                numFolds=model_cfg.get("numFolds", 2),
                parallelism=model_cfg.get("parallelism", 2)
            )

            logger.info("🚀 Training Spark GBT model with CrossValidator...")
            cvModel = crossval.fit(train_data)
            model = cvModel.bestModel
            spark.catalog.clearCache()
            best_params = {p.name: v for p, v in model.stages[-1].extractParamMap().items()}
            logger.info(f"🏆 Best Params (clean): {best_params}")

            # Log feature importances
            importances = model.stages[-1].featureImportances
            logger.info(f"🌟 Feature importances: {importances}")
            for name, score in zip(feature_cols, importances):
                logger.info(f"Feature: {name} - Importance: {score}")

            predictions_train = model.transform(train_data)
            predictions_test = model.transform(test_data)

            # 8. Metrics logging (accuracy, F1, precision, recall, ROC AUC)
            train_accuracy = evaluator.evaluate(predictions_train, {evaluator.metricName: "accuracy"})
            test_accuracy = evaluator.evaluate(predictions_test, {evaluator.metricName: "accuracy"})
            f1 = evaluator.evaluate(predictions_test, {evaluator.metricName: "f1"})
            precision = evaluator.evaluate(predictions_test, {evaluator.metricName: "weightedPrecision"})
            recall = evaluator.evaluate(predictions_test, {evaluator.metricName: "weightedRecall"})

            # Log ROC AUC for binary classification
            binary_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
            auc = binary_evaluator.evaluate(predictions_test)
            logger.info(f"ROC AUC: {auc:.4f}")
            mlflow.log_metric("roc_auc", auc)

            logger.info(f"✅ Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

            mlflow.log_metrics({
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall
            })
            mlflow.log_params(model_params)
            mlflow.log_param("train_rows", train_data.count())
            mlflow.log_param("test_rows", test_data.count())

            # Log confusion matrix
            preds_pd = predictions_test.select("prediction", "label").toPandas()
            cm = confusion_matrix(preds_pd["label"], preds_pd["prediction"])
            classes = sorted(np.unique(preds_pd["label"]))
            plot_confusion_matrix(cm, classes=[f"Class {int(c)}" for c in classes], logger=logger)

            input_example = df.select("value", "hour", "day_of_week", "month").limit(5)
            input_example = input_example.selectExpr("cast(value as double) as value",
                                         "cast(hour as double) as hour",
                                         "cast(day_of_week as double) as day_of_week",
                                         "cast(month as double) as month")

            try:
                mlflow.spark.log_model(model, artifact_path="model", input_example=input_example.toPandas())
                logger.info("📦 Spark model logged to MLflow under 'model'.")

                run_id = mlflow.active_run().info.run_id
                model_uri = f"runs:/{run_id}/model"
                model_name = "SparkGBTModel"

                client = mlflow.tracking.MlflowClient()
                try:
                    prev_versions = client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
                except Exception:
                    prev_versions = []

                has_previous = len(prev_versions) > 0
                best_prev_f1 = None
                best_prev_auc = None

                # Search for best previous F1 and ROC AUC
                for v in prev_versions:
                    prev_run = client.get_run(v.run_id)
                    prev_f1 = prev_run.data.metrics.get("f1_score")
                    prev_auc = prev_run.data.metrics.get("roc_auc")
                    if prev_f1 is not None:
                        if best_prev_f1 is None or prev_f1 > best_prev_f1:
                            best_prev_f1 = prev_f1
                    if prev_auc is not None:
                        if best_prev_auc is None or prev_auc > best_prev_auc:
                            best_prev_auc = prev_auc

                # Register model if no previous or if performance is better (using F1-score as main metric)
                if not has_previous or best_prev_f1 is None or f1 > best_prev_f1:
                    result = mlflow.register_model(model_uri=model_uri, name=model_name)
                    if not has_previous:
                        logger.info(
                            f"✅ Registered first model version ({result.version}) with f1_score={f1:.4f}"
                        )
                    else:
                        logger.info(
                            f"✅ Registered new model version ({result.version}) with f1_score={f1:.4f} "
                            f"(previous best: {best_prev_f1 if best_prev_f1 is not None else 'N/A'})"
                        )
                else:
                    logger.info(
                        f"ℹ️ Model NOT registered: f1_score={f1:.4f} <= previous best ({best_prev_f1:.4f})"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Failed to log/register Spark model to MLflow: {e}")

        logger.info("🌾 MLflow run completed.")

    except Exception as e:
        logger.exception(f"❌ Training pipeline failed: {e}")
        raise

    finally:
        try:
            spark.stop()
            logger.info("🚩 Spark session stopped.")
        except Exception as stop_err:
            logger.warning(f"⚠️ Failed to stop Spark session cleanly: {stop_err}")

    if start_ui:
        start_mlflow_ui(logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="prod", help="Environment: dev or prod")
    parser.add_argument("--mlflow-ui", action="store_true", help="Start MLflow UI after training")
    args = parser.parse_args()

    main(args.env, start_ui=args.mlflow_ui)







