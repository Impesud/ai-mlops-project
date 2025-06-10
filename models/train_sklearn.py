import sys
import argparse
import subprocess
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from utils.io import load_env_config
from utils.logging_utils import setup_logger

def main(mode: str, start_ui: bool = False, data_cfg=None, model_cfg=None):
    logger = setup_logger("train_sklearn", mode)
    # Allow passing config from train.py or loading here
    if data_cfg is None or model_cfg is None:
        cfg = load_env_config(mode)
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]

    input_path = data_cfg["local_processed_path"]
    test_size = model_cfg.get("test_size", 0.2)
    random_seed = model_cfg.get("random_seed", 42)
    model_params = model_cfg.get("model_params", {})

    logger.info(f"🔧 Mode: {mode}")
    logger.info(f"📥 Input path: {input_path}")

    try:
        df = pd.read_parquet(input_path)
        logger.info(f"✅ Loaded data with shape: {df.shape}")

        if "action" not in df.columns:
            raise ValueError("Missing 'action' column required for label creation.")

        # Binary label for classification
        df["label"] = (df["action"] == "purchase").astype(int)

        # Drop unused columns
        df = df.drop(columns=["user_id", "action", "event_time"], errors="ignore")

        X = df.drop("label", axis=1)
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_seed,
            stratify=y
        )
        logger.info(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

        try:
            from imblearn.over_sampling import SMOTE
            sm = SMOTE(random_state=random_seed)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            logger.info(f"🔄 SMOTE: {len(y_train)} → {len(y_res)} samples")
        except ImportError:
            logger.warning("⚠️ imbalanced-learn not installed. Skipping SMOTE.")
            X_res, y_res = X_train, y_train
            logger.info(f"ℹ️ Using original data: {len(y_train)} samples")

        # Convert integer columns to float64 to avoid MLflow schema issues
        int_cols = X_res.select_dtypes(include=["int", "int32", "int64"]).columns
        if len(int_cols) > 0:
            logger.info(f"🔁 Converting integer columns to float64 for MLflow compatibility: {list(int_cols)}")
            X_res[int_cols] = X_res[int_cols].astype("float64")

        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(f"sklearn-experiment")

        with mlflow.start_run():
            mlflow.log_params(model_params)

            model = RandomForestClassifier(
                **model_params,
                class_weight="balanced"
            )
            model.fit(X_res, y_res)

            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            train_acc = accuracy_score(y_train, y_pred_train)
            test_acc = accuracy_score(y_test, y_pred_test)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("test_accuracy", test_acc)

            logger.info(f"✅ Train Accuracy: {train_acc:.4f}")
            logger.info(f"✅ Test Accuracy: {test_acc:.4f}")
            logger.info("\n" + classification_report(y_test, y_pred_test))
            logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred_test)))

            input_example = X_res[:5]
            signature = infer_signature(X_res, model.predict(X_res))

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                input_example=input_example,
                signature=signature
            )

        logger.info("🎯 Training complete.")

    except Exception as e:
        logger.exception(f"❌ Training failed: {e}")
        raise

    if start_ui:
        start_mlflow_ui(logger)

def start_mlflow_ui(logger):
    logger.info("🚀 Launching MLflow UI...")
    try:
        subprocess.Popen([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", "./mlruns",
            "--port", "5000"
        ])
        if sys.stdin.isatty():
            input("Press Enter to stop MLflow UI...")
        else:
            logger.info("📡 MLflow UI started in background.")
            time.sleep(3)
    except Exception as e:
        logger.warning(f"⚠️ Could not start MLflow UI: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", help="Environment: dev or prod")
    parser.add_argument("--mlflow-ui", action="store_true", help="Start MLflow UI after training")
    args = parser.parse_args()
    main(args.env, args.mlflow_ui)

