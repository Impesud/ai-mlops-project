import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay,
    precision_recall_curve
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from utils.io import load_env_config
from utils.logging_utils import setup_logger

# ---------------------------------------------------------
def load_and_split_data(path: str, test_size: float, random_seed: int, logger: logging.Logger):
    df = pd.read_parquet(path)
    logger.info(f"Loaded data with shape: {df.shape}")
    df["label"] = (df["action"] == "purchase").astype(int)
    df["event_time"] = pd.to_datetime(df["event_time"])
    times = df["event_time"]

    feature_cols = [
        "value", "hour", "day_of_week", "day_of_month", "week_of_year", "month",
        "is_weekend", "total_value", "total_events", "purchase_events", "purchase_ratio",
        "avg_events_per_day", "recency_days", "rolling_purchase_7d", "rolling_value_7d",
        "rolling_events_7d", "rolling_avg_value_7d", "user_segment"
    ]
    X = df[feature_cols]
    y = df["label"]
    return train_test_split(X, y, times, test_size=test_size, random_state=random_seed, stratify=y)

# ---------------------------------------------------------
def build_pipeline(random_seed: int, model_params: dict):
    params = model_params.copy()
    params.setdefault("class_weight", "balanced")
    params.setdefault("n_jobs", -1)
    params["random_state"] = random_seed

    rf = RandomForestClassifier(**params)
    pipeline = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote_tomek", SMOTETomek(random_state=random_seed)),
        ("rf", rf),
    ])
    return pipeline

# ---------------------------------------------------------
def perform_hyperparameter_search(pipeline, X_train, y_train, param_grid, random_seed: int, logger: logging.Logger, cv_folds: int):
    logger.info("🚀 Starting GridSearchCV...")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    logger.info(f"✅ Best parameters: {grid_search.best_params_}")
    mlflow.log_params(grid_search.best_params_)
    return grid_search.best_estimator_

# ---------------------------------------------------------
def calibrate_pipeline(pipeline, X_train, y_train):
    X_res, y_res = pipeline.named_steps['smote_tomek'].fit_resample(
        pipeline.named_steps['scaler'].fit_transform(X_train), y_train)
    calibrated_rf = CalibratedClassifierCV(pipeline.named_steps['rf'], cv=3, method="isotonic")
    calibrated_rf.fit(X_res, y_res)
    pipeline.steps[-1] = ("calibrated_rf", calibrated_rf)
    return pipeline

# ---------------------------------------------------------
def evaluate_and_log(pipeline, X_train, y_train, X_test, y_test, t_test, logger: logging.Logger):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.nanargmax(f1_scores)
    best_threshold = thresholds[best_idx]
    mlflow.log_metric("best_threshold", best_threshold)
    logger.info(f"Auto best threshold: {best_threshold:.3f}")

    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba)
    }
    for k, v in metrics.items():
        mlflow.log_metric(k, v)
        logger.info(f"{k}: {v:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, val, ha="center")
    plt.title("Confusion Matrix")
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmpfile.name)
    mlflow.log_artifact(tmpfile.name, artifact_path="confusion_matrix")
    tmpfile.close(); os.unlink(tmpfile.name)
    plt.close()

    # Classification Report
    cr = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    tmpfile = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    cr.to_parquet(tmpfile.name)
    mlflow.log_artifact(tmpfile.name, artifact_path="classification_report")
    tmpfile.close(); os.unlink(tmpfile.name)

    # ROC Curve
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    plt.title("ROC Curve")
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmpfile.name)
    mlflow.log_artifact(tmpfile.name, artifact_path="roc_curve")
    tmpfile.close(); os.unlink(tmpfile.name)
    plt.close()

    # Precision-Recall Curve
    fig, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
    plt.title("Precision-Recall Curve")
    tmpfile = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmpfile.name)
    mlflow.log_artifact(tmpfile.name, artifact_path="precision_recall_curve")
    tmpfile.close(); os.unlink(tmpfile.name)
    plt.close()

    # Rolling window metrics (7 days)
    df_roll = pd.DataFrame({'date': t_test.dt.floor('D'), 'y_true': y_test, 'y_proba': y_proba})
    daily = df_roll.groupby('date').agg({'y_true': list, 'y_proba': list}).reset_index()
    if len(daily) >= 7:
        roll_metrics = []
        for i in range(6, len(daily)):
            window = daily.iloc[i-6:i+1]
            yt = np.concatenate(window['y_true'].to_list())
            pr = np.concatenate(window['y_proba'].to_list())
            yp = (pr >= best_threshold).astype(int)
            roll_metrics.append({
                'period': f"{window.iloc[0]['date'].date()} to {window.iloc[-1]['date'].date()}",
                'precision': precision_score(yt, yp, zero_division=0),
                'recall': recall_score(yt, yp, zero_division=0),
                'f1': f1_score(yt, yp, zero_division=0),
                'roc_auc': roc_auc_score(yt, pr) if len(np.unique(yt)) > 1 else np.nan
            })
        roll_df = pd.DataFrame(roll_metrics)
        tmpfile = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
        roll_df.to_parquet(tmpfile.name)
        mlflow.log_artifact(tmpfile.name, artifact_path="rolling_metrics")
        tmpfile.close(); os.unlink(tmpfile.name)

    # Full model export
    input_example = X_train.head(5).copy()
    for col in input_example.select_dtypes("integer").columns:
        input_example[col] = input_example[col].astype("float64")
    sig = infer_signature(input_example, pipeline.predict(input_example))
    mlflow.sklearn.log_model(pipeline, artifact_path="model", input_example=input_example, signature=sig)

# ---------------------------------------------------------
def main(env: str):
    logger = setup_logger("train_sklearn", env)
    cfg = load_env_config(env)
    data_cfg, model_cfg = cfg["data"], cfg["model"]

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("sklearn-experiment")
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test, t_train, t_test = load_and_split_data(
            data_cfg["local_processed_path"], model_cfg.get("test_size", 0.2), model_cfg.get("random_seed", 42), logger)

        pipeline = build_pipeline(model_cfg.get("random_seed", 42), model_cfg.get("model_params", {}))

        if model_cfg.get("do_hyper_search", False):
            param_grid = model_cfg.get("param_grid", {})
            cv_folds = model_cfg.get("cv_folds", 5)
            pipeline = perform_hyperparameter_search(pipeline, X_train, y_train, param_grid, model_cfg.get("random_seed", 42), logger, cv_folds)

        pipeline = calibrate_pipeline(pipeline, X_train, y_train)
        evaluate_and_log(pipeline, X_train, y_train, X_test, y_test, t_test, logger)

# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", help="Environment: dev or prod")
    args = parser.parse_args()
    main(args.env)




