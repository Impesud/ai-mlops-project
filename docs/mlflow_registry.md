# AI MLOps Project - MLflow Integration

---

## 🔗 MLflow Full Integration Overview

This document describes how MLflow is fully integrated into the AI MLOps Platform across both training engines.

---

## 🔄 MLflow Core Roles

- **Experiment Tracking:** Parameters, metrics, and artifacts logging
- **Model Registry:** Version control and conditional registration of models
- **Artifact Storage:** Confusion matrices, classification reports, PR & ROC curves, rolling metrics
- **Model Serving (future):** Compatible with `mlflow models serve`

---

## 🌐 MLflow Setup

- **Backend URI:** Local file system (`mlruns/` directory)
- **UI Access:**

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Access UI at: [http://localhost:5000](http://localhost:5000)

---

## 💡 What Gets Logged

| Type               | Content                                                                         |
| ------------------ | ------------------------------------------------------------------------------- |
| **Params**         | Full model and pipeline parameters from YAML config                             |
| **Metrics**        | Accuracy, Precision, Recall, F1-score, ROC AUC, Rolling metrics                 |
| **Artifacts**      | Confusion Matrix, Classification Report, ROC Curve, PR Curve, Rolling 7-day CSV |
| **Models**         | Full model serialization with signature (sklearn & Spark)                       |
| **Best Threshold** | Auto-selected threshold maximizing F1-score                                     |

---

## 🏅 Registration Logic

- Models are automatically registered in MLflow Model Registry **only if**:

  - New model achieves better F1-score than previous versions.
  - Comparison is performed across all existing versions for same model name.

- Spark and sklearn models are independently versioned.

---

## 🌿 MLflow Usage in `train_sklearn.py`

- Autologging activated via `mlflow.sklearn.autolog()`
- Manual logging of metrics, threshold, artifacts, rolling window metrics.
- Full calibrated model pipeline logged with input examples and signatures.

---

## 🌿 MLflow Usage in `train_spark.py`

- Manual logging of all Spark model metrics.
- Confusion Matrix logged via seaborn visual artifact.
- Spark MLlib model logged via `mlflow.spark.log_model()`.
- Automatic model versioning and registry update if improvement detected.

---

## 🌐 Artifacts Example

- `confusion_matrix.png`
- `classification_report.csv`
- `roc_curve.png`
- `pr_curve.png`
- `rolling_metrics.csv`
- Full model object (`model/` folder)

---

## 🔄 Example Workflow

```bash
# Train (dev mode)
python models/train.py --env dev

# Train (prod mode)
python models/train.py --env prod

# Start MLflow UI
mlflow ui --backend-store-uri ./mlruns --port 5000
```

---

## 🎯 Next Planned MLflow Enhancements

- Full remote MLflow tracking backend (S3 & others)
- Model lifecycle automation (Staging -> Production promotion)
- Model monitoring dashboards (via MLflow + Grafana)
- Integrated serving deployment pipeline

---

**Documentation updated: June 2025 (Post Calibration Update)**
