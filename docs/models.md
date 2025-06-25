# AI MLOps Project - Models Training Documentation

---

## 📂 Module Overview

This documentation describes the full model training architecture implemented in the `models/` package of the AI MLOps Platform.

The module supports two fully independent training pipelines depending on the environment:

- **Development Mode (sklearn)** → Local training using Scikit-learn
- **Production Mode (spark)** → Distributed training using Spark MLlib

---

## 🔧 Model Training Pipelines

### 1️⃣ `train_sklearn.py` (Development Mode)

#### 🔑 Key Characteristics

| Component                 | Description                                                      |
| ------------------------- | ---------------------------------------------------------------- |
| **Algorithm**             | Random Forest Classifier                                         |
| **Data Preprocessing**    | StandardScaler normalization + SMOTETomek oversampling           |
| **Hyperparameter Tuning** | GridSearchCV (configurable via YAML)                             |
| **Calibration**           | Isotonic Calibration (CalibratedClassifierCV)                    |
| **Threshold Selection**   | Automatic optimal threshold via Precision-Recall curve           |
| **Evaluation Metrics**    | Accuracy, Precision, Recall, F1, ROC AUC, Rolling Window F1      |
| **Rolling Metrics**       | 7-day sliding evaluation window for time-series stability        |
| **Experiment Tracking**   | MLflow (local filesystem backend)                                |
| **Artifacts Exported**    | Model signature, confusion matrix, classification report, curves |
| **Feature Engineering**   | Includes rolling 7-day windows and user segmentation             |

#### 🔧 Pipeline Architecture

```
Raw Features --> StandardScaler --> SMOTETomek --> RandomForest --> Calibration --> Threshold Optimization
```

#### 🔗 Recent Results (June 2025)

| Metric             | Value   |
| ------------------ | ------- |
| **Accuracy**       | 71.84%  |
| **Precision**      | 43.96%  |
| **Recall**         | 67.99%  |
| **F1-score**       | 53.39%  |
| **ROC AUC**        | 77.33%  |
| **Best Threshold** | 0.215   |

✅ These results reflect the latest improvements:

- Porting of advanced behavioral features from Spark
- Full GridSearchCV optimization
- Isotonic model calibration
- Thresholding via F1 maximization
- MLflow artifact tracking (no residual local files)

#### 🔗 Configuration File

- Path: `configs/dev/model.yaml`
- Fully drives hyperparameter search, calibration, feature selection, and MLflow logging.

#### 🔖 Main Outputs Logged to MLflow

- Confusion Matrix (`confusion_matrix/confusion_matrix.png`)
- ROC Curve (`roc_curve/roc_curve.png`)
- Precision-Recall Curve (`precision_recall_curve/precision_recall_curve.png`)
- Classification Report (`classification_report/classification_report.parquet`)
- Rolling Window Metrics (`rolling_metrics/rolling_metrics.parquet`)
- Full Scikit-learn model (`model/` with signature & input example)

---

### 2️⃣ `train_spark.py` (Production Mode)

#### 🔑 Key Characteristics

| Component                 | Description                                         |
| ------------------------- | --------------------------------------------------- |
| **Algorithm**             | Gradient Boosted Trees (GBTClassifier)              |
| **Pipeline**              | VectorAssembler + GBTClassifier (Spark ML Pipeline) |
| **Imbalance Handling**    | Class Weight Compensation (calculated dynamically)  |
| **Hyperparameter Tuning** | Spark CrossValidator (ParamGridBuilder)             |
| **Distributed Training**  | Fully parallelized Spark execution                  |
| **Evaluation Metrics**    | Accuracy, F1, Precision, Recall, ROC AUC            |
| **Experiment Tracking**   | MLflow (local or remote tracking backend)           |
| **Model Registration**    | Full MLflow Model Registry integration              |
| **Feature Engineering**   | Rolling 7-day metrics + segmentation (same as dev)  |

#### 🔧 Pipeline Architecture

```
Raw Features --> VectorAssembler --> GBTClassifier --> CrossValidation --> Model Registration
```

#### 🔗 Configuration File

- Path: `configs/prod/model.yaml`
- Fully configurable hyperparameter space and Spark resources.

#### 🔖 Main Outputs Logged to MLflow

- Confusion Matrix (seaborn heatmap)
- Full Spark MLlib model serialization
- Model Registry version control logic
- Metrics: precision_adj, recall_adj, f1_adj, roc_auc, f1_score, accuracy

#### 🔗 Recent Results (June 2025)

| Metric             | Value   |
| ------------------ | ------- |
| **Accuracy**       | 91.25%  |
| **Precision**      | 63.87%  |
| **Recall**         | 100.0%  |
| **F1-score adj**   | 77.95%  |
| **ROC AUC**        | 98.17%  |

---

## 🚀 Model Dispatcher Logic

The `train.py` script dynamically dispatches to the appropriate training pipeline:

```bash
python train.py --env dev
python train.py --env prod --mlflow-ui
```

- If `dev` → Executes `train_sklearn.py`
- If `prod` → Executes `train_spark.py`

Central configuration is always loaded dynamically from environment YAML.

---

## 🏗️ Directory Structure

```bash
models/
├── train_sklearn.py   # Scikit-learn training pipeline (with calibration, tuning, rolling metrics)
├── train_spark.py     # Distributed Spark MLlib training pipeline
└── train.py           # Unified training dispatcher
```

---

## 🏅 Consolidated Improvements (June 2025 Release)

- ✅ Complete pipeline calibration (isotonic method)
- ✅ Dynamic threshold optimization (F1 maximization)
- ✅ SMOTETomek balancing for class imbalance
- ✅ Full GridSearch hyperparameter tuning (sklearn & Spark)
- ✅ MLflow artifact tracking and conditional model registry
- ✅ Daily rolling window evaluation for time-based stability
- ✅ Fully reproducible YAML-driven configurations
- ✅ Unified feature engineering across Spark and Scikit-learn (rolling stats & segmentation)

---

**Documentation updated: June 2025 — Synchronized Dev & Prod Pipelines ✅**
