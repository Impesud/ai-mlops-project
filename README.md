# AI MLOps Project

---

## ✨ Latest Updates (June 2025)

- ✅ Full refactor of model training pipelines ([see Models Documentation](./docs/models.md))
- ✅ Full behavioral feature engineering extension ([see Data Processing](./docs/data_processing.md))
- ✅ Advanced calibration pipeline (Isotonic Calibration)
- ✅ Automatic optimal threshold selection (Precision-Recall curve)
- ✅ Rolling 7-day evaluation window for stability monitoring
- ✅ Full MLflow integration ([see MLflow Integration](./docs/mlflow_registry.md))
- ✅ YAML-driven modular architecture for full reproducibility
- ✅ Fully operational Makefile added for streamlined execution

---

## 🔍 Project Overview

AI MLOps Platform for **behavioral purchase prediction** models with full MLOps automation:

- 🚚 Data ingestion (local & cloud)
- 🔄 Spark-based feature engineering (time, recency, frequency, behavioral aggregates)
- 🔹 Imbalanced learning handling (SMOTE-Tomek, class weights)
- 💡 Dual model engines: Spark MLlib (prod) & scikit-learn (dev)
- 🔄 Full hyperparameter search integration
- 🔹 MLflow experiment tracking and model versioning
- 🔹 Rolling stability metrics (time-aware validation)
- 🔹 Cloud-ready (AWS S3, EMR, SageMaker supported)
- 🔹 Fully containerized (Docker)
- 📁 Complete CI/CD automation (GitHub Actions)

---

## 🔠 Current Model Results (Sklearn Dev Mode)

| Metric             | Value  |
| ------------------ | ------ |
| **Accuracy**       | 88.85% |
| **Precision**      | 74.73% |
| **Recall**         | 80.14% |
| **F1-score**       | 77.33% |
| **ROC AUC**        | 93.44% |
| **Best Threshold** | 0.457  |

These results reflect major improvements after feature engineering extension, full calibration, dynamic thresholding and hyperparameter optimization.

---

## 🔀 Training Pipelines

| Mode | Engine       | Description                                              | Entry Point            |
| ---- | ------------ | -------------------------------------------------------- | ----------------------- |
| dev  | Scikit-learn | Random Forest + SMOTETomek + Calibration + GridSearchCV | `models/train_sklearn.py` |
| prod | Spark MLlib  | GBTClassifier + CrossValidator (distributed Spark ML)    | `models/train_spark.py`   |

**Unified Dispatcher Logic:**

```bash
python models/train.py --env dev
python models/train.py --env prod --mlflow-ui
```

---

## 🏢 Project Structure

```bash
ai-mlops-project/
├── data_processing/   # Spark-based ETL and feature engineering
├── models/            # Full model training pipelines (sklearn & Spark)
├── configs/           # YAML configs for environments
├── docs/              # Documentation (data, models, mlflow)
├── mlruns/            # MLflow tracking logs
├── tests/             # Unit tests
├── scripts/           # Execution orchestration scripts
└── Makefile           # Fully integrated Makefile commands
```

---

## 🛠️ Quickstart Commands

```bash
# Create virtual environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Full dev pipeline (sklearn)
make train-dev

# Full prod pipeline (Spark)
make train-prod

# Start MLflow UI
make mlflow
```

The Makefile now wraps all major operations for simplified reproducibility.

---

## 🔍 Full Documentation

- 🔹 [Data Processing & Feature Engineering](./docs/data_processing.md)
- 🔹 [Models Training Pipelines](./docs/models.md)
- 🔹 [MLflow Integration Details](./docs/mlflow_registry.md)

---

## 🔄 CI/CD Pipeline (GitHub Actions)

- Build and test environment setup
- Data ingestion tests
- Model training validations
- MLflow model artifact handling
- Docker image build (optional extension)

---

## 🔹 Upcoming Roadmap

- Optuna-based hyperparameter search (planned)
- Cloud deployment automation (EMR, SageMaker)
- Full model serving via MLflow Serve + FastAPI
- Automated LLM-powered monitoring & reporting
- Full production-ready CI/CD deployment pipeline

---

**Maintainer:** Erick Jara — CTO & AI/Data Engineer
GitHub: [Impesud](https://github.com/Impesud) • Contact: [erick.jara@hotmail.it](mailto:erick.jara@hotmail.it)






