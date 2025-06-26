# AI MLOps Project

[![CI/CD](https://github.com/Impesud/ai-mlops-project/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Impesud/ai-mlops-project/actions/workflows/ci-cd.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Apache Spark](https://img.shields.io/badge/Spark-MLlib-FDEE21?logo=apachespark&logoColor=black)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)

---

## ✨ Latest Updates (June 2025)

- ✅ Advanced time-based feature engineering (Spark-driven)
- ✅ `train_sklearn.py`: Calibrated + threshold-optimized RandomForest
- ✅ `train_spark.py`: Distributed GBTClassifier + full evaluation
- ✅ Precision and F1 improvements through GridSearch and calibration
- ✅ MLflow artifact tracking fully structured and consistent
- ✅ YAML-driven hyperparameter config with rolling metric logs
- ✅ Added model signature, input examples, confusion matrices, ROC, PR curves

---

## 🔍 Project Overview

AI MLOps Platform for **behavioral purchase prediction**, equipped with:

- 🚚 Data ingestion (local + AWS S3)
- 🔄 Spark-based feature engineering (time, recency, behavioral signals)
- ⚖️ Class imbalance handling (SMOTETomek, dynamic class weights)
- 🧠 Dual model engines: Scikit-learn (dev) & Spark MLlib (prod)
- 🔍 GridSearch & CrossValidator support
- 📈 MLflow experiment tracking + threshold-based metrics
- 📦 Docker-ready + CI/CD (GitHub Actions)
- 🤖 OpenAI API integration for automated AI reporting, with dynamic prompts and structured analysis of model results and business KPIs.

---

## 🔢 Current Model Results

### 🧪 `Scikit-learn` (Dev)

| Metric        | Value  |
| ------------- | ------ |
| Accuracy      | 71.84% |
| Precision     | 43.96% |
| Recall        | 67.99% |
| F1-score      | 53.39% |
| ROC AUC       | 77.33% |
| BestThreshold | 0.215  |

### 🧪 `Spark MLlib` (Prod)

| Metric        | Value  |
| ------------- | ------ |
| Accuracy      | 91.25% |
| Precision     | 63.87% |
| Recall        | 100.00% |
| F1-score      | 77.95% |
| ROC AUC       | 98.17% |

---

## ⚙️ Model Pipelines

| Mode | Engine       | Pipeline Summary                                       |
| ---- | ------------ | ------------------------------------------------------ |
| dev  | Scikit-learn | SMOTETomek + RF + Calibration + GridSearchCV           |
| prod | Spark MLlib  | GBTClassifier + VectorAssembler + CrossValidator       |

Unified Dispatcher:

```bash
python models/train.py --env dev
python models/train.py --env prod
```

---

## 🏗️ Project Structure

```bash
ai-mlops-project/
├── configs/           # Environment YAMLs, training parameters, MLflow config
├── data/              # Datasets (raw, intermediate, processed) 
├── data_ingestion/    # Scripts for data download, loading, and storage
├── data_processing/   # Spark-based feature engineering and preprocessing pipelines
├── docs/              # Markdown documentation, diagrams, architecture notes
├── generative_ai/     # LLM integrations, prompts, embedding pipelines, NLP tools
├── mlruns/            # MLflow experiment logs and artifacts (auto-generated)
├── models/            # Sklearn & Spark training pipelines, model classes, training logic
├── scripts/           # CLI scripts for training, evaluation, deployment
├── tests/             # Pytest-based unit and integration tests
├── Makefile           # Full pipeline automation: make train, make deploy, make test, etc.
└── .github/           # CI/CD workflows, CODEOWNERS, and GitHub Actions
```

---

## 🚀 Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

make train-dev-local   # Local sklearn pipeline
make train-prod-local  # Distributed Spark training
make mlflow-local      # MLflow UI
```

---

## 📘 Documentation

- 📊 [Data Processing](./docs/data_processing.md)
- 🔧 [Model Training](./docs/models.md)
- 📦 [MLflow Registry](./docs/mlflow_registry.md)
- 📦 [Dependencies](./docs/dependencies.md)

---

## 🔄 CI/CD Pipeline

- 🧪 Unit tests and validations
- 📦 Model artifact verification
- 🔁 MLflow logging consistency checks
- 🐳 Docker image

---

## 🧭 Roadmap

- ⏳ Optuna optimization support
- ☁️ Cloud deployment (EMR, SageMaker)
- 🌐 MLflow + FastAPI model serving
- 🤖 LLM-powered monitoring

---

**Maintainer:** Erick Jara — CTO & AI/Data Engineer  
📧 [erick.jara@hotmail.it](mailto:erick.jara@hotmail.it) | 🌐 GitHub: [Impesud](https://github.com/Impesud)





