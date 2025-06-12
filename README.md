# AI MLOps Project

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Spark-3.5.5-orange)](https://spark.apache.org/)
[![Docker](https://img.shields.io/badge/docker-20.10-blue)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.6.2-green)](https://mlflow.org/)
[![LLM](https://img.shields.io/badge/GenerativeAI-OpenAI-blueviolet)](https://openai.com/)
[![Build](https://github.com/impesud/ai-mlops-project/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/impesud/ai-mlops-project/actions/workflows/ci-cd.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/impesud/ai-mlops-project/blob/main/LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/impesud/ai-mlops-project)](https://github.com/impesud/ai-mlops-project/commits/main)
[![Platform](https://img.shields.io/badge/platform-Ubuntu-blue)]()

🚀 **AI MLOps Project** – A production-grade MLOps pipeline for scalable, reproducible, and cloud-ready machine learning with Spark, scikit-learn, MLflow, and LLM-powered analytics.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [What Does This Project Do?](#what-does-this-project-do)
3. [Type of Model and Objective](#type-of-model-and-objective)
4. [Algorithms & Models](#algorithms--models)
5. [Environments: dev vs prod](#environments-dev-vs-prod)
6. [Requirements](#requirements)
7. [Setup & Installation](#setup--installation)
8. [Configuration](#configuration)
9. [Pipeline Usage](#pipeline-usage)
10. [MLflow Tracking & Model Registry](#mlflow-tracking--model-registry)
11. [Testing](#testing)
12. [CI/CD Pipeline](#cicd-pipeline)
13. [Next Steps](#next-steps)
14. [License](#license)

---

## Project Overview

- **Data ingestion** from S3/local using Apache Spark (batch or streaming)
- **Feature engineering** in Pandas or Spark, with support for class imbalance (class weights, SMOTE)
- **Model training** with Spark MLlib (GBTClassifier, RandomForestClassifier, LogisticRegression) or scikit-learn (RandomForest)
- **Configurable pipeline** via YAML files for dev/prod environments
- **Hyperparameter tuning** via grid search and cross-validation
- **Full MLflow tracking**: parameters, metrics (accuracy, F1, ROC AUC, confusion matrix), models, artifacts
- **Model versioning and auto-registration** based on best F1-score
- **Automated LLM-based reporting** (OpenAI)
- **Dockerized** for local and cloud deployment
- **CI/CD** with GitHub Actions
- **Cloud extensible**: AWS EMR, SageMaker, Databricks

---

## What Does This Project Do?

This project implements a **complete MLOps pipeline for binary classification on tabular data**, with a focus on predicting user "purchase" events.  
It automates all steps from data ingestion to model training, evaluation, registration, and reporting, making it easy to scale from local development to production and cloud.

---

## Type of Model and Objective

- **Type of model:** Binary classifier (purchase vs. non-purchase)
- **Objective:** Predict whether a user action is a "purchase" or not, using features such as transaction value, hour, day of week, and month.
- **Target metric:** F1-score and recall on the minority class ("purchase"), with support for class imbalance handling.
- **Model registry:** Only models that outperform previous versions (by F1-score) are registered in MLflow Model Registry.
- **Automated reporting:** After each run, an LLM (e.g., OpenAI GPT) generates a markdown report summarizing data, model performance, issues, and recommendations.

---

## Algorithms & Models

**Supported algorithms:**
- **Spark MLlib**
  - `GBTClassifier` (Gradient-Boosted Trees, default for prod)
  - `RandomForestClassifier`
  - `LogisticRegression`
- **scikit-learn**
  - `RandomForestClassifier` (default for dev)
  - (Easily extendable to XGBoost, LightGBM, etc.)

**Model selection and hyperparameters** are fully configurable via YAML files.

**Model registry:**  
- Models are automatically registered in MLflow Model Registry if they outperform previous versions (by F1-score).
- Both Spark and scikit-learn models are tracked and versioned.

---

## Environments: dev vs prod

| Environment | Data Source         | Model         | Config File                   | Typical Use         |
|-------------|---------------------|--------------|-------------------------------|---------------------|
| dev         | `sample_1k.csv`     | sklearn RF   | `configs/dev/model.yaml`      | Fast local testing  |
| prod        | S3 Parquet/CSV      | Spark GBT    | `configs/prod/model.yaml`     | Full-scale training |

**Switch environment** by passing `dev` or `prod` to scripts or CLI.

---

## Requirements

- **Python** 3.8+ (3.10 or 3.11 recommended)
- **Java** 8+ (for Apache Spark)
- **Docker**
- **MLflow** CLI (`pip install mlflow`)
- **AWS CLI** / GCP SDK / Azure CLI (for cloud)
- **Git**
- **OpenAI API key** (for LLM reporting)

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/impesud/ai-mlops-project.git
cd ai-mlops-project

# Create and activate virtualenv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools
pip install -r requirements.txt

# Configure AWS profile (if using S3/EMR)
# Edit ~/.aws/config and export AWS_PROFILE
export AWS_PROFILE=<AWS_user>

# Set OpenAI key (for LLM reporting)
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
source ~/.bashrc  # or ~/.zshrc

# (Optional) Build Docker image
docker build -t ai-mlops-project .
```

---

## Configuration

- **Dev config:** `configs/dev/model.yaml`
- **Prod config:** `configs/prod/model.yaml`

**Example YAML:**
```yaml
input_path: "s3a://my-mlops-processed-data/prod/"
test_size: 0.2
random_seed: 42
features:
  - value
  - hour
  - day_of_week
  - month
model_params:
  n_estimators: 40      # For sklearn RF
  max_depth: 10
  stepSize: 0.05
grid:
  maxIter: [40, 60]     # For Spark GBT
  maxDepth: [5, 10]
  stepSize: [0.05, 0.1]
numFolds: 2
parallelism: 2
```
- **All pipeline steps, features, and hyperparameters are controlled via YAML.**
- The pipeline automatically maps `n_estimators`/`max_depth` to Spark's `maxIter`/`maxDepth` when needed.

---

## Pipeline Usage

### Run the full pipeline (dev or prod)

```bash
# Dev mode (local, fast, sklearn)
./scripts/run_pipeline.sh dev

# Prod mode (Spark, S3, full data)
./scripts/run_pipeline.sh prod
```

### Manual stage execution

```bash
# Data ingestion (Spark)
./scripts/run_ingest.sh dev
./scripts/run_ingest.sh prod

# Model training (Spark or sklearn)
./scripts/run_train.sh dev
./scripts/run_train.sh prod

# Generate LLM report
python generative_ai/generate.py --prompt "Data analysis"
```

### Run with Docker

```bash
docker run --rm -it \
  -e ENV=prod \
  -e AWS_PROFILE=$AWS_PROFILE \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -v $(pwd):/workspace \
  ai-mlops-project ./scripts/run_pipeline.sh prod
```

---

## MLflow Tracking & Model Registry

### Start MLflow UI

```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```
Access: [http://localhost:5000](http://localhost:5000)

### What is tracked

- **Parameters:** All model and pipeline params from YAML
- **Metrics:** Accuracy, F1, ROC AUC, confusion matrix, precision, recall
- **Artifacts:** Model, signature, LLM report, plots
- **Model registry:**  
  - Models are auto-registered if they outperform previous F1-score
  - Both Spark and sklearn models are versioned

### Example MLflow UI

![MLflow Overview](./docs/images/mlflow-01.png)
![MLflow Overview](./docs/images/mlflow-02.png)

---

## Testing

Automated tests are in `tests/` and run with `pytest`.

| File             | Validates                                   |
|------------------|---------------------------------------------|
| `test_ingest.py` | Data ingestion and Parquet generation       |
| `test_train.py`  | Model training, logging, and MLflow output  |

Run all tests:
```bash
pytest tests/
```

---

## CI/CD Pipeline

- **GitHub Actions**:  
  - Python env setup, dependency install
  - Data ingestion, model training, MLflow tracking
  - Unit tests with `pytest`
  - Model artifact upload
  - Docker image build and push
  - (Optional) Model serving test with `mlflow models serve`

---

## Next Steps

- 🧠 Advanced feature engineering (user-level, time-based, aggregations)
- 🎯 Hyperparameter tuning with Optuna/GridSearchCV
- 📈 Log ROC and precision-recall curves in MLflow
- 🔐 Enhanced CI/CD (SonarQube, multi-env)
- ☁️ Full cloud deployment scripts (AWS EMR, SageMaker, Databricks)
- 🛰️ Model serving via FastAPI or `mlflow models serve`
- 📊 Automated LLM-based data and model reporting


---

[![Contribute](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Impesud/ai-mlops-project/blob/main/CONTRIBUTING.md)

---


## Contact

Created by Erick Jara – CTO & Senior AI/Data Engineer.
GitHub: [Impesud](https://github.com/Impesud) | Email: erick.jara@hotmail.it






