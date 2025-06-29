# AI MLOps Project

[![CI/CD](https://github.com/Impesud/ai-mlops-project/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Impesud/ai-mlops-project/actions/workflows/ci-cd.yml)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Spark MLlib](https://img.shields.io/badge/Spark-MLlib-orange?logo=apache-spark)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?logo=scikit-learn)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Container-lightblue?logo=docker)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-black?logo=github-actions)
![Black](https://img.shields.io/badge/Black-Formatter-black?logo=python)
![Flake8](https://img.shields.io/badge/Flake8-Linter-red?logo=python)
![MyPy](https://img.shields.io/badge/MyPy-Type_Checking-informational?logo=python)
![Bandit](https://img.shields.io/badge/Bandit-Security-lightcoral?logo=python)
![Pre-Commit](https://img.shields.io/badge/Pre--Commit-Hooks-critical?logo=pre-commit)
![Semantic Release](https://img.shields.io/badge/Semantic--Release-Automation-success?logo=semantic-release)

---

## ✨ Latest Updates (June 2025)

- ✅ Advanced time-based feature engineering (Spark-driven)
- ✅ `train_sklearn.py`: Calibrated + threshold-optimized RandomForest
- ✅ `train_spark.py`: Distributed GBTClassifier + full evaluation
- ✅ Precision and F1 improvements through GridSearch and calibration
- ✅ MLflow artifact tracking fully structured and consistent
- ✅ YAML-driven hyperparameter config with rolling metric logs
- ✅ Added model signature, input examples, confusion matrices, ROC, PR curves
- ✅ CI/CD improved with Buildx caching, protected branches, and PAT for semantic-release
- ✅ Docker build hardened with retry logic and compressed layer caching
- ✅ Multiple PR templates + branch rules enforced in GitHub Actions
- ✅ OpenAI-powered LLM reporting integrated into CI pipeline
- ✅ Full Code Quality & Static Analysis (Black, Flake8, Isort, MyPy, Bandit)
- ✅ Pre-commit hooks enabled to catch issues before commits

---

## 🔍 Project Overview

AI MLOps Platform for **behavioral purchase prediction**, equipped with:

- 🚚 Data ingestion (local + AWS S3)
- 🔀 Spark-based feature engineering (time, recency, behavioral signals)
- ⚖️ Class imbalance handling (SMOTETomek, dynamic class weights)
- 🧠 Dual model engines: Scikit-learn (dev) & Spark MLlib (prod)
- 🔍 GridSearch & CrossValidator support
- 📈 MLflow experiment tracking + threshold-based metrics
- 📦 Docker-ready + CI/CD (GitHub Actions)
- 🤖 OpenAI API integration for automated AI reporting, with dynamic prompts and structured analysis of model results and business KPIs.

---

## 📃 Semantic Release

- Uses `@semantic-release` to automatically version and tag builds from main
- Integrated PAT token to bypass protection rules
- Full changelog and release notes updated via plugin

Command for dry-run:

```bash
npx semantic-release --dry-run
```

---

## 🔢 GitHub Workflows Summary

Main CI/CD file: `.github/workflows/ci-cd.yml`

Includes jobs:

- `build-and-test`: linting, type-checking, static analysis, unit tests, and pipeline logic
- `docker-build-and-push`: buildx + GHCR
- `model-smoke-test`: simple inference test
- `semantic-release`: versioning and release

Protected branches supported via PAT: `secrets.PAT_GITHUB`

---

## 🔹 Run Full Project (DEV & PROD)

```bash
# Create and activate environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# DEVELOPMENT
make ingest-dev-local
make process-dev-local
make train-dev-local
make test-dev

# PRODUCTION
make ingest-prod-local
make process-prod-local
make train-prod-local
make test-prod

# Launch MLflow UI
make mlflow-local
```

---

## 🔢 Model Pipelines

| Mode | Engine       | Pipeline Summary                                 |
| ---- | ------------ | ------------------------------------------------ |
| dev  | Scikit-learn | SMOTETomek + RF + Calibration + GridSearchCV     |
| prod | Spark MLlib  | GBTClassifier + VectorAssembler + CrossValidator |

Dispatcher:

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

## 📘️ Documentation

- 📊 [Data Processing](./docs/data_processing.md)
- 🔧 [Model Training](./docs/models.md)
- 📦 [MLflow Registry](./docs/mlflow_registry.md)
- 📦 [Dependencies](./docs/dependencies.md)
- ⚙️ [Code Quality & Static Analysis](./docs/code-quality.md)
- 🧪 [Github Workflow](./docs/github-workflow.md)
- 🧩 [Semantic Commit Guide](./docs/semantic-commit-guide.md)
- 📦 [Makefile Reference](./Makefile)

---

## 🔄 CI/CD Pipeline

- 🧪 Unit tests and validations (local + Docker-based)
- 📦 Model artifact verification
- 🔀 MLflow logging consistency checks
- 🐳 Docker image with Buildx caching
- 🌐 Semantic Release with PAT (protected branch support)
- 📌 PR Templates and branch rules enforced
- 🤖 OpenAI-integrated reporting (Generative AI)
- ✅ Code linting, static analysis & type-checking enforced via CI & pre-commit

---

## 🔮 Roadmap

- ⏳ Optuna optimization support
- ☁️ Cloud deployment (EMR, SageMaker)
- 🌐 MLflow + FastAPI model serving
- 🤖 LLM-powered monitoring

---

**Maintainer:** Erick Jara — CTO & AI/Data Engineer\
📧 [erick.jara@hotmail.it](mailto\:erick.jara@hotmail.it) | 🌐 GitHub: [Impesud](https://github.com/Impesud)




