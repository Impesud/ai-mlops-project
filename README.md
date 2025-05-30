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

🚀 **AI MLOps Project** – A production-grade MLOps pipeline combining Big Data processing, scalable training workflows, and LLM-powered analytics.

This end-to-end solution features:  
- Data ingestion from S3 using Apache Spark  
- Model training with full MLflow tracking (parameters, metrics, models, artifacts)  
- Automated data reporting using OpenAI's Generative AI  
- Testing via `pytest`  
- CI/CD workflows powered by GitHub Actions  
- Dockerized deployment pipeline

✅ Fully compatible with Ubuntu + GitHub Actions  
✅ Cloud extensible (AWS, Azure, GCP)  
✅ Docker-ready with automated image build and deployment  
✅ Modular design for fast customization and reuse

---

## 📋 Table of Contents

1. [Model and Training Pipeline](#-model-and-training-pipeline)  
2. [Requirements](#-requirements)  
3. [Ubuntu Setup](#-ubuntu-setup)  
4. [Components](#-components)  
5. [Usage Example](#-usage-example)  
6. [MLflow UI](#-mlflow-ui)  
7. [MLOps and Tracking Server](#-mlops-and-tracking-server)  
8. [Testing](#-testing)  
9. [CI/CD Pipeline](#-cicd-pipeline)  
10. [Next Steps](#-next-steps)  
11. [License](#-license)

---

## 🔍 Model and Training Pipeline

Starting from a synthetic CSV file, the project performs batch ingestion with Apache Spark, followed by feature engineering in Pandas. The preprocessed data is used to train a Random Forest classifier on a binary classification task: purchase vs. non-purchase.

The model predicts, for each event with its features (including value and timestamp), whether it's a purchase (1) or another action (0, e.g., click/view/signup). Results are logged with MLflow, and reports are generated via OpenAI.

---

## 🔧 Requirements

* **Python** 3.8+ (3.10 or 3.11 recommended)  
* **Java** 8+ (for Apache Spark)  
* **Docker**  
* **MLflow** CLI (`pip install mlflow`)  
* **AWS CLI** / GCP SDK / Azure CLI  
* **Git**  
* **OpenAI API key** (for generative AI)

---

## ⚙️ Ubuntu Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/impesud/ai-mlops-project.git
   cd ai-mlops-project
   ```

2. **Create and activate virtualenv**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools
   pip install -r requirements.txt
   ```

3. **Configure AWS profile**  
   Edit `~/.aws/config`:
   ```ini
   [<AWS_user>]
   region = eu-central-1
   ```
   Export environment variable:
   ```bash
   export AWS_PROFILE=<AWS_user>
   ```

4. **Set OpenAI key**  
   Add in `~/.bashrc` or `~/.zshrc`:
   ```bash
   export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
   ```
   Then run:
   ```bash
   source ~/.bashrc
   ```

5. **Update `config.yaml`**  
   Edit `data_ingestion/config.yaml`:
   ```yaml
   format: csv
   path: s3a://my-mlops-raw-data/
   output_path: s3a://my-mlops-processed-data/
   local_output_path: data/processed
   aws:
     region: eu-central-1
   test_size: 0.2
   random_seed: 42
   model_params:
     n_estimators: 100
     max_depth: 5
   generative_ai:
     enabled: true
     prompt: "Data analysis"
     output_path: "report.txt"
   ```

---

## 🧩 Components

* `data_ingestion/`: Spark batch/streaming, data cleaning, write to S3/local  
* `data_processing/`: notebooks + transformations  
* `models/`: training (`train.py`), SMOTE, feature engineering, MLflow tracking  
* `scripts/`: orchestrator `pipeline.py` (ingestion, training, AI, logging)  
* `mlops/`: `entrypoint.sh` for MLflow Tracking Server  
* `generative_ai/`: `generate.py` for LLM reports (OpenAI)  
* `.github/workflows/`: CI/CD and automated testing

---

## 🎯 Usage Example

```bash
# Activate virtualenv
source venv/bin/activate

# Full pipeline
python scripts/pipeline.py

# Separate steps
bash scripts/run_ingest.sh
bash scripts/run_train.sh

# Generate AI report
python generative_ai/generate.py --prompt "Data analysis" --output report.txt
```

---

## 📊 MLflow UI

Launch MLflow UI:
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Access: [http://localhost:5000](http://localhost:5000)  
View:
1. Parameters and metrics  
2. Model and signature  
3. `report.txt` artifact generated by LLM

---

## ⚙️ MLOps and Tracking Server

Start MLflow Tracking Server:
```bash
bash mlops/entrypoint.sh
```
Access:
```bash
http://localhost:5000
```
Can be used in Docker containers or local networks.

---

## 🧪 Testing

The `test/` folder includes automated tests with `pytest`.

| File             | Validates                                   |
|------------------|---------------------------------------------|
| `test_ingest.py` | Generation of Parquet files                |
| `test_train.py`  | Model logging and artifact in MLflow       |

Run all tests:
```bash
pytest test/
```

---

### 🔁 **CI/CD Pipeline**

The GitHub Actions pipeline executes the following steps:

- ✅ Set up Python environment and install all dependencies  
- 🔄 Run full data ingestion and model training workflow  
- 📊 Track parameters, metrics, models, and artifacts using MLflow  
- 🧪 Execute unit tests with `pytest`  
- ☁️ Upload the trained model artifact to GitHub Actions  
- 🐳 Build and push the Docker image to Docker Hub  
- 📦 Deploy and test the inference endpoint using `mlflow models serve`

---

### 🚀 **Next Steps**

- 🧠 Advanced feature engineering (hour, weekday, user-level stats)  
- 🎯 Hyperparameter tuning via Optuna or GridSearchCV  
- ⚙️ Containerization + Helm charts for Docker/Kubernetes deployment  
- ✨ Dynamic prompt generation + structured artifact logging  
- 🔐 Enhanced CI/CD with SonarQube and multi-environment pipelines  
- 📡 Model serving via `mlflow models serve` or FastAPI + REST API

---

## 📜 License

MIT © 2025 Erick Jara - Impesud

✍️ Attribution:  
If you use this project, please mention  
"Based on the AI MLOps Project by Erick Jara - Impesud".



