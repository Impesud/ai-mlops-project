# AI MLOps Project

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Spark-3.5.5-orange)](https://spark.apache.org/)
[![Docker](https://img.shields.io/badge/docker-20.10-blue)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.6.2-green)](https://mlflow.org/)

Template di progetto per integrazione di Big Data, Data Cloud, MLOps e IA Generativa, **production-ready**.

---

## 📋 Table of Contents

1. [Prerequisiti](#-prerequisiti)
2. [Setup su Windows](#-setup-su-windows)
3. [Componenti](#-componenti)
4. [Esempio di utilizzo](#-esempio-di-utilizzo)
5. [MLflow UI](#mlflow-ui)
6. [Punti mancanti e prossimi passi](#punti-mancanti-e-prossimi-passi)
7. [Licenza](#licenza)

---

## 🔧 Prerequisiti

* **Python** 3.8+ (consigliato 3.10 o 3.11)
* **Java** 8+ (per Apache Spark)
* **Docker**
* **MLflow** CLI (`pip install mlflow`)
* **AWS CLI** / GCP SDK / Azure CLI
* **Git**

## ⚙️ Setup su Windows

1. **Clona il repository**

   ```bat
   git clone https://github.com/impesud/ai-mlops-project.git
   cd ai-mlops-project
   ```
2. **Crea e attiva il virtualenv**

   ```bat
   python -m venv venv
   call venv\Scripts\activate
   pip install --upgrade pip setuptools
   pip install -r requirements.txt
   ```
3. **Configura AWS Profile**

   * Crea/modifica `C:\Users\<TUO_UTENTE>\.aws\config`:

     ```ini
     [<AWS_user>]
     region = eu-central-1
     ```
   * Esporta il profilo:

     ```bat
     setx AWS_PROFILE <AWS_user>
     ```
4. **Aggiorna `config.yaml`**
   Modifica `data_ingestion/config.yaml`:

   ```yaml
   format: csv
   path: s3a://my-mlops-raw-data/
   output_path: s3a://my-mlops-processed-data/
   aws:
     region: eu-central-1
   ```

---

## 🧩 Componenti

* **data_ingestion/**: Spark batch e streaming, pulizia dati, scrittura Parquet.
* **data_processing/**: script e notebook per pulizia/trasformazione.
* **models/**: training (`train.py`) con SMOTE, class weights e MLflow.
* **scripts/**: orchestratore end-to-end (`pipeline.py`, batch `.bat`).
* **mlops/**: Dockerfile, `entrypoint.bat`, IaC e manifest CI/CD.
* **generative_ai/**: script `generate.py` per reportistica LLM.
* **.github/workflows/**: pipeline GitHub Actions per CI/CD.

---

## 🎯 Esempio di utilizzo

```bat
:: 1) Attiva virtualenv
call venv\Scripts\activate

:: 2) Esegui pipeline completa
python scripts\pipeline.py

:: 3) Step separati
call scripts\run_ingest.bat    :: ingest Spark
call scripts\run_train.bat     :: train & MLflow UI

:: 4) Generative AI report
python generative_ai\generate.py --prompt "Analisi dei dati" --output report.txt
```

---

## 📊 MLflow UI

Avvia la UI:

```bat
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Accedi: [http://localhost:5000](http://localhost:5000) e seleziona l’esperimento **my-experiment**.

---

## 🚀 Prossimi passi

1. **Feature engineering avanzato**: ora/giorno, weekend, aggregazioni per user_id.
2. **Hyperparameter tuning**: Grid/RandomizedSearchCV su recall/F1 per class 1.
3. **Deployment**: script `mlops/entrypoint.bat`, Docker image, Kubernetes Helm charts.
4. **Generative AI full integration**: pipeline con prompt dinamici e artifact MLflow.
5. **CI/CD**: completare test unitari, sonar scan, deploy in staging/prod.

---

## 📜 Licenza

MIT © 2025 impesud




