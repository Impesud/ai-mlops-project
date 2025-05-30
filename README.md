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

🚀 **AI MLOps Project** – Pipeline completa di MLOps con Big Data, Spark, MLflow e Generative AI  
Integrazione pronta per la produzione: ingestion da S3 con Spark, training con MLflow tracking, report generativo LLM (OpenAI), test automatici con pytest, e CI/CD via GitHub Actions e Docker.

✅ Compatibile con Ubuntu + GitHub Actions  
✅ Logging completo su MLflow (parametri, metriche, modelli, artifact)  
✅ Estensibile a cloud (AWS, Azure, GCP)  
✅ Docker-ready con build automatizzato e deploy via GitHub Actions

---

## 📋 Table of Contents

1. [Modello e pipeline di addestramento](#-modello-e-pipeline-di-addestramento)  
2. [Prerequisiti](#-prerequisiti)  
3. [Setup su Ubuntu](#-setup-su-ubuntu)  
4. [Componenti](#-componenti)  
5. [Esempio di utilizzo](#-esempio-di-utilizzo)  
6. [MLflow UI](#-mlflow-ui)  
7. [MLOps e Tracking Server](#-mlops-e-tracking-server)  
8. [Testing](#-testing)  
9. [CI/CD Pipeline](#-cicd-pipeline)  
10. [Prossimi passi](#-prossimi-passi)  
11. [Licenza](#-licenza)

---

## 🔍 Modello e Pipeline di Addestramento

Partendo da un file CSV sintetico, il progetto esegue un job batch di ingestione tramite Apache Spark e successivamente una pipeline di feature engineering in Pandas. Il dato preprocessato viene utilizzato per addestrare un classificatore Random Forest su un problema di classificazione binaria: acquisto vs non-acquisto (purchase vs non-purchase).

Il modello predice, per ogni evento con le sue feature (inclusi value e timestamp), se si tratta di un acquisto (1) o un'altra azione (0, es. click/view/signup). I risultati vengono tracciati in MLflow e i report generati tramite OpenAI.

---

## 🔧 Prerequisiti

* **Python** 3.8+ (consigliato 3.10 o 3.11)
* **Java** 8+ (per Apache Spark)
* **Docker**
* **MLflow** CLI (`pip install mlflow`)
* **AWS CLI** / GCP SDK / Azure CLI
* **Git**
* **OpenAI API key** (per IA generativa)

---

## ⚙️ Setup su Ubuntu

1. **Clona il repository**
   ```bash
   git clone https://github.com/impesud/ai-mlops-project.git
   cd ai-mlops-project
   ```

2. **Crea e attiva il virtualenv**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools
   pip install -r requirements.txt
   ```

3. **Configura AWS Profile**
   Crea/modifica `~/.aws/config`:
   ```ini
   [<AWS_user>]
   region = eu-central-1
   ```
   Esporta la variabile d'ambiente:
   ```bash
   export AWS_PROFILE=<AWS_user>
   ```

4. **Imposta la chiave OpenAI**  
   Aggiungi in `~/.bashrc` o `~/.zshrc`:
   ```bash
   export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
   ```
   Poi esegui:
   ```bash
   source ~/.bashrc
   ```

5. **Aggiorna `config.yaml`**
   Modifica `data_ingestion/config.yaml`:
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
     prompt: "Analisi dei dati"
     output_path: "report.txt"
   ```

---

## 🧩 Componenti

* `data_ingestion/`: Spark batch/streaming, pulizia dati, scrittura su S3 e locale  
* `data_processing/`: notebook + trasformazioni dati  
* `models/`: training (train.py), SMOTE, feature engineering, tracking MLflow  
* `scripts/`: orchestratore `pipeline.py` (ingestion, training, AI, logging)  
* `mlops/`: `entrypoint.sh` per MLflow Tracking Server  
* `generative_ai/`: `generate.py` per report LLM (OpenAI)  
* `.github/workflows/`: CI/CD e testing automatico

---

## 🎯 Esempio di utilizzo

```bash
# Attiva virtualenv
source venv/bin/activate

# Pipeline completa
python scripts/pipeline.py

# Step separati
bash scripts/run_ingest.sh
bash scripts/run_train.sh

# Genera report AI
python generative_ai/generate.py --prompt "Analisi dei dati" --output report.txt
```

---

## 📊 MLflow UI

Avvia MLflow UI:
```bash
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Accedi su: [http://localhost:5000](http://localhost:5000)  
Visualizza:
1. Parametri e metriche  
2. Modello e firma  
3. Artifact `report.txt` generato da LLM

---

## ⚙️ MLOps e Tracking Server

Avvia Tracking Server MLflow:
```bash
bash mlops/entrypoint.sh
```
Accedi a:
```bash
http://localhost:5000
```
Può essere integrato in container Docker o esposto su rete locale.

---

## 🧪 Testing

La cartella `test/` contiene test automatizzati con `pytest`.

| File             | Verifica                                      |
|------------------|-----------------------------------------------|
| `test_ingest.py` | Generazione file Parquet                      |
| `test_train.py`  | Logging e artifact in MLflow                  |

Esegui tutti i test:
```bash
pytest test/
```

---

## 🔁 CI/CD Pipeline

La pipeline GitHub Actions esegue:

1. Installazione ambiente Python e dipendenze
2. Ingestione dati + training del modello
3. Tracking completo su MLflow
4. Test automatici con pytest
5. Upload artifact modello addestrato
6. Build e push immagine Docker su Docker Hub
7. Deploy e test endpoint di inferenza locale con `mlflow models serve`

---

## 🚀 Prossimi passi

1. Feature engineering avanzata (ora, giorno, stats utente)  
2. Tuning iperparametri con Optuna  
3. Containerizzazione + Helm + deploy Docker/K8s  
4. Prompt dinamici + salvataggio nei log  
5. CI/CD con SonarQube + ambienti  
6. Model serving con `mlflow models serve` o FastAPI

---

## 📜 Licenza

MIT © 2025 Erick Jara - Impesud

✍️ Attribution:
If you use this project, please mention:
"Based on the AI MLOps Project by Erick Jara - Impesud".


