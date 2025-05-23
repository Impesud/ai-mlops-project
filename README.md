# AI MLOps Project

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![Spark](https://img.shields.io/badge/Spark-3.5.5-orange)](https://spark.apache.org/)
[![Docker](https://img.shields.io/badge/docker-20.10-blue)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-2.6.2-green)](https://mlflow.org/)
[![Tests](https://github.com/impesud/ai-mlops-project/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/impesud/ai-mlops-project/actions/workflows/ci-cd.yml)

Template di progetto per integrazione di Big Data, Data Cloud, MLOps e IA Generativa, **production-ready**.

---

## 📋 Table of Contents

1. [Prerequisiti](#-prerequisiti)
2. [Setup su Windows](#-setup-su-windows)
3. [Componenti](#-componenti)
4. [Esempio di utilizzo](#-esempio-di-utilizzo)
5. [MLflow UI](#mlflow-ui)
6. [MLOps e Tracking Server](#mlops-e-tracking-server)
7. [Testing](#testing)
8. [Punti mancanti e prossimi passi](#punti-mancanti-e-prossimi-passi)
9. [Licenza](#licenza)

---

## 🔧 Prerequisiti

* **Python** 3.8+ (consigliato 3.10 o 3.11)
* **Java** 8+ (per Apache Spark)
* **Docker**
* **MLflow** CLI (`pip install mlflow`)
* **AWS CLI** / GCP SDK / Azure CLI
* **Git**
* **OpenAI API key** (per IA generativa)

## ⚙️ Setup su Ubuntu

1. **Clona il repository**

   ```bat
   git clone https://github.com/impesud/ai-mlops-project.git
   cd ai-mlops-project
   ```
2. **Crea e attiva il virtualenv**

   ```bat
   python3 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip setuptools
   pip install -r requirements.txt
   ```
3. **Configura AWS Profile**

   * Crea/modifica ~/.aws/config:

     ```ini
     [<AWS_user>]
     region = eu-central-1
     ```
   * Esporta la variabile d'ambiente:

     ```bat
     setx AWS_PROFILE <AWS_user>
     ```
4. **Imposta la chiave OpenAI**

Aggiungi in ~/.bashrc o ~/.zshrc:

   ```bat
   export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
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

* **data_ingestion/: Spark batch e streaming, pulizia dati, scrittura su S3 e locale.
* **data_processing/**: data_processing/: notebook e script di trasformazione.
* **models/**: training (train.py) con SMOTE, feature engineering, salvataggio e firma modello MLflow.
* **scripts/**: orchestratore (pipeline.py) per ingestion, training, AI generativa e logging.
* **mlops/**: script entrypoint.sh per avviare un MLflow Tracking Server locale o containerizzato.
* **generative_ai/**: generate.py usa LLM (OpenAI) per generare report analitici.
* **.github/workflows/**: pipeline GitHub Actions per test e CI/CD.

---

## 🎯 Esempio di utilizzo

```bat
# 1) Attiva virtualenv
source venv/bin/activate

# 2) Esegui pipeline completa
python scripts/pipeline.py

# 3) Step separati
bash scripts/run_ingest.sh    # ingest Spark
bash scripts/run_train.sh     # train & MLflow UI

# 4) Generative AI report
python generative_ai/generate.py --prompt "Analisi dei dati" --output report.txt
```

---

## 📊 MLflow UI

Avvia la UI:

```bat
mlflow ui --backend-store-uri ./mlruns --port 5000
```

Accedi: [http://localhost:5000](http://localhost:5000) e seleziona l’esperimento **my-experiment**:
1. **Parametri e metriche**
2. **Firma del modello**
3. **Artifact report.txt generato dinamicamente**

---

## ⚙️ MLOps e Tracking Server
La cartella mlops/ contiene strumenti per lanciare un server MLflow centralizzato.

**mlops/entrypoint.sh**

Script per avviare un server MLflow:
```bat
bash mlops/entrypoint.sh
```

Visualizza il server all’indirizzo:
```bat
http://localhost:5000
```

Può essere usato anche in container Docker o su rete interna per più client.

---

## 🧪 Testing

La cartella test/ contiene test automatizzati con pytest.

| File             | Testa cosa                                          |
| ---------------- | --------------------------------------------------- |
| `test_ingest.py` | Che i file Parquet vengano generati correttamente   |
| `test_train.py`  | Che il modello venga loggato su MLflow con artifact |

Lancia tutti i test:

```bat
pytest test/
```

---

## 🚀 Prossimi passi

1. **Feature engineering avanzato**: estrazione di giorno, ora, weekend, e user-level stats.
2. **Hyperparameter tuning**: ricerca su F1/recall via GridSearch o Optuna.
3. **Deployment**: containerizzazione, Helm chart, e deploy con MLflow su Docker/Kubernetes.
4. **Generative AI full integration**: integrazione di prompt personalizzati con salvataggio in artifact.
5. **CI/CD avanzato**: test automatici, linting, SonarQube e deploy in ambienti reali.
6. **Model Serving**: integrazione con mlflow models serve o FastAPI.

---

## 📜 Licenza

MIT © 2025 impesud




