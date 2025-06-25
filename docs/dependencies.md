# 📦 Module Dependencies — AI MLOps Project

This document outlines the **functional dependencies** between the key modules and directories of the project, as derived from the actual folder structure.

---

## 🔄 High-Level Dependency Graph

```
configs ─────► data_ingestion ────► data_processing ────► models
       │                                │                      │
       └────► generative_ai             └────► utils           ▼
                          ▲                                 mlruns
                          └────────────── scripts ────────────────────┐
                                                  │
                                                Makefile
```

---

## 🧹 Module-by-Module Breakdown

### `configs/`

- Contains YAML files for all environments (dev/prod/cloud).
- Used by: `data_ingestion/`, `data_processing/`, `models/`, `scripts/`, `generative_ai/`.

---

### `data_ingestion/`

- Ingests data from local or cloud (e.g., S3).
- Outputs intermediate parquet or delta files.
- Uses: `configs/`, `utils/`
- Feeds: `data_processing/`

---

### `data_processing/`

- Feature engineering with Spark (time features, behavioral signals).
- Uses: `configs/`, `utils/`
- Feeds: `models/`

---

### `models/`

- Contains training pipelines:
  - `train_sklearn.py` → Dev mode (local)
  - `train_spark.py` → Prod mode (distributed)
- Uses: `data_processing/`, `configs/`, `utils/`, `mlruns/`
- Outputs: model artifacts, rolling metrics, MLflow logs

---

### `generative_ai/`

- Integrates GPT (e.g., OpenAI) for:
  - Reporting model results
  - Generating summaries
  - Interacting with trained models
- Uses: `configs/`, `mlruns/`, `models/`
- Feeds into: `scripts/`, `docs/`

---

### `utils/`

- Shared functions:
  - YAML/IO parsers
  - Logging utilities
  - Evaluation metrics
- Used by: nearly all modules (`data_ingestion/`, `data_processing/`, `models/`, `scripts/`, `generative_ai/`)

---

### `scripts/`

- Command-line orchestrators:
  - Launch training
  - Preprocess data
  - Generate AI reports
- Uses: `configs/`, `models/`, `generative_ai/`, `mlruns/`
- Called by: `Makefile`

---

### `mlruns/`

- Local MLflow tracking store (artifacts, runs, experiments).
- Used by: `models/`, `generative_ai/`, `scripts/`

---

### `Makefile`

- Wraps all operations:
  - `make train-dev`
  - `make train-prod`
  - `make mlflow`
  - `make generate-report`
- Calls: `scripts/`, `models/`, `mlruns/`

---

## 🏗️ Optional Additions for Future

| Planned Module | Role                                                              |
| -------------- | ----------------------------------------------------------------- |
| `mlops/`       | Terraform/SageMaker/EMR deploy logic for cloud deployment         |
| `serving/`     | FastAPI model serving endpoints (in future)                       |
| `monitoring/`  | Logs/metrics drift monitoring module with alerting or LLM summary |

---

**Documentation updated: June 2025 (Dependencies Update)**