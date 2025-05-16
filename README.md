# AI MLOps Project

Template di progetto per integrazione di Big Data, Data Cloud, MLOps e IA Generativa.

## Prerequisiti

* Python 3.8+ (consigliato 3.10 o 3.11)
* Java 8+ (per Spark)
* Docker
* MLflow
* AWS CLI / GCP SDK / Azure CLI

## Setup su Windows

1. **Clona il repository**

   ```bat
   git clone https://github.com/tuo-utente/ai-mlops-project.git
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

   * Apri o crea il file `C:\Users\<TUO_UTENTE>\.aws\config`
   * Assicurati di avere una sezione senza prefisso `profile `:

     ```ini
     [impesud]
     region = eu-central-1
     ```
   * Imposta la variabile d'ambiente nel cmd:

     ```bat
     set AWS_PROFILE=impesud
     ```
4. **Imposta credenziali nel config**

   * Modifica `data_ingestion/config.yaml` con il tuo bucket S3 e le variabili d'ambiente:

     ```yaml
     format: csv
     path: s3a://my-mlops-raw-data/
     output_path: s3a://my-mlops-processed-data/
     aws:
       region: eu-central-1
     ```

## Componenti e script Windows

* **data_ingestion**: lettura CSV da S3/locale, pulizia e scrittura Parquet.
* **models**: training e valutazione con Spark→Pandas→sklearn, SMOTE e class_weight.
* **scripts**: orchestrazione end-to-end.

### Esempio di utilizzo

```bat
REM 1) Attiva virtualenv
call venv\Scripts\activate

REM 2) Esegui pipeline completa (ingest + train)
python scripts\pipeline.py

REM 3) In alternativa, step separati:
call scripts\run_ingest.bat  :: esegue data_ingestion\ingest_spark.py
call scripts\run_train.bat   :: esegue models\train.py

REM 4) Generative AI
python generative_ai/generate.py --prompt "Dammi un'analisi dei dati" --output report.txt
```

## MLflow UI

Dopo il training, la UI di MLflow sarà disponibile su [http://localhost:5000](http://localhost:5000).
I run e le metriche (train_accuracy, test_accuracy) sono salvati in `mlruns/`.

## Punti mancanti e prossimi passi

1. **Feature engineering** avanzato: estrazione di ora/giorno, aggregazioni per user_id.
2. **Hyperparameter tuning** con CV e ottimizzazione su F1/recall per la classe purchase.
3. **Deployment** del modello MLflow (serve script di deploy o Docker image)
4. **Generative AI**: integrazione con `generative_ai/generate.py` e parametrizzazione nel pipeline.
5. **CI/CD**: completare il workflow in `.github/workflows/ci-cd.yml` per automatizzare ingest→train→deploy.

---

Mantieni aggiornati `requirements.txt` e `data_ingestion/config.yaml` per riflettere nuove dipendenze o bucket.



