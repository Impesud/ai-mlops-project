# models/train.py
# Script di training del modello con feature engineering temporale, SMOTE e class weights
import os
import sys
import subprocess
import yaml
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature

# Carica configurazione
def load_config(path="config.yaml"):
    base = os.path.dirname(__file__)
    config_path = os.path.join(base, path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file non trovato: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # 1) Caricamento config
    cfg = load_config()

    # 2) Inizializza SparkSession con supporto S3A
    spark = (
        SparkSession.builder
        .appName("ModelTraining")
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.1")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", os.environ['AWS_ACCESS_KEY_ID'])
        .config("spark.hadoop.fs.s3a.secret.key", os.environ['AWS_SECRET_ACCESS_KEY'])
        .config("spark.hadoop.fs.s3a.endpoint", f"s3.{os.environ['AWS_REGION']}.amazonaws.com")
        .getOrCreate()
    )

    # 3) Lettura dati preprocessati e rimozione null
    df = spark.read.parquet(cfg['input_path'])
    df = df.filter(col('event_time').isNotNull() & col('value').isNotNull())

    # 4) Conversione in Pandas
    pandas_df = df.toPandas()

    # 5) Feature engineering temporale
    pandas_df['timestamp'] = pandas_df['event_time'].astype('int64') // 10**9

    # 6) Definizione del target binario
    pandas_df['label'] = (pandas_df['action'] == 'purchase').astype(int)

    # 7) Drop colonne non numeriche
    pandas_df = pandas_df.drop(columns=['user_id', 'action', 'event_time'])

    # 8) Definisci X e y
    X = pandas_df.drop('label', axis=1)
    y = pandas_df['label']

    # 9) Split train/test stratificato
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg['test_size'],
        random_state=cfg['random_seed'],
        stratify=y
    )

    # 10) Resampling SMOTE e class weights
    try:
        from imblearn.over_sampling import SMOTE
        sm = SMOTE(random_state=cfg['random_seed'])
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print(f"Resampling SMOTE completato: da {len(y_train)} a {len(y_res)} esempi bilanciati")
    except ImportError:
        print("imbalanced-learn non installato: procedo con dati originali")
        X_res, y_res = X_train, y_train

    # 11) Inizializza MLflow Tracking locale
    # Forza tracking URI per GitHub Actions
    tracking_dir = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_dir)

    # Prepara cartella artifact
    os.makedirs("mlruns", exist_ok=True)
    mlflow.set_experiment("my-experiment")
    with mlflow.start_run():
        # Log parametri
        mlflow.log_params(cfg['model_params'])

        # Training
        model = RandomForestClassifier(
            **cfg['model_params'],
            class_weight="balanced"
        )
        model.fit(X_res, y_res)

        # Valutazione
        train_preds = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        mlflow.log_metric('train_accuracy', train_acc)

        test_preds = model.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        mlflow.log_metric('test_accuracy', test_acc)

        print(f"Training set accuracy: {train_acc:.4f}")
        print(f"Test set accuracy: {test_acc:.4f}")
        print("Classification report (test):")
        print(classification_report(y_test, test_preds))
        print("Confusion matrix (test):")
        print(confusion_matrix(y_test, test_preds))

        # Salva modello
        X_res_safe = X_res.copy()
        int_cols = X_res_safe.select_dtypes(include='int').columns
        X_res_safe[int_cols] = X_res_safe[int_cols].astype('float64')
        
        signature = infer_signature(X_res_safe, model.predict(X_res_safe))
        input_example = X_res[:5]

        mlflow.sklearn.log_model(
            model,
            artifact_path='model',
            input_example=input_example,
            signature=signature
        )

        # Genera report con AI
        prompt = cfg.get("generative_ai", {}).get("prompt", "Analisi dei dati")
        output_path = cfg.get("generative_ai", {}).get("output_path", "report.txt")

        print(f"🧠 Genero report generativo con prompt: \"{prompt}\"")
        try:
            subprocess.run(
                [sys.executable, "generative_ai/generate.py",
                "--prompt", prompt,
                "--output", output_path],
                check=True,
                text=True
            )
            mlflow.log_artifact(output_path)
            print(f"✅ Report loggato su MLflow: {output_path}")
        except Exception as e:
            print(f"⚠️ Errore durante generazione/logging report: {e}")

    # 13) Arresta Spark
    spark.stop()
    print("Run MLflow completato.")

    # 14) Avvio MLflow UI in background
    def avvia_mlflow_ui():
        print("Avvio MLflow UI in background…")
        subprocess.Popen([
            sys.executable, "-m", "mlflow", "ui",
            "--backend-store-uri", "./mlruns",
            "--port", "5000"
        ])
        if sys.stdin.isatty():
            input("Premi Invio per chiudere MLflow UI e uscire…")
        else:
            print("Esecuzione non interattiva: MLflow UI avviata in background senza attesa.")
            time.sleep(3)  # breve attesa opzionale per sicurezza

    avvia_mlflow_ui()
