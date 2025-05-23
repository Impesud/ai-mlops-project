# scripts/pipeline.py
# Pipeline di orchestrazione end-to-end: verifica CSV raw, crea bucket, sync CSV raw, ingest, train
import os
import sys
import subprocess
import boto3
import yaml

# Carica la configurazione da project_root/data_ingestion/config.yaml
def load_config(path="data_ingestion/config.yaml"):
    scripts_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(scripts_dir)
    config_path = os.path.join(project_root, path)
    if not os.path.exists(config_path):
        print(f"ERRORE: config file non trovato in {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# Verifica o crea un bucket S3
def ensure_bucket(s3_client, bucket_name, region):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' già esistente")
    except s3_client.exceptions.NoSuchBucket:
        print(f"Creo bucket '{bucket_name}' in regione {region}")
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": region}
        )

if __name__ == "__main__":
    # Caricamento config
    cfg = load_config()
    region = cfg['aws']['region']
    raw_bucket = cfg['path'].replace('s3a://','').rstrip('/')
    proc_bucket = cfg['output_path'].replace('s3a://','').rstrip('/')

    # Verifica presenza file CSV raw di test
    sample_csv = os.path.join('data', 'raw-data', 'sample_raw_data.csv')
    if not os.path.exists(sample_csv):
        print("ERRORE: file 'sample_raw_data.csv' non presente in data/raw-data/")
        sys.exit(1)
    else:
        print(f"Trovato file di input: {sample_csv}")

    # Inizializza client S3
    session = boto3.Session(profile_name=os.environ.get('AWS_PROFILE'), region_name=region)
    s3 = session.client('s3')

    # Assicura bucket raw e processed
    ensure_bucket(s3, raw_bucket, region)
    ensure_bucket(s3, proc_bucket, region)

    # Sincronizza dati raw su S3
    print(f"Sincronizzo data/raw-data/ su s3://{raw_bucket}/")
    subprocess.run([
        "aws", "s3", "sync",
        os.path.join('data', 'raw-data') + os.sep,
        f"s3://{raw_bucket}/"
    ], check=True)

    # Esegui Spark ingestion via python (usa la SparkSession di PySpark)
    env = os.environ.copy()
    env['PYSPARK_PYTHON'] = sys.executable
    env['PYSPARK_DRIVER_PYTHON'] = sys.executable
    print("Avvio fase di ingestione Spark...")
    
    try:
        subprocess.run([
            sys.executable,
            "data_ingestion/ingest_spark.py"
        ], check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"ERRORE: ingest_spark.py terminato con codice {e.returncode}")
        sys.exit(e.returncode)

    # Esegui training
    print("Avvio fase di training MLflow...")
    try:
        completed = subprocess.run(
            [sys.executable, "models/train.py"],
            check=True,
            env=env,
            text=True
        )
        print(completed.stdout)
    except subprocess.CalledProcessError as e:
        print("===== TRAINING STDOUT =====")
        print(e.stdout)
        print("===== TRAINING STDERR =====")
        print(e.stderr)
        print(f"ERRORE: il training è terminato con codice {e.returncode}")
        sys.exit(e.returncode)

    print("Pipeline completata con successo")

