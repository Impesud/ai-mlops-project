# scripts/pipeline.py
import os
import sys
import subprocess
import boto3
import yaml

def load_config(path="data_ingestion/config.yaml"):
    scripts_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(scripts_dir)
    config_path = os.path.join(project_root, path)
    if not os.path.exists(config_path):
        print(f"ERRORE: config file non trovato in {config_path}")
        sys.exit(1)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

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
    cfg = load_config()
    mode = cfg.get("mode", "dev")
    region = cfg['aws']['region']
    
    # Percorso file dinamico (dev o prod)
    local_csv = cfg[cfg['mode']]['path']
    s3_path = cfg['s3_path']
    s3_output_path = cfg['s3_output_path'] 
    raw_bucket = s3_path.replace('s3a://', '').rstrip('/')
    proc_bucket = s3_output_path.replace('s3a://','').rstrip('/')

    if not os.path.exists(local_csv):
        print(f"ERRORE: file CSV non trovato in {local_csv}")
        sys.exit(1)
    else:
        print(f"Trovato file di input: {local_csv}")

    session = boto3.Session(profile_name=os.environ.get('AWS_PROFILE'), region_name=region)
    s3 = session.client('s3')

    ensure_bucket(s3, raw_bucket, region)
    ensure_bucket(s3, proc_bucket, region)

    print(f"Sincronizzo data/raw-data/ su s3://{raw_bucket}/")
    subprocess.run([
        "aws", "s3", "sync",
        os.path.join('data', 'raw-data') + os.sep,
        f"s3://{raw_bucket}/"
    ], check=True)

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

    print("✅ Pipeline completata con successo")

