import os
import glob

# Cerca tutti i modelli MLflow salvati
matches = glob.glob("mlruns/*/*/artifacts/model/MLmodel", recursive=True)

if not matches:
    raise FileNotFoundError("❌ Nessun modello MLflow trovato in mlruns/*/*/artifacts/model/MLmodel")

# Ordina per modifica più recente
latest_model = max(matches, key=os.path.getmtime)

# Stampa solo il percorso della cartella "model"
model_dir = os.path.dirname(latest_model)
print(model_dir)

