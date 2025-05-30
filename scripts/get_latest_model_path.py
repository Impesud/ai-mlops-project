import os
import glob

def find_latest_model_artifact():
    # Trova tutte le directory "model" dentro mlruns
    matches = glob.glob("mlruns/*/*/artifacts/model", recursive=True)

    if not matches:
        raise FileNotFoundError("No MLflow model directory found in mlruns/*/*/artifacts/model")

    # Ordina per ultima modifica
    latest_model_dir = max(matches, key=os.path.getmtime)

    # Stampa solo il path assoluto, senza testo extra
    print(os.path.abspath(latest_model_dir))

if __name__ == "__main__":
    find_latest_model_artifact()



