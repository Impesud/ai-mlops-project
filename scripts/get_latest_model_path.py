import os
import glob

def find_latest_model_artifact():
    # Trova tutte le directory "model" dentro mlruns
    matches = glob.glob("mlruns/*/*/artifacts/model", recursive=True)

    if not matches:
        raise FileNotFoundError("Nessun modello MLflow trovato in mlruns/*/*/artifacts/model")

    # Ordina per ultima modifica
    latest_model_dir = max(matches, key=os.path.getmtime)
    abs_path = os.path.abspath(latest_model_dir)

    print(f"Ultima directory modello: {abs_path}")

    # Controlla se esiste il file MLmodel
    mlmodel_file = os.path.join(abs_path, "MLmodel")
    if os.path.isfile(mlmodel_file):
        print(f"File MLmodel: {mlmodel_file}")
    else:
        print("Nessun file MLmodel trovato.")

    # Cerca eventuali file di modello salvato (es. .pkl, .onnx, ecc.)
    model_files = glob.glob(os.path.join(abs_path, "*.*"))
    for f in model_files:
        if f.endswith((".pkl", ".bin", ".onnx", ".joblib")):
            print(f"🔍 File modello trovato: {f}")

if __name__ == "__main__":
    find_latest_model_artifact()


