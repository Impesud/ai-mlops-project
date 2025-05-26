import mlflow
import os
import sys

mlflow.set_tracking_uri("file:./mlruns")

try:
    experiment = mlflow.get_experiment_by_name("my-experiment")
    if not experiment:
        raise Exception("Esperimento 'my-experiment' non trovato.")

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time desc"])
    if runs.empty:
        raise Exception("Nessun run trovato.")

    run_id = runs.iloc[0]["run_id"]
    model_path = os.path.join("mlruns", experiment.experiment_id, run_id, "artifacts", "model")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato in: {model_path}")

    print(model_path)

except Exception as e:
    print(f"Errore: {e}")
    sys.exit(1)
