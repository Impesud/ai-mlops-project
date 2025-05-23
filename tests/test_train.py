# Test for train.py
import mlflow
import os

def test_mlflow_last_run_has_model():
    runs = mlflow.search_runs(experiment_names=["my-experiment"], order_by=["start_time DESC"])
    assert not runs.empty, "❌ Nessun run trovato in MLflow"
    last_run = runs.iloc[0]
    model_path = os.path.join("mlruns", str(last_run.experiment_id), last_run.run_id, "artifacts", "model")
    assert os.path.exists(model_path), "❌ Modello non salvato come artifact in MLflow"
