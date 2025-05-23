# Test for train.py
import mlflow
import os

def test_mlflow_last_run_has_model():
    runs = mlflow.search_runs(experiment_names=["my-experiment"], order_by=["start_time DESC"])
    assert not runs.empty, "❌ Nessun run trovato in MLflow"
    
    run_id = runs.iloc[0].run_id

    try:
        model_dir = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="model")
        assert os.path.exists(model_dir), "❌ Modello non trovato tra gli artifact MLflow"
        print(f"✅ Modello trovato: {model_dir}")
    except Exception as e:
        raise AssertionError(f"❌ Errore durante il download degli artifact: {e}")
