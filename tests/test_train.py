# tests/test_train.py
import mlflow
import os
import pytest


def test_mlflow_last_run_has_model():
    """
    This test verifies that a model was successfully logged in the last MLflow run
    for the experiment "my-experiment".
    """

    test_env = os.getenv("TEST_ENV", "dev")
    experiment_name = f"my-experiment"

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"]
    )

    assert not runs.empty, f"❌ No MLflow runs found for experiment '{experiment_name}' in env '{test_env}'"

    run_id = runs.iloc[0].run_id

    try:
        model_dir = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="model"
        )
        assert os.path.exists(model_dir), "❌ Model artifact directory not found"
        print(f"✅ Model found: {model_dir}")
    except Exception as e:
        raise AssertionError(f"❌ Error while downloading model artifacts: {e}")

