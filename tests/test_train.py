import mlflow
import os
import pytest

def test_mlflow_last_run_has_model():
    """
    Verifies that a model was successfully logged in the last MLflow run
    for the correct experiment (from config) and that the artifact exists.
    """
    test_env = os.getenv("TEST_ENV", "dev")
    # Use the experiment name from your pipeline/config
    experiment_name = "spark-experiment" if test_env == "prod" else "sklearn-experiment"

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

def test_mlflow_last_run_has_llm_report():
    """
    Verifies that the LLM-generated report was logged as an artifact in the last MLflow run.
    """
    test_env = os.getenv("TEST_ENV", "dev")
    experiment_name = "spark-experiment" if test_env == "prod" else "sklearn-experiment"

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["start_time DESC"]
    )

    assert not runs.empty, f"❌ No MLflow runs found for experiment '{experiment_name}' in env '{test_env}'"

    run_id = runs.iloc[0].run_id

    try:
        report_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="report.txt"
        )
        assert os.path.exists(report_path), "❌ LLM report artifact not found"
        print(f"✅ LLM report found: {report_path}")
    except Exception as e:
        raise AssertionError(f"❌ Error while downloading LLM report artifact: {e}")

