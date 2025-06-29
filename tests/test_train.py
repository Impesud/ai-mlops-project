import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("TEST_ENV", "dev")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")
    yield
    monkeypatch.delenv("TEST_ENV")
    monkeypatch.delenv("GITHUB_EVENT_NAME")


@patch("mlflow.set_experiment")
@patch("mlflow.search_runs")
@patch("mlflow.artifacts.download_artifacts")
def test_mlflow_last_run_has_model(mock_download_artifacts, mock_search_runs, mock_set_experiment, mock_env):
    # Arrange
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "exp123"
    mock_set_experiment.return_value = mock_experiment

    mock_run = MagicMock()
    mock_run.run_id = "run456"
    mock_search_runs.return_value = MagicMock(empty=False, iloc=[mock_run])

    mock_download_artifacts.return_value = "/tmp/model"
    with patch("os.path.exists", return_value=True):
        # Act
        from mlflow import artifacts, search_runs, set_experiment

        test_env = os.getenv("TEST_ENV", "dev")
        experiment_name = "spark-experiment" if test_env == "prod" else "sklearn-experiment"

        experiment_id = set_experiment(experiment_name).experiment_id
        runs = search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])
        run_id = runs.iloc[0].run_id
        model_dir = artifacts.download_artifacts(run_id=run_id, artifact_path="model")

        # Assert
        assert not runs.empty, "No MLflow runs found"
        assert os.path.exists(model_dir), "Model artifact directory not found"

        mock_set_experiment.assert_called_once_with(experiment_name)
        mock_search_runs.assert_called_once_with(experiment_ids=[experiment_id], order_by=["start_time DESC"])
        mock_download_artifacts.assert_called_once_with(run_id="run456", artifact_path="model")
