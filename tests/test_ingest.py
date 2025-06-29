from pathlib import Path
from unittest.mock import patch

import pytest

from utils.io import get_intermediate_output_path


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("TEST_ENV", "dev")
    yield
    monkeypatch.delenv("TEST_ENV")


@patch("utils.io.load_env_config")
def test_get_intermediate_output_path_returns_path(mock_config):
    mock_config.return_value = {"data": {"local_intermediate_path": "/tmp/data/intermediate"}}
    result = get_intermediate_output_path("dev")
    assert isinstance(result, Path)
    assert str(result) == "/tmp/data/intermediate"
    mock_config.assert_called_once_with("dev")


@patch("pathlib.Path.glob")
@patch("utils.io.load_env_config")
def test_check_parquet_files_exist(mock_config, mock_glob, mock_env):
    mock_config.return_value = {"data": {"local_intermediate_path": "/tmp/data/intermediate"}}
    mock_glob.return_value = [Path("file1.parquet"), Path("file2.parquet")]

    result_path = get_intermediate_output_path("dev")
    files = list(result_path.glob("*.parquet"))
    assert len(files) > 0


@patch("pathlib.Path.stat")
@patch("pathlib.Path.glob")
@patch("utils.io.load_env_config")
def test_parquet_files_not_empty(mock_config, mock_glob, mock_stat):
    mock_config.return_value = {"data": {"local_intermediate_path": "/tmp/data/intermediate"}}
    dummy_file = Path("file1.parquet")
    mock_glob.return_value = [dummy_file]
    mock_stat.return_value.st_size = 10

    result_path = get_intermediate_output_path("dev")
    files = list(result_path.glob("*.parquet"))
    assert all(f.stat().st_size > 0 for f in files)
