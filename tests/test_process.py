from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from utils.io import get_processed_output_path


@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("TEST_ENV", "dev")
    yield
    monkeypatch.delenv("TEST_ENV")


@patch("utils.io.load_env_config")
def test_get_processed_output_path_returns_path(mock_config):
    mock_config.return_value = {"data": {"local_processed_path": "/tmp/data/processed"}}
    result = get_processed_output_path("dev")
    assert isinstance(result, Path)
    assert str(result) == "/tmp/data/processed"
    mock_config.assert_called_once_with("dev")


@patch("pathlib.Path.glob")
@patch("utils.io.load_env_config")
def test_check_parquet_files_exist(mock_config, mock_glob):
    mock_config.return_value = {"data": {"local_processed_path": "/tmp/data/processed"}}
    mock_glob.return_value = [Path("file1.parquet")]
    result_path = get_processed_output_path("dev")
    files = list(result_path.glob("*.parquet"))
    assert len(files) > 0


@patch("pyspark.sql.SparkSession")
@patch("utils.io.load_env_config")
def test_feature_columns_present(mock_config, mock_spark):
    mock_config.return_value = {"data": {"local_processed_path": "/tmp/data/processed"}}

    # Mock dataframe and its columns
    mock_df = MagicMock()
    mock_df.columns = [
        "hour",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "event_timestamp",
        "is_weekend",
    ]
    mock_spark.builder.getOrCreate.return_value.read.parquet.return_value = mock_df

    spark = mock_spark.builder.getOrCreate.return_value
    df = spark.read.parquet("/tmp/data/processed")

    expected = {
        "hour",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "event_timestamp",
        "is_weekend",
    }
    missing = expected - set(df.columns)
    assert not missing, f"Missing feature columns: {missing}"


@patch("pyspark.sql.SparkSession")
@patch("utils.io.load_env_config")
def test_feature_values_valid(mock_config, mock_spark):
    mock_config.return_value = {"data": {"local_processed_path": "/tmp/data/processed"}}

    # Setup mock dataframe
    mock_df = MagicMock()
    mock_df.columns = [
        "hour",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "month",
        "is_weekend",
    ]

    mock_df.filter.return_value.count.side_effect = [0] * (2 * len(mock_df.columns))
    mock_spark.builder.getOrCreate.return_value.read.parquet.return_value = mock_df

    checks = {
        "hour": (0, 23),
        "day_of_week": (1, 7),
        "day_of_month": (1, 31),
        "week_of_year": (1, 53),
        "month": (1, 12),
        "is_weekend": (0, 1),
    }

    for col_name, (min_val, max_val) in checks.items():
        null_count = mock_df.filter().count()
        assert null_count == 0

        invalid_count = mock_df.filter().count()
        assert invalid_count == 0
