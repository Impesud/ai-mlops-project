import os
import logging
from pathlib import Path
from utils.io import load_env_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_intermediate_output_path(env="dev"):
    """
    Retrieve the intermediate output path from the environment configuration.

    Args:
        env (str): Environment name ('dev', 'prod', ...)

    Returns:
        Path: Path to the intermediate .parquet folder
    """
    cfg = load_env_config(env)
    return Path(cfg["data"]["local_intermediate_path"])

def test_parquet_written_to_intermediate():
    """
    Test that at least one .parquet file was created in the intermediate output folder.
    """
    env = os.getenv("TEST_ENV", "dev")
    intermediate_path = get_intermediate_output_path(env)

    logger.info(f"📁 Checking intermediate output path: {intermediate_path}")
    assert intermediate_path.exists(), f"❌ Folder '{intermediate_path}' does not exist."

    parquet_files = list(intermediate_path.glob("*.parquet"))
    logger.info(f"✅ Found {len(parquet_files)} .parquet file(s) in intermediate path.")
    assert len(parquet_files) > 0, f"❌ No .parquet files found in '{intermediate_path}'."


    
