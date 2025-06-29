import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from cerberus import Validator

from utils.config_schema import (
    aws_schema,
    cloud_schema,
    data_schema,
    generative_ai_schema,
    model_schema_sklearn,
    model_schema_spark,
)
from utils.logging_utils import setup_logger

logger = setup_logger("utils_io")


def load_yaml(path: str, interpolate_env: bool = False) -> Dict[str, Any]:
    """
    Load a single YAML file and return its content as a dictionary.
    If interpolate_env is True, replaces ${VAR} with os.environ["VAR"] (or leaves as-is if not found).
    Returns {} if the file is empty or not a dict.
    """
    try:
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
        logger.info(f"Loaded YAML file: {path}")
    except Exception as e:
        logger.error(f"Failed to read YAML file {path}: {e}")
        raise

    if interpolate_env:
        pattern = re.compile(r"\$\{(\w+)\}")

        def repl(m: re.Match[str]) -> str:
            var = m.group(1)
            val = os.getenv(var)
            if val is None:
                logger.warning(f"Environment variable '{var}' not set for interpolation in {path}")
                return f"${{{var}}}"
            return val

        content = pattern.sub(repl, content)

    try:
        raw_data = yaml.safe_load(content)
        if raw_data is None:
            return {}
        if not isinstance(raw_data, dict):
            raise TypeError(f"YAML content is not a dict: {type(raw_data)}")
        return raw_data
    except Exception as e:
        logger.error(f"Failed to parse YAML file {path}: {e}")
        raise


def validate_config_sections(config: dict, env: str):
    """
    Validate each config section using its corresponding Cerberus schema.
    Raises ValueError if any section is invalid or missing.
    """
    schema_map = {
        "aws": aws_schema,
        "cloud": cloud_schema,
        "data": data_schema,
        "generative_ai": generative_ai_schema,
    }

    if env == "dev":
        schema_map["model"] = model_schema_sklearn
    elif env == "prod":
        schema_map["model"] = model_schema_spark
    else:
        logger.error(f"Unknown environment '{env}' provided for config validation.")
        raise ValueError(f"Unknown environment '{env}'")

    for section, schema in schema_map.items():
        if section in config:
            v = Validator(schema)
            if not v.validate(config[section]):
                logger.error(f"Config '{section}' is invalid: {v.errors}")
                raise ValueError(f"Config '{section}' is invalid: {v.errors}")
            else:
                logger.info(f"✅ Config section '{section}' validated successfully.")
        else:
            logger.error(f"Config section '{section}' is missing!")
            raise KeyError(f"Config section '{section}' is missing!")


def load_env_config(env: str) -> dict:
    """
    Load and merge all YAML configuration files from configs/<env>/.
    Each file is nested under its base name (without extension) as a top-level key.
    Only 'aws.yaml' will interpolate environment variables.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(base_dir, "configs", env)

    if not os.path.isdir(config_dir):
        logger.error(f"Config directory not found: {config_dir}")
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    config = {}
    for filename in sorted(os.listdir(config_dir)):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            key = os.path.splitext(filename)[0]
            filepath = os.path.join(config_dir, filename)
            interpolate = key == "aws"
            if key in config:
                logger.warning(f"Duplicate config key '{key}' in {config_dir}. Overwriting previous value.")
            config[key] = load_yaml(filepath, interpolate_env=interpolate)

    validate_config_sections(config, env)
    logger.info(f"Configuration for environment '{env}' loaded and validated.")
    return config


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


def get_processed_output_path(env="dev"):
    cfg = load_env_config(env)
    return Path(cfg["data"]["local_processed_path"])
