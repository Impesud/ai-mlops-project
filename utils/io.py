import yaml
import os
import re
import logging
from cerberus import Validator
from utils.config_schema import (
    aws_schema,
    cloud_schema,
    data_schema,
    generative_ai_schema,
    model_schema,
)
from utils.logging_utils import setup_logger

logger = setup_logger("utils_io")

def load_yaml(path: str, interpolate_env: bool = False) -> dict:
    """
    Load a single YAML file and return its content as a dictionary.
    If interpolate_env is True, replaces ${VAR} with os.environ["VAR"] (or leaves as-is if not found).
    Returns {} if the file is empty.
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
        def repl(m):
            var = m.group(1)
            val = os.getenv(var)
            if val is None:
                logger.warning(f"Environment variable '{var}' not set for interpolation in {path}")
                return f"${{{var}}}"
            return val
        content = pattern.sub(repl, content)

    try:
        data = yaml.safe_load(content)
        if data is None:
            data = {}
        return data
    except Exception as e:
        logger.error(f"Failed to parse YAML file {path}: {e}")
        raise

def validate_config_sections(config: dict):
    """
    Validate each config section using its corresponding Cerberus schema.
    Raises ValueError if any section is invalid or missing.
    """
    schema_map = {
        "aws": aws_schema,
        "cloud": cloud_schema,
        "data": data_schema,
        "generative_ai": generative_ai_schema,
        "model": model_schema,
    }
    for section, schema in schema_map.items():
        if section in config:
            v = Validator(schema)
            if not v.validate(config[section]):
                logger.error(f"Config '{section}' is invalid: {v.errors}")
                raise ValueError(f"Config '{section}' is invalid: {v.errors}")
            else:
                logger.info(f"Config section '{section}' validated successfully.")
        else:
            logger.error(f"Config section '{section}' is missing!")
            raise KeyError(f"Config section '{section}' is missing!")

def load_env_config(env: str) -> dict:
    """
    Load and merge all YAML configuration files from configs/<env>/.
    Each file is nested under its base name (without extension) as a top-level key.
    Only 'aws.yaml' will interpolate environment variables.
    If a file is empty, its value will be an empty dict.
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))  # root of the project
    config_dir = os.path.join(base_dir, "configs", env)

    if not os.path.isdir(config_dir):
        logger.error(f"Config directory not found: {config_dir}")
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    config = {}
    # Sort files for deterministic loading
    for filename in sorted(os.listdir(config_dir)):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            key = os.path.splitext(filename)[0]
            filepath = os.path.join(config_dir, filename)
            interpolate = key == "aws"
            if key in config:
                logger.warning(f"Duplicate config key '{key}' in {config_dir}. Overwriting previous value.")
            config[key] = load_yaml(filepath, interpolate_env=interpolate)

    validate_config_sections(config)
    logger.info(f"Configuration for environment '{env}' loaded and validated.")
    return config
