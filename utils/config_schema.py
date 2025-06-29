# schemas.py

aws_schema = {
    "access_key_id": {"type": "string", "required": True, "empty": False},
    "secret_access_key": {"type": "string", "required": True, "empty": False},
    "region": {"type": "string", "required": True, "empty": False},
}

cloud_schema = {
    "s3_bucket_input": {"type": "string", "required": True, "empty": False},
    "s3_bucket_intermediate": {"type": "string", "required": True, "empty": False},
    "s3_bucket_output": {"type": "string", "required": True, "empty": False},
}

data_schema = {
    "local_input_path": {"type": "string", "required": True, "empty": False},
    "local_intermediate_path": {"type": "string", "required": True, "empty": False},
    "local_processed_path": {"type": "string", "required": True, "empty": False},
    "format": {"type": "string", "required": True, "allowed": ["csv", "parquet"]},
}

generative_ai_schema = {
    "enabled": {"type": "boolean", "required": True},
    "prompt": {"type": "string", "required": True, "empty": False},
    "output_path": {"type": "string", "required": True, "empty": False},
}

# ----------------- MODEL SCHEMA SKLEARN --------------------
model_schema_sklearn = {
    "input_path": {"type": "string", "required": True, "empty": False},
    "test_size": {"type": "float", "required": True, "min": 0, "max": 1},
    "random_seed": {"type": "integer", "required": True},
    "features": {"type": "list", "required": True, "schema": {"type": "string"}},
    "model_params": {
        "type": "dict",
        "required": True,
        "schema": {
            "n_estimators": {"type": "integer", "min": 1, "required": False},
            "max_depth": {"type": "integer", "min": 1, "required": False},
            "min_samples_leaf": {"type": "integer", "min": 1, "required": False},
            "max_features": {"type": ["string", "float"], "required": False},
            "class_weight": {
                "type": "string",
                "allowed": ["balanced", "balanced_subsample"],
                "required": False,
            },
        },
    },
    "param_grid": {
        "type": "dict",
        "required": False,
        "keysrules": {"type": "string"},
        "valuesrules": {
            "type": "list",
            "schema": {"type": ["integer", "float", "string"]},
        },
    },
    "do_hyper_search": {"type": "boolean", "required": False},
    "cv_folds": {"type": "integer", "required": False, "min": 2},
}

# ----------------- MODEL SCHEMA SPARK --------------------
model_schema_spark = {
    "input_path": {"type": "string", "required": True, "empty": False},
    "test_size": {"type": "float", "required": True, "min": 0, "max": 1},
    "random_seed": {"type": "integer", "required": True},
    "features": {"type": "list", "required": True, "schema": {"type": "string"}},
    "model_params": {
        "type": "dict",
        "required": True,
        "schema": {
            "maxIter": {"type": "integer", "min": 1, "required": True},
            "maxDepth": {"type": "integer", "min": 1, "required": True},
            "stepSize": {"type": "float", "min": 0, "required": True},
            "subsamplingRate": {"type": "float", "min": 0, "max": 1, "required": True},
            "minInstancesPerNode": {"type": "integer", "min": 1, "required": True},
            "minInfoGain": {"type": "float", "min": 0, "required": True},
        },
    },
    "grid": {
        "type": "dict",
        "required": False,
        "keysrules": {"type": "string"},
        "valuesrules": {
            "type": "list",
            "schema": {"type": ["integer", "float", "string"]},
        },
    },
    "numFolds": {"type": "integer", "required": False, "min": 2},
    "parallelism": {"type": "integer", "required": False, "min": 1},
}
