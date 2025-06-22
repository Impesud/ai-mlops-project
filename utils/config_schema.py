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

model_schema = {
    "input_path": {"type": "string", "required": True, "empty": False},
    "test_size": {"type": "float", "required": True, "min": 0, "max": 1},
    "random_seed": {"type": "integer", "required": True},
    "features": {"type": "list", "required": False, "schema": {"type": "string"}},

    # Default model parameters for RandomForestClassifier
    "model_params": {
        "type": "dict",
        "required": True,
        "schema": {
            "n_estimators": {"type": "integer", "required": True, "min": 1},
            "max_depth": {"type": "integer", "required": True, "min": 1},
            "min_samples_leaf": {"type": "integer", "required": False, "min": 1},
            "max_features": {"type": ["string", "float"], "required": False},
            "class_weight": {"type": "string", "required": False, "allowed": ["balanced", "balanced_subsample"]}
        }
    },
    # Flag to enable hyperparameter search
    "do_hyper_search": {"type": "boolean", "required": False},
    # Number of folds for cross-validation
    "cv_folds": {"type": "integer", "required": False, "min": 2},
    # Grid for hyperparameter search
    "param_grid": {
        "type": "dict",
        "required": False,
        "keysrules": {"type": "string"},
        "valuesrules": {
            "type": "list",
            "schema": {"type": ["integer", "float", "string"]}
        }
    },
    # Parallelism for grid search
    "parallelism": {"type": "integer", "required": False, "min": 1}
}
