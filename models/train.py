import argparse
import sys
from utils.io import load_env_config
from utils.logging_utils import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", help="Environment: dev or prod")
    parser.add_argument("--mlflow-ui", action="store_true", help="Start MLflow UI after training")
    args = parser.parse_args()

    logger = setup_logger("train", args.env)
    logger.info(f"Starting training in environment: {args.env}")

    try:
        cfg = load_env_config(args.env)
        data_cfg = cfg["data"]
        model_cfg = cfg["model"]
        logger.info(f"Loaded config for environment '{args.env}'.")

        if args.env == "dev":
            from models import train_sklearn
            logger.info("Dispatching to train_sklearn.main()")
            train_sklearn.main(args.env, data_cfg=data_cfg, model_cfg=model_cfg, start_ui=args.mlflow_ui)
        elif args.env == "prod":
            from models import train_spark
            logger.info("Dispatching to train_spark.main()")
            train_spark.main(args.env, data_cfg=data_cfg, model_cfg=model_cfg, start_ui=args.mlflow_ui)
        else:
            logger.error(f"Unknown environment: {args.env}")
            raise ValueError("Unknown environment")
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)
    


    
    