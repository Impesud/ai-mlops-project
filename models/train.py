import argparse
import sys
from utils.logging_utils import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", help="Environment: dev or prod")
    parser.add_argument("--mlflow-ui", action="store_true", help="Start MLflow UI after training")
    args = parser.parse_args()

    logger = setup_logger("train", args.env)
    logger.info(f"🚀 Starting training in environment: {args.env}")

    try:
        if args.env == "dev":
            from models import train_sklearn
            logger.info("Dispatching to train_sklearn...")
            train_sklearn.main(env=args.env)
            
        elif args.env == "prod":
            from models import train_spark
            logger.info("Dispatching to train_spark...")
            train_spark.main(env=args.env, start_ui=args.mlflow_ui)
            
        else:
            logger.error(f"❌ Unknown environment: {args.env}")
            raise ValueError("Unknown environment")
            
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        sys.exit(1)

    


    
    