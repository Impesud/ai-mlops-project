import argparse
import sys
from utils.logging_utils import setup_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="dev", help="Environment: dev or prod")
    args = parser.parse_args()

    logger = setup_logger("train", args.env)
    logger.info(f"🚀 Starting training in environment: {args.env}")

    try:
        if args.env == "dev":
            from models import train_sklearn
            logger.info("⚡ Dispatching to train_sklearn (scikit-learn)...")
            train_sklearn.main(env=args.env)
            
        elif args.env == "prod":
            from models import train_spark
            logger.info("⚡ Dispatching to train_spark (Spark)...")
            train_spark.main(env=args.env)
            
        else:
            logger.error(f"❌ Unknown environment: {args.env}")
            raise ValueError("Unknown environment")
            
    except Exception as e:
        logger.exception(f"❌ Training failed: {e}")
        sys.exit(1)

    


    
    