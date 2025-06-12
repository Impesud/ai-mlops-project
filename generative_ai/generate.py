# generate.py

import argparse
import sys 
import os
from datetime import datetime
import glob
import pandas as pd
from openai import OpenAI, RateLimitError, APIError

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.io import load_env_config
from utils.logging_utils import setup_logger

def generate_text(prompt: str, output_file: str, dev_mode: bool = False, mode: str = "dev"):
    logger = setup_logger("generate_ai", mode)
    cfg = load_env_config(mode)
    data_cfg = cfg["data"]
    gen_cfg = cfg.get("model", {}).get("generative_ai", {})

    base_parquet_path = data_cfg.get("local_processed_path", f"data/processed/{mode}/")

    logger.info(f"Active mode: {mode}")
    logger.info(f"Parquet data path: {base_parquet_path}")

    # Determine fallback CSV path dynamically
    local_input_path = data_cfg.get("local_input_path", f"data/raw-data/{mode}/")
    if os.path.isdir(local_input_path):
        csv_files = sorted(glob.glob(os.path.join(local_input_path, "*.csv")))
        fallback_csv = csv_files[0] if csv_files else None
    else:
        fallback_csv = local_input_path if local_input_path.endswith(".csv") else None

    if not fallback_csv or not os.path.exists(fallback_csv):
        logger.error(f"❌ No fallback CSV file found in {local_input_path}")
        raise FileNotFoundError(f"No CSV file found in {local_input_path}")

    try:
        df = pd.read_parquet(base_parquet_path)
        logger.info(f"✅ Data loaded from Parquet: {base_parquet_path}")
    except Exception as e:
        logger.warning(f"⚠️ Error reading Parquet: {e}")
        logger.info(f"🔁 Falling back to CSV: {fallback_csv}")
        df = pd.read_csv(fallback_csv)

    # Check required columns
    for col in ["action", "value"]:
        if col not in df.columns:
            logger.error(f"❌ Column '{col}' not found in data.")
            raise ValueError(f"Column '{col}' not found in data.")

    total = len(df)
    purchase_count = int((df['action'] == 'purchase').sum())
    non_purchase = total - purchase_count
    value_stats = df['value'].describe().to_dict()

    summary = (
        f"# 📈 Dataset Summary\n"
        f"- Total events: {total}\n"
        f"- Purchases: {purchase_count}\n"
        f"- Non-purchases: {non_purchase}\n"
        f"- Value statistics: {value_stats}\n\n"
    )

    configured_prompt = gen_cfg.get("prompt", prompt)
    full_prompt = summary + f"# 🧠 User Prompt\n{prompt or configured_prompt}\n"

    if os.getenv("GITHUB_ACTIONS", "false").lower() == "true":
        dev_mode = True

    if dev_mode:
        text = "[DEV MODE] Simulated report.\n\n" + full_prompt
        logger.info("🛠️ Dev mode enabled: skipping OpenAI API call.")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("❌ OPENAI_API_KEY environment variable not found.")
            raise ValueError("OPENAI_API_KEY environment variable not found.")
        client = OpenAI(api_key=api_key)
        try:
            logger.info("🤖 Generating AI report via OpenAI API...")
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a skilled data analyst."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=500
            )
            text = resp.choices[0].message.content
            logger.info("✅ AI report generated successfully.")
        except RateLimitError:
            logger.error("🚫 OpenAI API rate limit exceeded.")
            raise RuntimeError("OpenAI API rate limit exceeded.")
        except APIError as e:
            logger.error(f"🚫 OpenAI API error: {e}")
            raise RuntimeError(f"OpenAI API error: {e}")

    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"docs/reports/report_{mode}_pipeline_{timestamp}.txt"

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# 📝 Generative Report\n\n{full_prompt}\n## 📋 AI Output\n{text}\n")
        logger.info(f"📁 Report saved to: {output_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save report to {output_file}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--dev-mode', action='store_true', help='Use simulated mode (no API calls)')
    parser.add_argument('--env', type=str, default='dev', help='Environment: dev or prod')
    args = parser.parse_args()
    generate_text(args.prompt, args.output, args.dev_mode, args.env)
