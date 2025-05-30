import argparse
import os
from datetime import datetime
import pandas as pd
import yaml
from openai import OpenAI, RateLimitError, APIError


def load_config(path="models/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_text(prompt: str, output_file: str, dev_mode: bool = False):
    """
    Genera un report narrativo sui dati del progetto usando OpenAI LLM.
    Se dev_mode è True (o se viene eseguito su GitHub Actions), non usa le API OpenAI.
    """
    config = load_config()
    mode = config.get("mode", "dev")
    base_parquet_path = config[mode].get("output_path", f"data/processed/{mode}/")

    # Scegli il file CSV fallback in base alla modalità
    if mode == "dev":
        raw_data_fallback = "data/raw-data/sample_1k.csv"
    else:
        raw_data_fallback = "data/raw-data/sample_100k.csv"

    print(f"📊 Modalità attiva: {mode}")
    print(f"📂 Percorso dati Parquet: {base_parquet_path}")

    try:
        df = pd.read_parquet(base_parquet_path)
        print(f"✅ Dati caricati da {base_parquet_path}")
    except Exception as e:
        print(f"⚠️ Errore nella lettura Parquet: {e}")
        print(f"↩️ Fallback su CSV: {raw_data_fallback}")
        df = pd.read_csv(raw_data_fallback)

    # Statistiche di base
    total = len(df)
    purchase_count = int((df['action'] == 'purchase').sum())
    non_purchase = total - purchase_count
    value_stats = df['value'].describe().to_dict()

    # Costruzione del contesto
    summary = (
        f"# 📈 Dataset Summary\n"
        f"- Totale eventi: {total}\n"
        f"- Acquisti: {purchase_count}\n"
        f"- Non-acquisti: {non_purchase}\n"
        f"- Statistiche valore: {value_stats}\n\n"
    )

    # Prompt da config (se non sovrascritto via CLI)
    configured_prompt = config[mode]['generative_ai'].get("prompt", prompt)
    full_prompt = summary + f"# 🧠 Prompt Utente\n{prompt or configured_prompt}\n"

    # Ambiente GitHub Actions
    if os.getenv("GITHUB_ACTIONS", "false").lower() == "true":
        dev_mode = True

    if dev_mode:
        text = "[DEV MODE] Report simulato.\n\n" + full_prompt
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("❌ Variabile OPENAI_API_KEY non trovata.")
        client = OpenAI(api_key=api_key)
        try:
            print("🤖 Generazione AI in corso...")
            resp = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "Sei un analista dati esperto."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=500
            )
            text = resp.choices[0].message.content
        except RateLimitError:
            raise RuntimeError("⚠️ Rate limit superato.")
        except APIError as e:
            raise RuntimeError(f"❌ Errore API OpenAI: {e}")

    # Output dinamico
    if not output_file:
        today = datetime.today().strftime("%Y-%m-%d")
        output_file = f"docs/reports/report-{mode}-{today}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# 📝 Report Generativo\n\n{full_prompt}\n## 📋 Output AI\n{text}\n")

    print(f"✅ Report salvato in: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--dev-mode', action='store_true', help='Usa modalità simulata (senza API)')
    args = parser.parse_args()
    generate_text(args.prompt, args.output, args.dev_mode)

