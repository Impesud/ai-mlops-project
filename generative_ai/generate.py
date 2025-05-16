import argparse
import os
from openai import OpenAI, RateLimitError, APIError

def generate_text(prompt: str, output_file: str, dev_mode: bool = False):
    """
    Genera un report narrativo sui dati del progetto usando OpenAI LLM.
    - Carica i dati preprocessati (Parquet) in Pandas.
    - Calcola statistiche chiave.
    - Invoca l'API OpenAI per generare un'analisi testuale.
    - Salva l'output su file.
    """
    # 1) Carica dati preprocessati
    import pandas as pd
    data_path = os.getenv("PROCESSED_DATA_PATH", "data/processed")
    try:
        df = pd.read_parquet(data_path)
    except Exception:
        df = pd.read_csv(os.getenv("RAW_DATA_PATH", "data/raw-data/sample_raw_data.csv"))

    # 2) Calcola statistiche di base
    total = len(df)
    purchase_count = int((df['action']== 'purchase').sum())
    non_purchase = total - purchase_count
    value_stats = df['value'].describe().to_dict()

    # 3) Prepara messaggio contestuale
    summary = (
        f"Dataset summary:"
        f"- Total events: {total}"
        f"- Purchase: {purchase_count}"
        f"- Non-purchase: {non_purchase}"
        f"- Value stats: {value_stats}"
    )

    # 4) Costruisci prompt completo
    full_prompt = summary + "User prompt: " + prompt

    # 5) Recupera API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not dev_mode:
        raise ValueError("Missing OPENAI_API_KEY.")

    # 6) Invoca LLM oppure fallback dev
    if dev_mode:
        text = "[DEV MODE] Analysis simulated. " + summary
    else:
        client = OpenAI(api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "system", "content": "You are a data analyst."},
                          {"role": "user", "content": full_prompt}],
                max_tokens=500
            )
            text = resp.choices[0].message.content
        except RateLimitError:
            raise RuntimeError("Rate limit exceeded. Riprova più tardi.")
        except APIError as e:
            raise RuntimeError(f"OpenAI API error: {e}")

    # 7) Salva su file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Output saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--output', type=str, default='result.txt')
    parser.add_argument('--dev-mode', action='store_true', help='Usa risposta simulata per test')
    args = parser.parse_args()
    generate_text(args.prompt, args.output, args.dev_mode)