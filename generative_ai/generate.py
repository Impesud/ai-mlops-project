import argparse
import os
import pandas as pd
from openai import OpenAI, RateLimitError, APIError


def generate_text(prompt: str, output_file: str, dev_mode: bool = False):
    """
    Genera un report narrativo sui dati del progetto usando OpenAI LLM.
    - Carica i dati preprocessati (Parquet) in Pandas.
    - Calcola statistiche chiave.
    - Invoca l'API OpenAI per generare un'analisi testuale.
    - Salva l'output su file.
    """
    print("📊 Caricamento dati...")
    data_path = os.getenv("PROCESSED_DATA_PATH", "data/processed")
    try:
        df = pd.read_parquet(data_path)
        print(f"Dati caricati da {data_path}")
    except Exception:
        fallback_path = os.getenv("RAW_DATA_PATH", "data/raw-data/sample_raw_data.csv")
        df = pd.read_csv(fallback_path)
        print(f"⚠️ Parquet non trovato. Fallback su CSV: {fallback_path}")

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

    # Prompt per il modello
    full_prompt = summary + f"# 🧠 Prompt Utente\n{prompt}\n"

    # API Key OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and not dev_mode:
        raise ValueError("❌ Variabile OPENAI_API_KEY non trovata.")

    # Generazione testo
    if dev_mode:
        text = "[DEV MODE] Report simulato.\n\n" + full_prompt
    else:
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

    # Scrittura del report su file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# 📝 Report Generativo\n\n{full_prompt}\n## 📋 Output AI\n{text}\n")

    print(f"✅ Report salvato in: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--output', type=str, default='report.txt')
    parser.add_argument('--dev-mode', action='store_true', help='Usa modalità simulata (senza API)')
    args = parser.parse_args()
    generate_text(args.prompt, args.output, args.dev_mode)