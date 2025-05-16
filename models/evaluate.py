# Evaluation script stub
import mlflow
import pandas as pd

def evaluate(model_uri: str, test_data: str):
    model = mlflow.sklearn.load_model(model_uri)
    df = pd.read_parquet(test_data)
    X = df.drop('label', axis=1)
    y = df['label']
    preds = model.predict(X)
    # add metrics here

if __name__ == "__main__":
    evaluate('runs:/<RUN_ID>/model', 'data/processed/data.parquet')