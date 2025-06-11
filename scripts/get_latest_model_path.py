import os
import glob

def find_latest_model_artifact():
    # Find all "model" directories in mlruns (standard MLflow run artifact)
    matches = glob.glob("mlruns/*/*/artifacts/model", recursive=True)

    # If not found, look for Model Registry artifacts
    if not matches:
        matches = glob.glob("mlruns/*/models/*/artifacts", recursive=True)

    if not matches:
        raise FileNotFoundError("No MLflow model directory found in mlruns/*/*/artifacts/model or mlruns/*/models/*/artifacts")

    # Sort by last modification time
    latest_model_dir = max(matches, key=os.path.getmtime)

    # Print only the absolute path, no extra text
    print(os.path.abspath(latest_model_dir))

if __name__ == "__main__":
    find_latest_model_artifact()



