from setuptools import find_packages, setup

setup(
    name="ai_mlops_project",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "pyspark==3.5.5",
        "pyyaml",
        "mlflow",
        "scikit-learn",
        "pandas",
        "pyarrow",
        "boto3",
        "imbalanced-learn",
        "openai",
        "cerberus",
        "seaborn>=0.12",
    ],
    python_requires=">=3.10",
)
