from setuptools import setup, find_packages

setup(
    name='ai_mlops_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pyspark',
        'pyyaml',
        'mlflow',
        'scikit-learn',
        'pandas',
        'pyarrow',
        'openai'
    ],
)
