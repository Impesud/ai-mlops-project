FROM python:3.11-slim

WORKDIR /app

# Copia tutto il progetto
COPY . .

# Installa MLflow e requirements del progetto
RUN pip install --upgrade pip && \
    pip install mlflow && \
    pip install -r requirements.txt

# Espone la porta 5000 (MLflow UI)
EXPOSE 5000

# Entrypoint per avviare il server
ENTRYPOINT ["bash", "mlops/entrypoint.sh"]