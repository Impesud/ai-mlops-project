FROM python:3.11-slim

# Install system dependencies (curl, openjdk, make, build-essential)
RUN apt-get update && apt-get install -y curl openjdk-17-jdk make build-essential && rm -rf /var/lib/apt/lists/*

# Install Spark
ENV SPARK_VERSION=3.5.6
ENV HADOOP_VERSION=3
#RUN curl -L https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz | tar -xz -C /opt/
RUN curl -L --retry 5 --retry-delay 10 https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz | tar -xz -C /opt/
ENV SPARK_HOME=/opt/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}
ENV PATH="${SPARK_HOME}/bin:${PATH}"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . /app
WORKDIR /app

# Keep container alive
CMD ["tail", "-f", "/dev/null"]


