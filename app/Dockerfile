FROM python:3.9-slim

EXPOSE $PORT

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy essential files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY ./app /app
COPY models/ /app/models/
COPY data/raw/names.txt names.txt

RUN pip install -e .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install --no-cache-dir --upgrade -r requirements.txt

ARG OTEL_EXPORTER_OTLP_ENDPOINT="http://0.0.0.0:4318/"
ARG OTEL_SERVICE_NAME="butterfly-classification-api"

CMD ["uvicorn", "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "80:80"]
