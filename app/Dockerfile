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

CMD exec uvicorn app.main:app --port $PORT --host 0.0.0.0 --workers 1




