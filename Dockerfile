# Base image
FROM python:3.8.5-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy essential files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY models/ models/
COPY reports/ reports/
COPY data/ data/

# Install dependencies
WORKDIR /
RUN pip install -e .
RUN pip install -r requirements.txt --no-cache-dir

# Entrypoint
ENTRYPOINT ["python", "-u", "src/models-lightning/train_model.py"]