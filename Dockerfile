# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy essential files
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# Install dependencies
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# Entrypoint
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
