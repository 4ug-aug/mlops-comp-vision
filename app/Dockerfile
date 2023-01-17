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
RUN pip install -e .

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# install libgit
RUN apt-get add --no-cache libgit2
RUN apt-get add --no-cache libgit2-dev


RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./app /app
COPY models/ /app/models/
WORKDIR /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
