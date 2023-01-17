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
RUN apt-get install ffmpeg libsm6 libxext6 -y

ENV LIBGIT_VERSION 1.1.0

# Downloading and building libgit2
RUN apt-get install -y cmake \
    && wget https://github.com/libgit2/libgit2/archive/v${LIBGIT_VERSION}.tar.gz \
    && tar xzf v${LIBGIT_VERSION}.tar.gz \
    && cd libgit2-${LIBGIT_VERSION} \
    && cmake . \ 
    && make \
    && make install \
    && pip install pygit2 \
    && ldconfig

RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./app /app
COPY models/ /app/models/
WORKDIR /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
