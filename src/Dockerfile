FROM python:3.11-slim

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

# Corrected path to requirements.txt
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Be more specific about what you copy
COPY ./src /app/src

ENV PYTHONPATH=/app