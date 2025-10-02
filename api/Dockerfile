FROM python:3.11-slim
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
RUN pip install --no-cache-dir fastapi uvicorn[standard]

COPY ./src /app/src
COPY ./api/srcp /app/api/srcp

ENV PYTHONPATH=/app

CMD ["uvicorn", "api.srcp.main:app", "--host", "0.0.0.0", "--port", "8000"]