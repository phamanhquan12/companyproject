FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn

COPY ./src /app/src 
COPY ./api/srcp /app/api/srcp


ENV PYTHONPATH=/app

CMD ["uvicorn", "api.srcp.main:app", "--host", "0.0.0.0", "--port", "8000"]