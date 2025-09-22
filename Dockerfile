# 1. Start from a slim Python base image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# 4. Copy and install Python requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your source data and application code into the image
COPY ./data ./data
COPY ./src ./src

# The command to run the app (e.g., uvicorn or celery) will be
# provided by your docker-compose.yml file.