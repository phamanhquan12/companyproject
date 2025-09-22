import os
from dotenv import load_dotenv
from celery import Celery
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL")
celery_app = Celery(
    'tasks',
    broker= REDIS_URL,
    backend= REDIS_URL,
    include=['src.workers.tasks']
)
