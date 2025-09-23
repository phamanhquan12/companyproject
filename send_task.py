# send_task.py
from src.workers.tasks import process_document_task

# --- Configure your test task ---
TEST_FILE_NAME = "luong"
TEST_MEDIA_ID = 100 # Use a new, unique media_id

print(f"Sending ingestion task for {TEST_FILE_NAME}...")

# .delay() sends the job to the Redis queue without blocking
process_document_task.delay(
    file_name=TEST_FILE_NAME,
    media_id=TEST_MEDIA_ID
)

print("Task sent to Celery worker!")