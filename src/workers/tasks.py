from .celery_app import celery_app
from .processing import process_document
from .delete_documents import delete_documents
import logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
@celery_app.task
def process_document_task(file_name : str, media_id: int, format : str = 'pdf'):
    process_document(file_name = file_name, media_id = media_id, format = format)
    return logging.info(f'Processing task started for {file_name} (M_ID : {media_id})')

@celery_app.task
def delete_document_task(media_id : int):
    delete_documents(media_id = media_id)
    return logging.info(f"Deleting task started for file with M_ID : {media_id}")

