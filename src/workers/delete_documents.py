import logging
from sqlalchemy import select
from src.core.database import SessionLocal
from src.models.source_documents import SourceDocument

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

def delete_documents(media_id : int):
    log.info(f'Recieved request to delete document with M_ID : {media_id}')
    session = SessionLocal()
    try:
        stmt = select(SourceDocument).where(SourceDocument.media_id == media_id)
        doc_to_delete = session.execute(stmt).scalars().first()
        if not doc_to_delete:
            log.warning(f'Document with M_ID {media_id} not found. Document might have already been deleted.')
            return
        session.delete(doc_to_delete)
        session.commit()
        log.info(f'Successfully deleted document (ID : {doc_to_delete.id}, M_ID : {doc_to_delete.media_id}) with all its chunks.')
    except Exception as e:
        log.error(f'An error occurred for document with M_ID : {media_id}: {e}')
        session.rollback()
    finally:
        session.close()


