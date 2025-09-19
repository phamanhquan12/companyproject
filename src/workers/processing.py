import os
import logging
import torch
from dotenv import load_dotenv
from datetime import datetime, timezone
from uuid import uuid4
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from src.core.database import SessionLocal
from src.models.source_documents import SourceDocument, IngestStatus
from src.models.chunks import Chunk, ChunkLevel
from src.load import load_from_document
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
load_dotenv()

EMBEDDING_FN = HuggingFaceEmbeddings(
    model_name = os.getenv('EMBEDDING_MODEL'),
    model_kwargs = {'device' : 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs = {'normalize_embeddings' : True}
)

def process_document(file_name : str, media_id : int, format : str = 'pdf'):
    log.info(f"--- Starting processing for: {file_name} ---")
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'processed')
    try:
        vietnamese_docs, _ = load_from_document(file_name)
        if not vietnamese_docs:
            log.warning(f'No content extracted from {file_name}.')
            return
    except Exception as e:
        log.error(f'Failed to load document {file_name}. Error : {e}')
        return
    
    with SessionLocal() as session:
        source_doc = None
        try:
            source_doc = SourceDocument(
                media_id = media_id,
                file_name = file_name,
                file_path = os.path.join(dir_path, f'{file_name}.{format}'),
                page_count = len(vietnamese_docs),
                status = IngestStatus.PROCESSING
            )
            session.add(source_doc)
            session.commit()
            session.refresh(source_doc)
            source_document_id = source_doc.id 
            source_doc_media_id = source_doc.media_id
            log.info(f"Created record document with initial id {source_document_id} and media id {source_doc_media_id}")

            semantic_splitter = SemanticChunker(
                embeddings=EMBEDDING_FN,
                breakpoint_threshold_type='percentile',
                breakpoint_threshold_amount=85
            )
            parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 1200,
                chunk_overlap = 200,
                separators = ['\n\n','\n', '.', ' ', ''],
                keep_separator = True
            )  
            child_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 400,
                chunk_overlap = 40,
                separators = ['\n\n','\n', '.', ' ', ''],
                keep_separator = True
            )  
            all_chunks = []
            parent_docs = parent_splitter.split_documents(vietnamese_docs)
            for p_doc in parent_docs:
                parent_id = str(uuid4())
                p_doc.metadata['id'] = parent_id
                p_doc.metadata['chunk_level'] = ChunkLevel.PARENT
                all_chunks.append(p_doc)
                
                child_docs = child_splitter.split_documents([p_doc])
                for c_doc in child_docs:
                    child_id = str(uuid4())
                    c_doc.metadata['id'] = child_id
                    c_doc.metadata['chunk_level'] = ChunkLevel.CHILD
                    c_doc.metadata['parent_id'] = parent_id
                    all_chunks.append(c_doc) #test this
            chunk_contents = [doc.page_content for doc in all_chunks]
            chunk_embeddings = EMBEDDING_FN.embed_documents(chunk_contents)
            log.info(f'Generated {len(chunk_embeddings)} embeddings of size 1024.')

            all_db_chunks = []
            for i, chunk in enumerate(all_chunks):
                all_db_chunks.append(
                    Chunk(
                        id = chunk.metadata['id'],
                        content = chunk.page_content,
                        chunk_level = chunk.metadata['chunk_level'],
                        chunk_metadata = {
                            'page_num' : chunk.metadata.get('page'),
                            'language' : chunk.metadata.get('language')
                        },
                        embedding = chunk_embeddings[i],
                        source_doc_id = source_document_id,
                        parent_id = chunk.metadata.get('parent_id', None)


                    )
                )
            session.add_all(all_db_chunks)
            source_doc.status = IngestStatus.COMPLETED
            source_doc.processedAt = datetime.now(timezone.utc)
            session.commit()
        except Exception as e:
            log.error(f'An error occurred during ingestion for {file_name} : {e}')
            session.rollback()
            if source_doc and source_doc.id:
                failed = session.get(SourceDocument, source_doc.id)
                if failed:
                    failed.status = IngestStatus.FAILED
                    session.commit()
                    log.info(f'Updated status to FAILED for ID : {source_doc.id}')
        finally:
            session.close()