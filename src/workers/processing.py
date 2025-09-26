import os
import re
import logging
import torch
import enum

from dotenv import load_dotenv
from datetime import datetime, timezone
from uuid import uuid4
from sqlalchemy import select, or_
from typing import List, Optional

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, MarkdownTextSplitter, TextSplitter
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
class DocType(enum.Enum):
    STRUCTURED = "STRUCTURED"
    UNSTRUCTURED = "UNSTRUCTURED"

def classify_document(docs: List[Document], threshold : float = .1) -> DocType:
    full_text = "".join([doc.page_content for doc in docs])
    if not full_text.strip():
        return DocType.UNSTRUCTURED
    legal_pattern = re.compile(r'(?i)\b(Điều|Article|Section)\s*\d+|第\s*\d+\s*条')
    legal_keyword_count = len(legal_pattern.findall(full_text))
    header_count = len(re.findall(r'^#+\s', full_text, re.MULTILINE))
    table_line_count = full_text.count('|') // 2
    if legal_keyword_count < 3 and header_count < 5:
        log.info("Document has few headers and legal keywords. Classifying as UNSTRUCTURED.")
        return DocType.UNSTRUCTURED
    score = 0
    score += legal_keyword_count * 2.0  # High weight for specific keywords
    score += header_count * 0.5
    score += table_line_count * 1.0
    
    # Normalize the score by the number of lines to get a density score
    num_lines = full_text.count('\n') + 1
    density_score = score / num_lines if num_lines > 0 else 0
    
    log.info(f"Legal Keywords: {legal_keyword_count}, Headers: {header_count}, Table Lines: {table_line_count}")
    log.info(f"Num lines: {num_lines}")
    log.info(f"Document Structure Score: {density_score:.4f} (Threshold: {threshold})")

    if density_score > threshold:
        log.info("Document classified as STRUCTURED.")
        return DocType.STRUCTURED
    else:
        log.info("Document classified as UNSTRUCTURED.")
        return DocType.UNSTRUCTURED
    

def _chunk_structured_document(docs: List[Document]) -> List[Document]:
    BILINGUAL_LEGAL_SEPARATORS = [
        "\n\nChương ", "\n\nChapter ", "\n\nPart ",
        "\n\nĐiều ", "\n\nArticle ", "\n\nSection ", "\n\nSec. ", "\n\nMục ",
        r"\n\d+\.\s", r"\n[a-z]\)\s", r"\n\(\d+\)\s", r"\n\([a-z]\)\s",
        "\n\n", "\n", ". ", " "
    ]
    
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,   
        chunk_overlap=120,    
        separators=BILINGUAL_LEGAL_SEPARATORS,
        keep_separator = True
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=[r"\n\d+\.\s", r"\n[a-z]\)\s", r"\n\(\d+\)\s", r"\n\([a-z]\)\s",
        "\n\n", "\n", ". ", " "],
        keep_separator=True
    )

    parent_chunks = parent_splitter.split_documents(docs)
    log.info(f"Split document into {len(parent_chunks)} parent chunks.")

    all_chunks = []
    for p_doc in parent_chunks:
        parent_id = str(uuid4())
        p_doc.metadata['id'] = parent_id
        p_doc.metadata['chunk_level'] = ChunkLevel.PARENT
        all_chunks.append(p_doc)

        child_docs = child_splitter.split_documents([p_doc])
        for c_doc in child_docs:
            c_doc.metadata['id'] = str(uuid4())
            c_doc.metadata['chunk_level'] = ChunkLevel.CHILD
            c_doc.metadata['parent_id'] = parent_id 
            all_chunks.append(c_doc)
            
    log.info(f"Created a total of {len(all_chunks)} chunks ({len(parent_chunks)} parents).")
    return all_chunks


def _chunk_semantic_document(docs: List[Document]) -> List[Document]:
    semantic_splitter = SemanticChunker(
        embeddings=EMBEDDING_FN, 
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=90
    )
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
    parent_chunks = semantic_splitter.split_documents(docs)
    
    all_chunks = []
    for p_doc in parent_chunks:
        parent_id = str(uuid4())
        p_doc.metadata['id'] = parent_id
        p_doc.metadata['chunk_level'] = ChunkLevel.PARENT
        all_chunks.append(p_doc)

        child_docs = child_splitter.split_documents([p_doc])
        for c_doc in child_docs:
            c_doc.metadata['id'] = str(uuid4())
            c_doc.metadata['chunk_level'] = ChunkLevel.CHILD
            c_doc.metadata['parent_id'] = parent_id
            all_chunks.append(c_doc)
            
    log.info(f"Created {len(parent_chunks)} parent chunks and {len(all_chunks) - len(parent_chunks)} child chunks using semantic strategy.")
    return all_chunks


def process_document(file_name : str, media_id : int, format : str = 'pdf'):
    log.info(f"--- Starting processing for: {file_name} ---")
    dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'processed')
    file_path = os.path.join(dir_path, f'{file_name}.pdf')
    
    with SessionLocal() as session:
        try:
            stmt = select(SourceDocument).where(or_(
                SourceDocument.media_id == media_id, SourceDocument.file_path == file_path)
            )
            existing_doc = session.execute(stmt).scalars().first()
            if existing_doc:
                log.warning(f"Document with M_ID {media_id} already exist. Skipping process.")
                return
        except Exception as e:
            log.error(f"Unexpected error : {e}")
            return
    try:
        docs_from_file = load_from_document(file_name)
        if not docs_from_file:
            log.warning(f"No content extracted. Aborting...")
            return
    except Exception as e:
        log.error(f"Failed to load documents : {e}")
        return
    with SessionLocal() as session:
        try:
            source_docs = SourceDocument(
                media_id = media_id,
                file_name = file_name,
                file_path = file_path,
                page_count = len(docs_from_file),
                status = IngestStatus.PROCESSING,
                created_at = datetime.now(timezone.utc)
            )
            session.add(source_docs)
            session.commit()
            session.refresh(source_docs)
            log.info(f"Created source documents with M_ID : {media_id}")
            

            doc_type = classify_document(docs_from_file)
            all_chunks = _chunk_structured_document(docs_from_file) if doc_type == DocType.STRUCTURED else _chunk_semantic_document(docs_from_file)
            chunks_content = [doc.page_content for doc in all_chunks]
            chunks_embedded = EMBEDDING_FN.embed_documents(chunks_content)

            all_db_chunks = []
            for i, chunk in enumerate(all_chunks):
                all_db_chunks.append(
                    Chunk(
                        id = chunk.metadata.get('id'),
                        content = chunk.page_content,
                        chunk_level = chunk.metadata.get('chunk_level'),
                        embedding = chunks_embedded[i],
                        chunk_metadata = {'page' : chunk.metadata.get('page') + 1},
                        source_doc_id = source_docs.id,
                        parent_id = chunk.metadata.get('parent_id')
                    )
                )
            session.add_all(all_db_chunks)
            source_docs.status = IngestStatus.COMPLETED
            source_docs.processed_at = datetime.now(timezone.utc)
            session.commit()
            log.info(f"Successfully stored object M_ID {media_id} with {len(all_db_chunks)} chunks.")
        except Exception as e:
            log.error(f"Error occurred during DB operations for {file_name} : {e}")
            session.rollback()
            if source_docs and source_docs.id:
                failed_docs = session.get(SourceDocument, SourceDocument.id)
                if failed_docs:
                    failed_docs.status = IngestStatus.FAILED
                    session.commit()
        finally:
            session.close()

