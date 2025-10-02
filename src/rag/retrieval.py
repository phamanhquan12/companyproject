import asyncio
import logging
import os
from collections import Counter
from dotenv import load_dotenv
from typing import List, Optional
from langdetect import detect, LangDetectException
from langchain_core.documents import Document
from FlagEmbedding import FlagReranker
from sqlalchemy import select
from sqlalchemy.orm import aliased

from src.core.database import AsyncSessionLocal
from src.models.chunks import Chunk, ChunkLevel
from src.models.source_documents import SourceDocument
from src.workers.processing import EMBEDDING_FN

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
RERANKER_MUL = os.getenv('RERANKER_MODEL')
RERANKER_VN_MODEL = os.getenv('VN_MODEL')

RERANKER_VN = FlagReranker(
    RERANKER_VN_MODEL, use_fp16=True
)


def rerank_documents_vn(question: str, docs: List[Document], reranker: FlagReranker, top_k=10) -> list[Document]:
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.compute_score(pairs)
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_score_pairs[:top_k]]


async def initial_retrieval(query: str, top_k_chunks: int = 100, top_k_ids: int = 3):
    log.info(f"Initial retrieval for query : {query}")
    query_embedding = await EMBEDDING_FN.aembed_query(query)
    async with AsyncSessionLocal() as asession:
        stmt = select(Chunk).order_by(
            Chunk.embedding.cosine_distance(query_embedding)
        ).limit(top_k_chunks)
        results = await asession.execute(stmt)
        chunks = results.scalars().all()

        if not chunks:
            log.warning(f"Initial retrieval found no chunks to map ids.")
            return []
        source_doc_ids = [chunk.source_doc_id for chunk in chunks]

        media_id_stmt = select(SourceDocument.media_id).where(SourceDocument.id.in_(source_doc_ids))
        media_id_res = await asession.execute(media_id_stmt)
        media_ids = media_id_res.scalars().all()

        if not media_ids:
            log.warning(f"Retrieved chunks, but can not find ids.")
            return []

        media_ids_count = Counter(media_ids)
        most_cmm_ids = [media_id for media_id, count in media_ids_count.most_common(top_k_ids)]
        log.info(f"Possible ids related to query : {most_cmm_ids}")
        return most_cmm_ids


async def retrieval_and_rerank(query: str, media_id: Optional[int] = None, k: int = 25, top_k: int = 10) -> List[
    Document]:
    log.info(f"Starting retrieval for query {query}")
    if media_id:
        log.info(f'Filtering by M_ID : {media_id}')
        top_media_ids = [media_id]
    else:
        top_media_ids = await initial_retrieval(query)
        if not top_media_ids:
            return []
            
    query_embedding = await EMBEDDING_FN.aembed_query(query)

    parent_ids = set()
    all_chunks = []
    async with AsyncSessionLocal() as asession:
        stmt = select(Chunk).where(Chunk.chunk_level == ChunkLevel.CHILD)

        stmt = stmt.join(SourceDocument).where(SourceDocument.media_id.in_(top_media_ids))

        child_stmt = stmt.order_by(Chunk.embedding.cosine_distance(query_embedding)).limit(k)
        child_result = await asession.execute(child_stmt)
        child_chunks = child_result.scalars().all()

        for chunk in child_chunks:
            if chunk.parent_id:
                parent_ids.add(chunk.parent_id)
        parent_stmt_base = select(Chunk).where(
            Chunk.chunk_level == ChunkLevel.PARENT
        ).join(SourceDocument).where(SourceDocument.media_id.in_(top_media_ids))

        parent_stmt_direct = parent_stmt_base.order_by(
            Chunk.embedding.cosine_distance(query_embedding)
        ).limit(k)
        parent_stmt_direct_results = await asession.execute(parent_stmt_direct)
        parent_stmt_direct_chunks = parent_stmt_direct_results.scalars().all()

        parent_ids = parent_ids.union({p.id for p in parent_stmt_direct_chunks})
        if not parent_ids:
            log.warning(f'No parent chunks found for retrieved {len(child_chunks)} child chunks')
            if not child_chunks:
                return []
            all_chunks = [Document(page_content=chunk.content, metadata=chunk.chunk_metadata) for chunk in child_chunks]
        else:
            parent_stmt = select(Chunk).where(Chunk.id.in_(list(parent_ids)))
            parent_results = await asession.execute(parent_stmt)
            parent_chunks_table = parent_results.scalars().all()
            parent_chunks = [Document(page_content=chunk.content, metadata=chunk.chunk_metadata) for chunk in
                             parent_chunks_table]
            all_chunks = parent_chunks + [Document(page_content=chunk.content, metadata=chunk.chunk_metadata) for chunk
                                          in child_chunks]

        log.info(f'Retrieved a total of {len(all_chunks)} chunks for reranking.')
    reranked_docs = await asyncio.to_thread(
        rerank_documents_vn, query, all_chunks, RERANKER_VN, top_k
    )
    log.info(f'Reranked to the top {len(reranked_docs)} chunks.')
    return reranked_docs