import asyncio
import logging 
import os
import torch
from dotenv import load_dotenv
from typing import List, Optional
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker
from sqlalchemy import select

from src.core.database import AsyncSessionLocal
from src.models.chunks import Chunk, ChunkLevel
from src.models.source_documents import SourceDocument

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')
RERANKER_VN = os.getenv('VN_MODEL')
EMBEDDING_FN = HuggingFaceEmbeddings(
    model_name = EMBEDDING_MODEL,
    model_kwargs = {'device' : 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs = {'normalize_embeddings' : True}
)

RERANKER = FlagReranker(RERANKER_VN, use_fp16=True)

def rerank_documents_vn(question: str, docs: List[Document], reranker : FlagReranker ,top_k = 10) -> list[Document]:
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.compute_score(pairs)
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_score_pairs[:top_k]]

async def retrieval_and_rerank(query : str, media_id : Optional[int] = None, k : int = 25, top_k : int = 10) -> List[Document] :
    log.info(f"Starting retrieval for query {query}")
    if media_id:
        log.info(f'Filtering by M_ID : {media_id}')
    query_embedding = await EMBEDDING_FN.aembed_query(query)

    parent_ids = set()
    parent_chunks = []
    async with AsyncSessionLocal() as asession:
        stmt = select(Chunk).where(Chunk.chunk_level == ChunkLevel.CHILD)

        if media_id is not None:
            stmt = stmt.join(SourceDocument).where(SourceDocument.media_id == media_id)

        child_stmt = stmt.order_by(Chunk.embedding.l2_distance(query_embedding)).limit(k)
        child_result = await asession.execute(child_stmt)
        child_chunks = child_result.scalars().all()

        for chunk in child_chunks:
            if chunk.parent_id:
                parent_ids.add(chunk.parent_id)
        if not parent_ids:
            log.warning(f'No parent chunks found for retrieved {len(child_chunks)} child chunks')
            return
        
        parent_stmt = select(Chunk).where(Chunk.id.in_(list(parent_ids)))
        parent_results = await asession.execute(parent_stmt)
        parent_chunks_table = parent_results.scalars().all()
        parent_chunks = [Document(page_content=chunk.content, metadata = chunk.chunk_metadata) for chunk in parent_chunks_table]
        all_chunks = parent_chunks + [Document(page_content=chunk.content, metadata = chunk.chunk_metadata) for chunk in child_chunks]
        log.info(f'Retrieved a total of {len(all_chunks)} chunks for reranking.')

    reranked_docs = await asyncio.to_thread(
        rerank_documents_vn, query, all_chunks, RERANKER, top_k
    )
    log.info(f'Reranked to the top {len(reranked_docs)} chunks.')
    return reranked_docs    





