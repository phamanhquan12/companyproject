from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from FlagEmbedding import FlagReranker
import logging
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
JP_MODEL = 'hotchpotch/japanese-reranker-cross-encoder-large-v1'
VN_MODEL = 'namdp-ptit/ViRanker'
def rerank_documents_vn(question: str, docs: List[Document], reranker : FlagReranker ,top_k = 10) -> list[Document]:
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.compute_score(pairs)
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_score_pairs[:top_k]]

def rerank_documents_jp(question: str, docs: List[Document], reranker : CrossEncoder ,top_k = 5) -> list[Document]:
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    doc_score_pairs = list(zip(docs, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, score in doc_score_pairs[:top_k]]

