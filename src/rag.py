import logging
from typing import List
from pydantic import BaseModel, Field

# LangChain and model imports
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from FlagEmbedding import FlagReranker

# Import from your other project files
from hchunk import HierarichicalLC, combine_retrieved_docs, DB_DIR
from load import load_from_document
# This import now assumes your reranker.py is correct
from reranker import rerank_documents_vn, VN_MODEL

# --- Configuration ---
MODEL_NAME = 'llama3.1:8b-instruct-q4_K_M'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Relevance Check Router ---
class RelevanceCheck(BaseModel):
    is_relevant: bool = Field(description="A boolean value indicating if the context can answer the query.")

def is_context_relevant(query: str, context: str, llm: ChatOllama) -> bool:
    parser = PydanticOutputParser(pydantic_object=RelevanceCheck)
    prompt = PromptTemplate(
        template="""You are an expert at evaluating the relevance of a given context to a user's query.
Your sole task is to determine if the provided context contains enough information to answer the query.
Respond with ONLY a JSON object formatted according to the instructions below.
{format_instructions}
Query: {query}
Context:
---
{context}
---
""",
        input_variables=["query", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | llm | parser
    try:
        result = chain.invoke({"query": query, "context": context})
        logging.info(f"Relevance check result: {result.is_relevant}")
        return result.is_relevant
    except Exception as e:
        logging.error(f"Error during relevance check: {e}")
        return False

# --- The Main RAG Pipeline ---
class RAG_Pipeline:
    def __init__(self, chunker: HierarichicalLC, llm: ChatOllama, reranker: FlagReranker):
        self.chunker = chunker
        self.llm = llm
        self.reranker = reranker

    def _retrieve_and_rerank(self, query: str) -> List[Document]:
        logging.info(f"Retrieving documents for query: '{query}'")
        retrieved_documents = combine_retrieved_docs(query = query, chunker = self.chunker)
        logging.info(f"Reranking documents...")
        reranked = rerank_documents_vn(
            question = query,
            docs = retrieved_documents,
            reranker = self.reranker
        )
        logging.info('Reranked documents')
        return reranked

    def ask_stream(self, query: str):
        # ... (This function remains the same as before)
        reranked_docs = self._retrieve_and_rerank(query)
        if not reranked_docs:
            yield "Không thể tìm thấy thông tin dựa vào câu hỏi."
            return
        context = "\n---\n".join([doc.page_content for doc in reranked_docs])
        if not is_context_relevant(query, context, self.llm):
            logging.warning("Không thể tìm thấy tài liệu liên quan dựa vào câu trả lời.")
            yield "Không trích xuất được tài liệu liên quan đến câu hỏi."
            return
        logging.info("Context is relevant. Proceeding to generation.")
        generation_template = """Bạn là một trợ lý hữu ích. Hãy trả lời câu hỏi của người dùng chỉ dựa vào ngữ cảnh sau đây.
                                Nếu câu trả lời không có trong ngữ cảnh, hãy nói rằng bạn không biết.
                                Hãy trả lời một cách chi tiết và đầy đủ, trích xuất tất cả các thông tin liên quan từ ngữ cảnh được cung cấp.

                                Ngữ cảnh:
                                ---
                                {context}
                                ---
                                Câu hỏi: {question}
                                """
        prompt = PromptTemplate.from_template(generation_template)
        chain = prompt | self.llm
        for chunk in chain.stream({"context": context, "question": query}):
            yield chunk.content

# --- Main Execution Block ---
if __name__ == "__main__":
    path_name = input('Path name : ')
    vietnamese_docs, _ = load_from_document(path_name)
    chunker = HierarichicalLC('vn_legal', DB_DIR)
    chunker.chunk_and_store(vietnamese_docs)
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    reranker = FlagReranker(VN_MODEL, use_fp16=True)
    pipeline = RAG_Pipeline(chunker, llm, reranker)
    while True:
        user_query = input('Query : ')
        if user_query.lower() == 'exit':
            break
        print(f"\n--- Querying with: '{user_query}' ---")
        full_response = ""
        for chunk in pipeline.ask_stream(user_query):
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n--- End of Response ---")
