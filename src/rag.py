import logging
import json
import time
import os
from datetime import datetime
from uuid import uuid4
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from FlagEmbedding import FlagReranker

from hchunk import HierarichicalLC, combine_retrieved_docs, DB_DIR
from load import load_from_document
from reranker import rerank_documents_vn, VN_MODEL

MODEL_NAME = 'llama3.1:8b'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RiskLogger:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path

    def log(self, data: Dict[str, Any]):
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent = 4) + '\n\n')

class RelevanceCheck(BaseModel):
    relevance_score: int = Field(description="An integer value between 0 and 10 representing the relevance of the context to the query.")

def is_context_relevant(query: str, context: str, llm: ChatOllama, threshold: int = 7) -> bool:
    # Use the corrected Pydantic object
    parser = PydanticOutputParser(pydantic_object=RelevanceCheck)
    
    prompt = PromptTemplate(
        template="""You are an expert at evaluating the relevance of a given context to a user's query.
Your sole task is to determine if the provided context contains enough information to answer the query.
Respond with ONLY a JSON object containing a single key 'relevance_score' with an integer value between 0 (not relevant at all) and 10 (perfectly relevant).
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
        score = result.relevance_score
        logging.info(f"Relevance score: {score}/{10}")
        return score >= threshold
    except Exception as e:
        logging.error(f"Error during relevance check: {e}")
        return False

class RAG_Pipeline:
    def __init__(self, chunker: HierarichicalLC, llm: ChatOllama, reranker: FlagReranker, logger: RiskLogger):
        self.chunker = chunker
        self.llm = llm
        self.reranker = reranker
        self.logger = logger

    def _retrieve_and_rerank(self, query: str) -> List[Document]:
        logging.info(f"Retrieving documents for query: '{query}'")
        retrieved_documents = combine_retrieved_docs(query=query, chunker=self.chunker)
        logging.info(f"Reranking {len(retrieved_documents)} documents...")
        if not retrieved_documents:
            return []
        reranked = rerank_documents_vn(
            question=query,
            docs=retrieved_documents,
            reranker=self.reranker,
            top_k=10
        )
        logging.info(f'Reranked to {len(reranked)} documents.')
        return reranked, retrieved_documents

    def _refine_answer(self, query: str, context: str, initial_answer: str):
        logging.info("Refining the initial answer...")
        refinement_template = """Bạn là một chuyên gia kiểm soát chất lượng. Nhiệm vụ của bạn là xem xét một câu trả lời được tạo ra và cải thiện nó dựa trên NGỮ CẢNH được cung cấp.

        Hãy làm theo các bước sau:
        1.  **So sánh câu trả lời ban đầu với ngữ cảnh.** Câu trả lời có chính xác không? Nó có bỏ sót thông tin quan trọng nào từ ngữ cảnh không? **Câu trả lời có trích dẫn nguồn và trang chính xác cho mỗi luận điểm không?**
        2.  **Viết lại câu trả lời để cải thiện nó.** Hãy làm cho nó đầy đủ, chính xác, và hoàn toàn dựa trên ngữ cảnh. Sửa chữa mọi sai sót. **Thêm hoặc sửa các trích dẫn (Số Trang) để chúng chính xác.**
        3.  **Chỉ trả ra câu trả lời cuối cùng, đã được cải thiện.** Không giải thích các bước của bạn.

        Ngữ cảnh:
        ---
        {context}
        ---
        Câu hỏi: {question}
        ---
        Câu trả lời ban đầu: {initial_answer}
        ---
        Câu trả lời đã được cải thiện của bạn:
        """
        prompt = PromptTemplate.from_template(refinement_template)
        chain = prompt | self.llm | StrOutputParser()
        
        refined_answer = chain.invoke({
            "context": context,
            "question": query,
            "initial_answer": initial_answer
        })
        return refined_answer

    def ask_stream(self, query: str, session_id: str):
        start_time = time.time()
        log_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "query": query
        }

        reranked_docs, retrieved_docs = self._retrieve_and_rerank(query)
        log_data['combine_retrieved_documents'] = [doc.page_content for doc in retrieved_docs]
        # log_data["reranked_doc_metadata"] = [doc.metadata for doc in reranked_docs]

        
        if not reranked_docs or not retrieved_docs:
            yield "Không thể tìm thấy thông tin dựa vào câu hỏi."
            log_data["final_answer"] = "No documents found."
            log_data["latency"] = time.time() - start_time
            self.logger.log(log_data)
            return
        context_sources = []
        for doc in reranked_docs:
            page = doc.metadata.get('page', 'N/A')
            content = doc.page_content
            context_source = f'''
                            Trang : {page},
                            Nội dung : {content}
                                '''
            context_sources.append(context_source)
        
        context = "\n\n==============\n\n".join(context_sources)
        log_data["context"] = context

        if not is_context_relevant(query, context, self.llm):
            logging.warning("Context deemed not relevant by the router.")
            yield "Không trích xuất được tài liệu liên quan đến câu hỏi."
            log_data["final_answer"] = "Context not relevant."
            log_data["latency"] = time.time() - start_time
            self.logger.log(log_data)
            return
            
        logging.info("Context is relevant. Proceeding to generation.")
        
        generation_template = """Bạn là một trợ lý hữu ích. 
        Hãy trả lời câu hỏi của người dùng chỉ dựa vào NGỮ CẢNH được cung cấp.
        
        Quy tắc:
        - Trình bày chi tiết và đầy đủ, trích xuất tất cả thông tin liên quan trong ngữ cảnh.
        - **QUAN TRỌNG: LUÔN LUÔN trích dẫn số trang cho MỖI thông tin bạn cung cấp bằng cách sử dụng định dạng (Trang: [số]).**
        - Nếu câu trả lời không có trong ngữ cảnh, hãy nói rằng bạn không biết.
        - Khi ngữ cảnh có dạng bảng (a | b), không biểu diễn dưới dạng (a|b) mà hãy chú thích và giải thích kĩ bên dưới với dạng gạch đầu dòng.
        - Không được thêm thông tin nằm ngoài ngữ cảnh.
        - Luôn trả lời bằng CÙNG NGÔN NGỮ với câu hỏi gốc.

        Ngữ cảnh:
        ---
        {context}
        ---
        Câu hỏi: {question}
        """
        prompt = PromptTemplate.from_template(generation_template)
        chain = prompt | self.llm | StrOutputParser()
        
        initial_answer = chain.invoke({"context": context, "question": query})
        log_data["initial_answer"] = initial_answer
        
        refined_answer = self._refine_answer(
            query=query,
            context=context,
            initial_answer=initial_answer
        )
        log_data["final_answer"] = refined_answer
        
        yield refined_answer

        end_time = time.time()
        log_data["latency"] = end_time - start_time
        self.logger.log(log_data)
        logging.info(f"\nQuery processed in {log_data['latency']:.2f} seconds.")

if __name__ == "__main__":
    session_id = str(uuid4())
    print(f"Starting new session: {session_id}")

    risk_log_file = os.path.join(os.path.dirname(__file__), 'risk_management_log2.jsonl')
    logger = RiskLogger(risk_log_file)

    path_name = 'true_test'
    vietnamese_docs, _ = load_from_document(path_name)
    chunker = HierarichicalLC('denso_main', DB_DIR)
    chunker.chunk_and_store(vietnamese_docs)
    llm = ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=8192, num_predict=4096)
    reranker = FlagReranker(VN_MODEL, use_fp16=True)

    pipeline = RAG_Pipeline(chunker, llm, reranker, logger)

    while True:
        user_query = input('Query : ')
        if user_query.lower() == 'exit':
            break
        print(f"\n--- Querying with: '{user_query}' ---")
        full_response = ""
        for chunk in pipeline.ask_stream(user_query, session_id=session_id):
            print(chunk, end="", flush=True)
            full_response += chunk
        print()
        print("\n--- End of Response ---")