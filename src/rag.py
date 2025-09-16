import logging
from typing import List
from pydantic import BaseModel, Field

# LangChain and model imports
from langchain_core.documents import Document
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from FlagEmbedding import FlagReranker

# Import from your other project files
from hchunk import HierarichicalLC, combine_retrieved_docs, DB_DIR
from load import load_from_document
# This import now assumes your reranker.py is correct
from reranker import rerank_documents_vn, VN_MODEL

# --- Configuration ---
MODEL_NAME = 'llama3.1:8b'
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
            reranker = self.reranker,
            top_k = 20
        )
        logging.info('Reranked documents')
        return reranked

    def ask_stream(self, query: str):
        reranked_docs = self._retrieve_and_rerank(query)
        if not reranked_docs:
            yield "Không thể tìm thấy thông tin dựa vào câu hỏi."
            return
        context = "\n---\n".join([doc.page_content for doc in reranked_docs])
        if not is_context_relevant(query, context, self.llm):
            logging.warning("Không thể tìm thấy tài liệu liên quan dựa vào câu trả lời.")
            yield "Không trích xuất được tài liệu liên quan đến câu hỏi."
            return
        logging.info("Context is relevant. Proceeding to first generation.")
        extraction_template = """Bạn là một trợ lý tìm kiếm.
        Nhiệm vụ:
        - Chỉ dựa vào NGỮ CẢNH dưới đây để tạo ra một câu trả lời NGẮN GỌN và CHÍNH XÁC, tập trung vào từ khóa quan trọng.
        - Nếu không tìm thấy câu trả lời trong ngữ cảnh, hãy đưa ra MỘT câu hỏi được viết lại rõ ràng hơn từ câu hỏi gốc, để cải thiện việc tìm kiếm.
        - Luôn sử dụng CÙNG NGÔN NGỮ với câu hỏi gốc (không dịch sang ngôn ngữ khác).
        - Không được thêm thông tin ngoài ngữ cảnh.
        - CHỈ ĐƯA RA CÂU TRẢ LỜI MỘT LẦN DUY NHẤT
        Ngữ cảnh:
        ----
        {context}
        ----
        Câu hỏi: {question}
        """
        extraction_chain = PromptTemplate.from_template(extraction_template) | self.llm | StrOutputParser()
        short_answer_query = extraction_chain.invoke({"context" : context, "question" : query}).strip()
        logging.info(f"Query expanded : {short_answer_query}")
        logging.info("Refining Answer....")
        final_docs = self._retrieve_and_rerank(short_answer_query)
        if not final_docs:
            final_docs = reranked_docs
        final_context = "\n---\n".join([doc.page_content for doc in final_docs])
        generation_template = """Bạn là một trợ lý hữu ích. 
        Hãy trả lời câu hỏi của người dùng chỉ dựa vào NGỮ CẢNH được cung cấp.
        
        Quy tắc:
        - Nếu câu trả lời không có trong ngữ cảnh, hãy nói rằng bạn không biết.
        - Khi ngữ cảnh có dạng bảng (a | b), không biểu diễn dưới dạng (a|b) mà hãy chú thích và giải thích kĩ bên dưới với dạng gạch đầu dòng.
        - Nếu có thể, đưa ra trích dẫn (nguồn, số trang trong tài liệu) CHÍNH XÁC để ủng hộ câu trả lời.
          Nếu không có trích dẫn, đừng bịa đặt hoặc thêm trích dẫn giả.
        - Không được thêm thông tin nằm ngoài ngữ cảnh.
        - Luôn trả lời bằng CÙNG NGÔN NGỮ với câu hỏi gốc.
        - Trình bày chi tiết và đầy đủ, trích xuất tất cả thông tin liên quan trong ngữ cảnh.

        Ngữ cảnh:
        ---
        {context}
        ---
        Câu hỏi: {question}
        """
        prompt = PromptTemplate.from_template(generation_template)
        chain = prompt | self.llm
        for chunk in chain.stream({"context": final_context, "question": query}):
            yield chunk.content

# --- Main Execution Block ---
if __name__ == "__main__":
    path_name = 'true_test'
    vietnamese_docs, _ = load_from_document(path_name)
    chunker = HierarichicalLC('denso_main2', DB_DIR)
    chunker.chunk_and_store(vietnamese_docs)
    llm = ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=8192, num_predict=2048 )
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
