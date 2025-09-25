import os
import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langdetect import detect, LangDetectException
from .retrieval import retrieval_and_rerank
from .definitions import RelevanceCheck, RELEVANCE_PROMPT, PROMPT_TEMPLATES

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

class RAG:
    def __init__(self):
        model_name = os.getenv("MODEL", "llama3.1:8b")
        self.llm = ChatOllama(
            model = model_name,
            num_ctx = 8192,
            num_predict = 4096
        )

        relevance_parser = PydanticOutputParser(pydantic_object = RelevanceCheck)
        relevance_prompt = PromptTemplate(
            template = RELEVANCE_PROMPT,
            input_variables = ["query", "context"],
            partial_variables = {"format_instructions" : relevance_parser.get_format_instructions()},
        )
        self.relevance_chain = relevance_prompt | self.llm | relevance_parser
        self.generation_chain = {
            lang : PromptTemplate.from_template(template) | self.llm | StrOutputParser()
            for lang, template in PROMPT_TEMPLATES.items()
        }

    
    def _format_context(self, docs : List[Document]) -> str:
        context_parts = []
        for doc in docs:
            page = doc.metadata.get('page', 'N/A')
            context_part = f"Trang {page}:\n{doc.page_content}"
            context_parts.append(context_part)
        return "\n\n---\n\n".join(context_parts)

    async def ask(self, query : str, media_id : Optional[int] = None, threshold : int = 7) -> str:
        try:
            lang = detect(query)
        except LangDetectException as l:
            log.warning(f"Error in detecion : {l}")
            lang = "vi"
        log.info(f"FOUND LANGUAGE : {lang}")
        retrieved_docs = await retrieval_and_rerank(
            query = query,
            media_id = media_id,
            k = 40,
            top_k = 20
        )
        if not retrieved_docs:
            return log.info("Không tìm thấy thông tin liên quan trong tài liệu.")
        formatted_context = self._format_context(retrieved_docs)
        try:
            relevancy = await self.relevance_chain.ainvoke({
                "query" : query,
                "context" : formatted_context
            })
            if relevancy.relevance_score < threshold:
                return log.warning("Tài liệu được tìm thấy không đủ liên quan để trả lời câu hỏi này.")
        except Exception as e:
            return log.error(f"Đã xảy ra lỗi trong quá trình kiểm tra mức độ liên quan: {e}")
        
        final_answer = await self.generation_chain.get(lang, self.generation_chain['vi']).ainvoke({
            "context": formatted_context,
            "question": query,
            "lang" : detect(query)
        })

        return final_answer
        