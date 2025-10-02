import os
import logging
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langdetect import detect, LangDetectException
from .retrieval import retrieval_and_rerank
from .definitions import RelevanceCheck, RELEVANCE_PROMPT, PROMPT_TEMPLATES, CONDENSE_QUESTION_PROMPT

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
        condense_prompt = PromptTemplate.from_template(
            CONDENSE_QUESTION_PROMPT
        )
        self.condense_chain = condense_prompt | self.llm | StrOutputParser()
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

    async def ask(self, query : str, media_id : Optional[int] = None, chat_history : Optional[List[Tuple[str, str]]] = None, threshold : int = 7) -> str:
        chat_history = chat_history or []
        if chat_history:
            formatted_history = "\n".join(
                [f"Người dùng: {q}\nTrợ lý: {a}" for q, a in chat_history]
            )
            standalone_question = await self.condense_chain.ainvoke({
                "chat_history": formatted_history,
                "question": query
            })
            log.info(f"Rewritten Question: {standalone_question}")
        else:
            standalone_question = query
            log.info(f"Standalone Question: {standalone_question}")
        try:
            lang = detect(standalone_question)
        except LangDetectException as l:
            log.warning(f"Error in detecion : {l}")
            lang = "vi"
        log.info(f"FOUND LANGUAGE : {lang}")
        retrieved_docs = await retrieval_and_rerank(
            query = standalone_question,
            media_id = media_id,
            k = 40,
            top_k = 20
        )
        if not retrieved_docs:
            no_info_answer = "Không tìm thấy thông tin liên quan trong tài liệu."
            log.info("Không tìm thấy thông tin liên quan trong tài liệu.")
            return {
                "answer" : no_info_answer,
                "history" : chat_history + [(query, no_info_answer)]
            }
        formatted_context = self._format_context(retrieved_docs)
        try:
            relevancy = await self.relevance_chain.ainvoke({
                "query" : query,
                "context" : formatted_context
            })
            if relevancy.relevance_score < threshold:
                log.warning("Tài liệu được tìm thấy không đủ liên quan để trả lời câu hỏi này.")
                not_relevant_answer = "Tài liệu được tìm thấy không đủ liên quan để trả lời câu hỏi này."
                return {
                    "answer": not_relevant_answer,
                    "history": chat_history + [(query, not_relevant_answer)]
                }
        except Exception as e:
            log.error(f"Đã xảy ra lỗi trong quá trình kiểm tra mức độ liên quan: {e}")
            error_answer = f"Đã xảy ra lỗi trong quá trình kiểm tra mức độ liên quan: {e}"
            return {
                "answer": error_answer,
                "history": chat_history + [(query, error_answer)]
            }
        
        final_answer = await self.generation_chain.get(lang, self.generation_chain['vi']).ainvoke({
            "context": formatted_context,
            "question": standalone_question,
            "lang" : detect(query)
        })
        chat_history.append((query, final_answer))
        return {
            "answer" : final_answer,
            "history" : chat_history
        }
        