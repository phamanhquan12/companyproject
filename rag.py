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
from hchunk import HierarichicalLC
from load import load_from_document, PDF_PATH
# This import now assumes your reranker.py is correct
from reranker import rerank_documents_vn, VN_MODEL

# --- Configuration ---
MODEL_NAME = 'llama3.1:8b'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Relevance Check Router ---
class RelevanceCheck(BaseModel):
    is_relevant: bool = Field(description="A boolean value indicating if the context can answer the query.")

def is_context_relevant(query: str, context: str, llm: ChatOllama) -> bool:
    # ... (This function remains the same as before)
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
        child_results = self.chunker.collection.query(
            query_texts=[query], n_results=10, where={"chunk_level": "child"}
        )
        parent_ids = list(set(meta.get('parent_id') for meta in child_results['metadatas'][0] if meta.get('parent_id')))
        if not parent_ids:
            logging.warning("No parent documents found.")
            return []
        
        # parent_docs_data is a dictionary of raw lists
        parent_docs_data = self.chunker.collection.get(ids=parent_ids)

        # --- KEY FIX IS HERE ---
        # Reconstruct Document objects from the raw ChromaDB output
        initial_docs = [
            Document(page_content=text, metadata=meta) 
            for text, meta in zip(parent_docs_data['documents'], parent_docs_data['metadatas'])
        ]
        logging.info(initial_docs)
        # -----------------------

        logging.info(f"Reranking {len(initial_docs)} documents.")
        # Now, rerank_documents_vn receives the correct List[Document] type
        reranked = rerank_documents_vn(query, initial_docs, self.reranker, top_k=3)
        return reranked

    def ask_stream(self, query: str):
        # ... (This function remains the same as before)
        reranked_docs = self._retrieve_and_rerank(query)
        if not reranked_docs:
            yield "I could not find any relevant information in the documents."
            return
        context = "\n---\n".join([doc.page_content for doc in reranked_docs])
        if not is_context_relevant(query, context, self.llm):
            logging.warning("Context deemed not relevant by the router.")
            yield "The retrieved documents do not contain a relevant answer to your question."
            return
        logging.info("Context is relevant. Proceeding to generation.")
        generation_template = """You are a helpful assistant. Answer the user's question based ONLY on the following context.
        If the answer is not in the context, say you don't know. Be concise.
        Context: {context}
        Question: {question}
        """
        prompt = PromptTemplate.from_template(generation_template)
        chain = prompt | self.llm
        for chunk in chain.stream({"context": context, "question": query}):
            yield chunk.content

# --- Main Execution Block ---
if __name__ == "__main__":
    vietnamese_docs, _ = load_from_document(PDF_PATH)
    chunker = HierarichicalLC('main_policies', 'database/chromadb')
    chunker.chunk_and_store(vietnamese_docs)
    llm = ChatOllama(model=MODEL_NAME, temperature=0)
    reranker = FlagReranker(VN_MODEL, use_fp16=True)
    pipeline = RAG_Pipeline(chunker, llm, reranker)
    user_query = "Giờ làm việc cơ bản là gì?"
    print(f"\n--- Querying with: '{user_query}' ---")
    full_response = ""
    for chunk in pipeline.ask_stream(user_query):
        print(chunk, end="", flush=True)
        full_response += chunk
    print("\n--- End of Response ---")
