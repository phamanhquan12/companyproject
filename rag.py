import os
import pickle
import faiss
import base64
import re
from pathlib import Path
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.retrievers import BM25Retriever
from langchain.docstore import InMemoryDocstore
from uuid import uuid4
print('import completed')

PDF_PATH = 'test_doc.pdf'
FAISS_INDEX_PATH =  'faiss_index'
IMAGE_DIR = 'extracted_images'
EMBEDDING_MODEL_NAME = 'mxbai-embed-large:335m'
MODEL_NAME = 'gemma3:12b'

image_dir_path = Path(IMAGE_DIR)
image_dir_path.mkdir(exist_ok=True)
try:
    loader = PyMuPDF4LLMLoader(PDF_PATH)
    all_docs = loader.load()
    for doc in all_docs:
        if "![image" in doc.page_content:
            doc.metadata['type'] = 'text_and_images'
        else:
            doc.metadata['type'] = 'text'
except FileNotFoundError:
    print('PDF File not found.')

print('Load completed')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(all_docs)
print('Chunking completed')
embedding = OllamaEmbeddings(model = EMBEDDING_MODEL_NAME)
len_sample = len(embedding.embed_query('test query'))
print('Embedding completed')
index = faiss.IndexHNSWFlat(len_sample, 32)
index.hnsw.efConstruction = 128
docs_id = [str(uuid4()) for _ in range(len(chunks))]
faiss_db = FAISS(embedding.embed_query, index, chunks, docs_id)
faiss_retriever = faiss_db.as_retriever(search_kwargs={"k": 5})
print('FAISS completed')
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5
print('BM25 completed')
ensemble_retriever = EnsembleRetriever(
    retrievers=[faiss_retriever, bm25_retriever],
    weights=[0.6, 0.4]
)
print('Ensemble completed')
def format_multimodal(retrieved_documents : List[Document], question: str):
    multimodal_content = [{'type': 'text', 'text': f'Question: {question}\n\nContext:'}]

    text_content = []
    image_content = []
    for doc in retrieved_documents:
        source_path = doc.metadata.get('source')
        page_num = doc.metadata.get('page')
        if doc.metadata.get('images'):
            for image_data in doc.metadata['images']:
                image_content.append({'type': 'image_url', 'image_url': {'url' : f'data:image/jpeg;base64,{image_data}'}})
    text_without_images = re.sub(r'!\[(.*?)\]\((.*?)\)', '', doc.page_content)
    text_content.append(f'--- (SOURCE: {source_path}, PAGE: {page_num}) ---\n\n{text_without_images}\n\n')
    if text_content:
        multimodal_content.append({'type': 'text', 'text': '\n\n'.join(text_content)})
    if image_content:
        multimodal_content.extend(image_content)

    return multimodal_content

# question = 'Khoản 1 là gì?'

# retrieved_documents = ensemble_retriever.invoke(question)
# print(retrieved_documents)
# print('Retrieval completed')
# multimodal_content = format_multimodal(retrieved_documents, question)
# print(multimodal_content)
# print('Multimodal content completed')

def run_rag(question: str, retriever : EnsembleRetriever, llm : ChatOllama):
    retrieved_documents = retriever.invoke(question)
    multimodal_message_content = format_multimodal(retrieved_documents, question)
    # router_prompt = ChatPromptTemplate.from_messages([
    #     ("system",'''Bạn là chuyên gia trong việc phân loại tính tương quan giữa các ngữ cảnh cơ bản.
    #                 Cho các ngữ cảnh, và câu hỏi, quyết định xem liệu các ngữ cảnh có liên quan đến câu hỏi không.
    #                 Trả lời DUY NHẤT với 'yes' hoặc 'no'.
    #                         '''),
    #     "human", "Ngữ cảnh: {retrieved_documents}\nCâu hỏi: {question}\n"
    # ])

    prompt = ChatPromptTemplate.from_messages([
       ("system", 'Bạn là một trợ lý chuyên nghiệp. Hãy sử dụng thông tin từ các tài liệu và hình ảnh được cung cấp để trả lời câu hỏi. Trả lời câu hỏi bằng Tiếng Việt. Trả lời rõ ràng và chi tiết liên quan đến tài liệu. Nếu không có thông tin, hãy nói bạn không biết.'),
       HumanMessagePromptTemplate.from_template("{question}")
    ])

    rag_chain = prompt | llm | StrOutputParser()
    answer = rag_chain.invoke({'question': multimodal_message_content})
    unique_sources = []
    for doc in retrieved_documents:
        source = f'Source: {doc.metadata.get("source")}, Page: {doc.metadata.get("page")}'
        if source not in unique_sources:
            unique_sources.append(source)

    return answer, unique_sources

if __name__ == '__main__':
    if not os.path.exists(PDF_PATH):
        print('PDF File not found.')
    else:
        print("Starting RAG...")
        try:
            llm = ChatOllama(model = MODEL_NAME, temperature=0, max_tokens=1024)
        except Exception as e:
            print(f'Error: {e}')
        print('System is ready.')
        while True:
            question = input('\nYou: ')
            if question.lower() in ['exit', 'quit']:
                print('Goodbye.')
                break
            try:
                answer, unique_sources = run_rag(question, ensemble_retriever, llm)
                print(f'AI: {answer}')
                print(f'Sources:')
                for source in unique_sources:
                    print(f"- {source}")
            except Exception as e:
                print(f'Error: {e}')