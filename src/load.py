import os
import pickle
import re
from typing import List
from langchain.docstore.document import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers.images import TesseractBlobParser
import logging
import pytesseract
# pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/opt/tesseract/bin/tesseract'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info('import completed')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)
def preprocess_text_unified(text: str) -> str:
    # 1. Remove specific HTML tags like <br>.
    text = text.replace('*', '')
    text = text.replace('_', '')
    text = re.sub(r'<br\s*/?>', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def load_from_document(full_pdf_path: str) -> List[Document]:
    """
    Loads a PDF document, processes its pages, and returns a single list of documents.
    Caches the processed result for faster subsequent loads.
    
    Args:
        full_pdf_path: The full, absolute path to the PDF file.
    """
    F_PATH = os.path.join(DATA_DIR, f'{full_pdf_path}.pdf')
    if not os.path.exists(F_PATH):
        logging.error(f'File does not exist: {full_pdf_path}')
        return []

    file_basename = os.path.basename(full_pdf_path)
    cache_path = os.path.join(PROCESSED_DIR, f'{file_basename}.pkl')

    if os.path.exists(cache_path):
        logging.info(f'Loading document from cache: {cache_path}')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
            
    try:
        loader = PyMuPDF4LLMLoader(
            F_PATH,
            # extract_images = True,
            # images_parser = TesseractBlobParser(langs=['vi','eng','ja']),
            mode = 'page',
            table_strategy = "lines_strict",
        )
        logging.info(f'Initialized loader for {full_pdf_path}')
    except Exception as e:
        logging.error(f"Failed to initialize loader: {e}")
        return []

    docs = loader.lazy_load()
    all_docs = []

    for doc in docs:
        page_num = doc.metadata.get('page', 'N/A')
        logging.info(f'Processing page : {page_num}')
        cleaned_content = preprocess_text_unified(doc.page_content)
        
        # Only include pages that have a meaningful amount of text
        if len(cleaned_content) > 20:  
            all_docs.append(
                Document(
                    page_content=cleaned_content,
                    metadata=doc.metadata  
                )
            )
            
    with open(cache_path, 'wb') as f:
        pickle.dump(all_docs, f)
        
    logging.info(f'Saved {len(all_docs)} processed pages to cache: {cache_path}')
    return all_docs

#test
if __name__ == '__main__':
    print('Testing load.py...')
    load_from_document("general_rule")
    print('load.py test complete.')
