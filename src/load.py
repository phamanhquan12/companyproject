import os
import pickle
import re
from typing import List
from langchain.docstore.document import Document
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_community.document_loaders.parsers.images import TesseractBlobParser
from langdetect import detect, LangDetectException
from uuid import uuid4
import logging
from underthesea import word_tokenize
import pytesseract
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/opt/tesseract/bin/tesseract'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info('import completed')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
PROCESSED_DIR = os.path.join(BASE_DIR, '..', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)
VIET_PATTERN = re.compile(r'[\w\sÀ-ỹ\u0300-\u036F.,;:!?()“”‘’\-–—]+', re.UNICODE)
JAP_PATTERN = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+") #Hiragana, Katakana, Kanji
def detect_language_with_fallback(text, fallback='vi'):
    if len(text.strip()) < 50:  # Too short → unreliable detection
        return fallback
    try:
        lang = detect(text)
        if lang in ['vi', 'ja']:
            return lang
        else:
            return fallback
    except LangDetectException:
        return fallback
def preprocess_vie(text):
    text = JAP_PATTERN.sub('', text)
    vie_chars = "ạảãàáâậầấẩẫăắằặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    pattern = re.compile(f"([{vie_chars}]) ([{vie_chars}])")
    while True:
        new_text = pattern.sub(r"\1\2", text)  
        if new_text == text:
            break
        text = new_text
    try:
        tokens = word_tokenize(text)
        text = " ".join(tokens)
    except:
        pass
    text = re.sub(r"\s+", ' ', text).strip()
    return text
def preprocess_jpn(text):
    VIET_LATIN_PATTERN = re.compile(r'[a-zA-ZÀ-ỹ\u0300-\u036F]+')
    text = VIET_LATIN_PATTERN.sub('', text)
    JP_KEEP_PATTERN = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF。、・「」『』（）！？…―\s]')
    text = ''.join(char for char in text if JP_KEEP_PATTERN.match(char))
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def load_from_document(path : str) -> List[Document]:
    pdf_path = os.path.join(DATA_DIR, f'{path}.pdf')
    if not os.path.exists(pdf_path):
        logging.error('File does not exist.')
        return [], []
    cache_path = os.path.join(PROCESSED_DIR, f'{path}.pkl')
    if os.path.exists(cache_path):
        logging.info(f'Loading document from {cache_path}')
        with open(f'{cache_path}', 'rb') as f:
            return pickle.load(f)
    try:    
        loader = PyMuPDF4LLMLoader(
            pdf_path, 
            mode='page', 
            extract_images=True,
            images_parser=TesseractBlobParser(langs=['vie', 'jpn', 'eng']),
            table_strategy='lines'
        )
        logging.info('Initialized loader')
    except Exception as e:
        logging.error(e)
    logging.info(f'Processing PDF')
    docs = loader.lazy_load()
    vietnamese_docs, japanese_docs = [], []

    for doc in docs:
        logging.info(f'Processing document: {doc.metadata.get("source", "unknown source")}, page {doc.metadata.get("page", "N/A")}')
        content = doc.page_content
        lang = detect_language_with_fallback(content, fallback='vi')

        if lang == 'vi':
            cleaned = preprocess_vie(content)
            if len(cleaned) > 10:  
                vietnamese_docs.append(
                    Document(
                        page_content=cleaned,
                        metadata={**doc.metadata, 'language': 'vi'}
                    )
                )
        elif lang == 'ja':
            cleaned = preprocess_jpn(content)
            if len(cleaned) > 10:
                japanese_docs.append(
                    Document(
                        page_content=cleaned,
                        metadata={**doc.metadata, 'language': 'ja'}
                    )
                )
    with open(f'{cache_path}', 'wb') as f:
        pickle.dump([vietnamese_docs, japanese_docs], f)
    logging.info(f'Saved documents to {cache_path}')
    return vietnamese_docs, japanese_docs


#test
if __name__ == '__main__':
    print('Testing load.py...')
    # Now we call it with just the filename
    load_from_document("true_test2")
    print('load.py test complete.')
