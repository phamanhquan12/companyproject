import os
import logging
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from load import load_from_document
from uuid import uuid4
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
PDF_PATH = 'true_test.pdf'
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'

class HierarichicalLC:
    def __init__(self, collection_name, persist_directory = './chromadb', embedding_fn = None):
        self.collection_name = collection_name
        self.vector_store = chromadb.PersistentClient(path = persist_directory)
        if embedding_fn is None:
            try: 
                self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name = EMBEDDING_MODEL_NAME,
                    device = 'cpu',
                    normalize_embeddings = True
                )
            except Exception as e:
                logging.error(f'Error : {e}')
        else:
            self.embedding_fn = embedding_fn
        self.collection = self.vector_store.get_or_create_collection(
            name = self.collection_name,
            embedding_function = self.embedding_fn
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1200,
            chunk_overlap = 200,
            separators = ['\n', '.', ' ', ''],
            keep_separator = True
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50,
            separators = ['\n', '.', ' ', ''],
            keep_separator = True
        )
    def chunk_and_store(self, documents: List[Document]):
        if self.collection.count() > 0:
            print(f"{self.collection_name} already has data, skipping the process.")
            return
        logging.info(f'Processing...')
        all_chunks = self._chunk(documents)
        self._add_to_vector_store(all_chunks)
        logging.info(f'Processing complete')
        return all_chunks
    def _chunk(self, documents: List[Document]) -> List[Document]:
        parent_chunks, child_chunks = [], []

        for doc in documents:
            p_docs = self.parent_splitter.split_documents([doc])
            for p_doc in p_docs:
                parent_id = str(uuid4())
                p_doc.metadata['chunk_level'] = 'parent'
                p_doc.metadata['id'] = parent_id
                parent_chunks.append(p_doc)

                child_docs = self.child_splitter.split_documents([p_doc])
                for c_doc in child_docs:
                    child_id = str(uuid4())
                    c_doc.metadata['chunk_level'] = 'child'
                    c_doc.metadata['parent_id'] = parent_id
                    c_doc.metadata['id'] = child_id
                    child_chunks.append(c_doc)
        return parent_chunks + child_chunks
    def _add_to_vector_store(self, chunks: List[Document]):
        """
        Adds a list of LangChain Document objects to the ChromaDB collection.
        """
        if not chunks:
            print("No chunks to add.")
            return

        # Extract contents, metadatas, and ids for ChromaDB
        contents = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.metadata['id'] for chunk in chunks]

        # Add to the collection
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
        logging.info(f'Successfully added {chunks}')

def combine_retrieved_docs(query, chunker):
    res_parent = chunker.collection.query(
        query_text = [query],
        n_results = 5,
        where = {'chunk_level' : 'parent'}
    )['documents'][0]
    res_child = chunker.collection.query(
        query_text = [query],
        n_results = 10,
        where = {'chunk_level' : 'child'}
    )['documents'][0]
    return res_parent + res_child


if __name__ == '__main__':
    vn_docs, jap_docs = load_from_document(PDF_PATH)
    chunker = HierarichicalLC('main_policies','database/chromadb')
    chunker.chunk_and_store(vn_docs)
    logging.info('Testing completed')
