import os
import logging
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
from load import load_from_document
from uuid import uuid4
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import SentenceTransformerEmbeddings
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logging.info('Imported')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, 'vecdb', 'database', 'chromadb')
logging.info(DB_DIR)
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-large'
EMBEDDING_FN = SentenceTransformerEmbeddings(
    model_name = EMBEDDING_MODEL_NAME,
    model_kwargs = {'device' : 'cpu'},
    encode_kwargs = {'normalize_embeddings' : True}
)
class HierarichicalLC:
    def __init__(self, collection_name, persist_directory = DB_DIR, embedding_fn = None):
        os.makedirs(persist_directory, exist_ok=True)
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
            separators = ['\n\n','\n', '.', ' ', ''],
            keep_separator = True
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 400,
            chunk_overlap = 40,
            separators = ['\n\n','\n', '.', ' ', ''],
            keep_separator = True
        )

        self.semantic_splitter = SemanticChunker(
            EMBEDDING_FN,
            breakpoint_threshold_type='percentile',
            breakpoint_threshold_amount=85
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
    combined = []
    res_parent = chunker.collection.query(
        query_texts = [query],
        n_results = 15,
        where = {'chunk_level' : 'parent'}
    )
    res_child = chunker.collection.query(
        query_texts = [query],
        n_results = 20,
        where = {'chunk_level' : 'child'}
    )

    if res_parent and res_parent['ids'][0]:
        for i in range(len(res_parent['ids'][0])):
            combined.append(
                Document(
                    page_content = res_parent['documents'][0][i],
                    metadata = res_parent['metadatas'][0][i]
                )
            )
    if res_child and res_child['ids'][0]:
        for i in range(len(res_child['ids'][0])):
            combined.append(
                Document(
                    page_content = res_child['documents'][0][i],
                    metadata = res_child['metadatas'][0][i]
                )
            )
    logging.info(f"Retrieved {len(combined)} most relevant chunks.")
    logging.info(f"Chunks : {combined}")
    return combined



if __name__ == '__main__':
    vn_docs, jap_docs = load_from_document('test_doc')
    chunker = HierarichicalLC('vn_legal',DB_DIR)
    chunker.chunk_and_store(vn_docs)
    logging.info('Testing completed')
