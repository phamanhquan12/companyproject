# run_tests.py
import asyncio
import logging
import argparse
from src.workers.tasks import process_document_task, delete_document_task
from src.rag.pipeline import RAG

# --- CONFIGURE YOUR TESTS HERE ---
TEST_FILE_NAME = "luong" 
TEST_MEDIA_ID = 100 # Use a unique ID for each file
TEST_QUERY = "Nêu nội dụng đầy đủ điều 6, và điều 7"
# --- END OF CONFIGURATION ---

async def test_query():
    """Tests the full RAG pipeline from query to answer."""
    print("--- Testing RAG Query Pipeline ---")
    pipeline = RAG()
    
    # Test with a media_id filter
    answer = await pipeline.ask(query=TEST_QUERY, media_id=TEST_MEDIA_ID)
    
    print("\n--- QUERY ---")
    print(TEST_QUERY)
    print("\n--- FINAL ANSWER ---")
    print(answer)

def test_ingest():
    """Tests the ingestion by sending a task to the (future) Celery worker."""
    print("--- Testing Document Ingestion ---")
    print("Sending task to the message queue...")
    # In a real app, your API would call .delay(). For this test,
    # we can call the function directly to see the output.
    from src.workers.processing import process_document
    process_document(file_name=TEST_FILE_NAME, media_id=TEST_MEDIA_ID)
    print("--- Ingestion test finished. Check worker logs and database. ---")

def test_delete():
    """Tests the deletion by sending a task to the (future) Celery worker."""
    print(f"--- Testing Deletion for Media ID: {TEST_MEDIA_ID} ---")
    from src.workers.delete_documents import delete_documents
    delete_documents(media_id=TEST_MEDIA_ID)
    print("--- Deletion test finished. Check worker logs and database. ---")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Test script for the RAG application.")
    parser.add_argument("test_type", choices=['ingest', 'query', 'delete'], help="The type of test to run.")
    args = parser.parse_args()

    if args.test_type == 'ingest':
        test_ingest()
    elif args.test_type == 'delete':
        test_delete()
    elif args.test_type == 'query':
        asyncio.run(test_query())