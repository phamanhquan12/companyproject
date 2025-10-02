import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# We must load the environment variables before importing our modules
from src.workers.processing import process_document
from src.workers.delete_documents import delete_documents
from src.rag.pipeline import RAG

# --- Test Parameters ---
FILE_TO_INGEST = "luong"
# Using a different media_id to avoid conflicts with other tests
MEDIA_ID = 101
QUESTION = "Liệt kê cho tôi các bậc nhân viên và mức lương tương ứng"
# The path to the data directory, relative to the script location
DATA_DIRECTORY = "data"


def print_header(title):
    """Helper function to print a formatted header."""
    print("\n" + "="*60)
    print(f"--- {title} ---")
    print("="*60)


async def main():
    """Main function to run the CLI test workflow."""
    print("🚀 Starting RAG System Core Logic (CLI) Test 🚀")

    # --- 1. Ingest and Process Document ---
    print_header("1. Testing Document Ingestion and Processing")
    try:
        file_path = os.path.join(DATA_DIRECTORY, FILE_TO_INGEST)
        print(f"-> Processing file: {FILE_TO_INGEST}.pdf for media_id: {MEDIA_ID}")
        # This function handles loading, chunking, embedding, and saving
        process_document(file_name=FILE_TO_INGEST, media_id=MEDIA_ID)
        print("-> ✅ Ingestion and processing completed successfully.")
    except Exception as e:
        print(f"-> ❌ FAILED during document processing: {e}")
        # If ingestion fails, we can't proceed with the rest of the test
        return

    # --- 2. Test RAG Pipeline ---
    print_header("2. Testing RAG Chat Pipeline")
    try:
        print("-> Initializing RAG pipeline...")
        rag_pipeline = RAG()
        print(f"-> Asking question: '{QUESTION}'")
        result = await rag_pipeline.ask(
            query=QUESTION,
            media_id=None,
        )
        print(f"-> ✅ Answer received: {result.get('answer')}")
    except Exception as e:
        print(f"-> ❌ FAILED during RAG pipeline execution: {e}")

    # --- 3. Clean Up ---
    print_header("3. Cleaning Up Test Data")
    try:
        print(f"-> Deleting all data for media_id: {MEDIA_ID}")
        # This function deletes the source document and all associated chunks
        delete_documents(media_id=MEDIA_ID)
        print("-> ✅ Cleanup completed successfully.")
    except Exception as e:
        print(f"-> ❌ FAILED during data cleanup: {e}")

    print("\n🏁 CLI Test run finished. 🏁")


if __name__ == "__main__":
    # The RAG pipeline's 'ask' method is async, so we run the main function
    # within an asyncio event loop.
    asyncio.run(main())
