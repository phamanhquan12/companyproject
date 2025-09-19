# test_query.py
import asyncio
import logging
from src.rag.retrieval import retrieval_and_rerank

# --- CONFIGURE YOUR TESTS HERE ---
TEST_QUERY = "Tôi muốn đi công tác thì sao"
# Use the media_id from the document you ingested with test_ingestion.py
TEST_MEDIA_ID = 999 # Example ID, change if needed
TOP_K = 5
# --- END OF CONFIGURATION ---

async def run_test_query(query: str, media_id: int | None = None):
    """Helper function to run a single test query and print results."""
    header = f"--- Testing Filtered Search (media_id={media_id}) ---" if media_id else "--- Testing General Search ---"
    print(header)

    try:
        # Pass the media_id to the retrieval function
        retrieved_docs = await retrieval_and_rerank(
            query=query, 
            media_id=None,
            top_k=TOP_K
        )
        
        print(f"\nQuery: '{query}'")
        print(f"\n--- Top {TOP_K} Retrieved & Reranked Documents ---")
        
        if not retrieved_docs:
            print("No documents found.")
        else:
            for i, doc in enumerate(retrieved_docs):
                print(f"\n[--- Document {i+1} ---]")
                print(f"Content: {doc.page_content[:400]}...")
                print(f"Metadata: {doc.metadata}")
        print("-" * len(header))
        
    except Exception as e:
        print(f"\n--- An error occurred during the test: {e} ---")

async def main():
    """Runs both test cases."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test Case 1: General search across all documents
    await run_test_query(query=TEST_QUERY)
    
    print("\n" + "="*50 + "\n") # Add a separator
    
    # Test Case 2: Filtered search within a specific document
    await run_test_query(query=TEST_QUERY, media_id=TEST_MEDIA_ID)

if __name__ == "__main__":
    asyncio.run(main())