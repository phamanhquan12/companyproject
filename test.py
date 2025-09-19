# test_ingestion.py
import logging
from src.workers.processing import process_document

def main():
    """
    A simple script to test the full document ingestion and storage pipeline.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print("--- Starting ingestion test ---")

    # --- CONFIGURE YOUR TEST HERE ---
    # 1. Make sure this file exists in your 'data/' directory (e.g., "true_test.pdf")
    test_file_name = "test_doc" 
    
    # 2. Use a sample ID as if it came from the admin database
    test_media_id = 101 
    # --- END OF CONFIGURATION ---

    try:
        process_document(
            file_name=test_file_name,
            media_id=test_media_id
        )
        print("\n--- Ingestion test completed! Check the database to verify. ---")
    except Exception as e:
        print(f"\n--- An error occurred during the test: {e} ---")

if __name__ == "__main__":
    main()