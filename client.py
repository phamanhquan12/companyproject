import requests
import json
import argparse

# The base URL of your API running in Docker
BASE_URL = "http://localhost:8000"

def handle_request(method, url, **kwargs):
    """A helper function to handle requests and print responses."""
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        print("✅ Request Successful:")
        # Try to parse and pretty-print JSON, otherwise print raw text
        try:
            print(json.dumps(response.json(), indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request Failed: {e}")
        # If the response exists, print its content for debugging
        if e.response is not None:
            print(f"Server Response: {e.response.text}")

def ingest_document(file_name: str, media_id: int):
    """
    Sends a request to the /ingest endpoint to start document processing.
    """
    print(f"--- Sending Ingest Request for file: {file_name}, media_id: {media_id} ---")
    url = f"{BASE_URL}/ingest"
    payload = {"file_name": file_name, "media_id": media_id}
    headers = {"Content-Type": "application/json"}
    handle_request("post", url, headers=headers, json=payload)

def ask_question(question: str, media_id: int = None):
    """
    Sends a question to the /chat endpoint.
    """
    print(f"--- Sending Chat Request ---")
    url = f"{BASE_URL}/chat"
    payload = {
        "question": question,
        "media_id": media_id,
        "history": []  # Example with no history, you can modify this
    }
    headers = {"Content-Type": "application/json"}
    handle_request("post", url, headers=headers, json=payload)

def delete_document(media_id: int):
    """
    Sends a request to the /delete endpoint to remove a document.
    """
    print(f"--- Sending Delete Request for media_id: {media_id} ---")
    url = f"{BASE_URL}/delete"
    payload = {"media_id": media_id}
    headers = {"Content-Type": "application/json"}
    # Note: FastAPI's @app.delete can accept a body, so we send it as JSON
    handle_request("delete", url, headers=headers, json=payload)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Command-line client for the RAG API.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Ingest Command ---
    parser_ingest = subparsers.add_parser("ingest", help="Ingest a new document.")
    parser_ingest.add_argument("file_name", type=str, help="The name of the file to ingest (without .pdf extension).")
    parser_ingest.add_argument("media_id", type=int, help="A unique integer ID for the media file.")

    # --- Chat Command ---
    parser_chat = subparsers.add_parser("chat", help="Ask a question.")
    parser_chat.add_argument("question", type=str, help="The question to ask the RAG pipeline.")
    parser_chat.add_argument("--media_id", type=int, default=None, help="Optional: An integer media_id to filter the search.")

    # --- Delete Command ---
    parser_delete = subparsers.add_parser("delete", help="Delete a document.")
    parser_delete.add_argument("media_id", type=int, help="The integer media_id of the document to delete.")

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_document(args.file_name, args.media_id)
    elif args.command == "chat":
        ask_question(args.question, args.media_id)
    elif args.command == "delete":
        delete_document(args.media_id)