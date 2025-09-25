from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Tuple, Optional
import sys
import os

from src.rag.pipeline import RAG
from src.workers.celery_app import celery_app
from src.workers.tasks import  process_document_task, delete_document_task


app = FastAPI(
    title = "TestingRAG",
    description= "Test API",
    version = "1.0.0"
)

rag_pipeline = RAG()

class IngestRequest(BaseModel):
    file_name: str
    media_id: int

class IngestResponse(BaseModel):
    message: str
    task_id: str

class DeleteRequest(BaseModel):
    media_id: int

class DeleteResponse(BaseModel):
    message: str
    task_id: str

class ChatRequest(BaseModel):
    question: str
    media_id: Optional[int] = None
    history: List[Tuple[str, str]] = []

class ChatResponse(BaseModel):
    answer: str
    history: List[Tuple[str, str]]


@app.post("/ingest", response_model = IngestResponse, summary="Import documents")
def ingest_document(request : IngestRequest):
    task = process_document_task.delay(
        file_name = request.file_name,
        media_id = request.media_id
    )
    return {
        "message" : "Document ingestion started.",
        "task_id" : task.id
    }

@app.post("/chat", response_model = ChatResponse, summary = "Chats")
async def chat_rag(request : ChatRequest):
    try:
        result = await rag_pipeline.ask(
            query = request.question,
            media_id = request.media_id,
            chat_history = request.history
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/delete", response_model=DeleteResponse, summary="Del documents")
def delete_document(request: DeleteRequest):
    task = delete_document_task.delay(media_id=request.media_id)
    return {"message": "Document deletion started.", "task_id": task.id}