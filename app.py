import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx

from pipeline.search import search_documents

# Load .env
load_dotenv(Path(__file__).resolve().parent / ".env")

app = FastAPI(
    title="NILA Backend API",
    description="Bridge API connecting WhatsApp n8n Workflow A, Vector Search RAG, and n8n Workflow B",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# n8n Workflow B webhook URL (configurable via .env)
N8N_WORKFLOW_B_WEBHOOK_URL = os.getenv("N8N_WORKFLOW_B_WEBHOOK_URL", "")


class UserMessageWebhook(BaseModel):
    user_id: str = Field(..., description="WhatsApp user phone number or unique identifier")
    status: str = Field(..., description="Markdown (.md) file content tracking user's status/history")
    need: str = Field(..., description="Current incoming user message / query")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata passed by n8n")

    model_config = {
        "extra": "allow"  # Automatically accepts any extra attributes from n8n
    }


class SearchRequest(BaseModel):
    query: str = Field(..., description="Text query to search semantically against Supabase vector database")
    match_count: int = Field(default=5, description="Number of top matching chunks to retrieve")


@app.get("/")
def read_root():
    return {
        "message": "NILA Backend API is running",
        "status": "healthy",
        "workflow_b_configured": bool(N8N_WORKFLOW_B_WEBHOOK_URL),
        "endpoints": {
            "workflow_a_entry": "POST /api/webhook/message",
            "vector_search_for_workflow_b": "POST /api/search"
        },
        "docs_url": "/docs"
    }


# ---------------------------------------------------------------------------
# Endpoint 1: Called by Workflow A (WhatsApp bot) to push user message & status
# ---------------------------------------------------------------------------
@app.post("/api/webhook/message")
@app.post("/webhook")
async def handle_user_interaction(payload: UserMessageWebhook):
    """
    Receives user_id, status (.md text), and need from n8n Workflow A.
    Forwards this data to n8n Workflow B webhook for processing.
    """
    timestamp = datetime.utcnow().isoformat()
    print(f"\n[{timestamp}] -> Received from n8n Workflow A:")
    print(f"User ID : {payload.user_id}")
    print(f"Need    : {payload.need}")
    print(f"Status  : {len(payload.status)} characters of Markdown content")

    forward_payload = {
        "user_id": payload.user_id,
        "status": payload.status,
        "need": payload.need,
        "metadata": payload.metadata or {},
        "forwarded_at": timestamp
    }

    # If Workflow B URL is configured in .env, forward directly to it
    if N8N_WORKFLOW_B_WEBHOOK_URL:
        try:
            print(f"Forwarding to Workflow B: {N8N_WORKFLOW_B_WEBHOOK_URL}...")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(N8N_WORKFLOW_B_WEBHOOK_URL, json=forward_payload)
                response.raise_for_status()

                try:
                    b_data = response.json()
                except Exception:
                    b_data = {"response_text": response.text}

                return {
                    "status": "forwarded_to_workflow_b",
                    "user_id": payload.user_id,
                    "workflow_b_response": b_data,
                    "timestamp": timestamp
                }
        except httpx.HTTPStatusError as http_err:
            print(f"Workflow B returned HTTP error: {http_err.response.status_code} - {http_err.response.text}")
            raise HTTPException(
                status_code=http_err.response.status_code,
                detail=f"Error from Workflow B: {http_err.response.text}"
            )
        except Exception as err:
            print(f"Failed to forward to Workflow B: {err}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not reach n8n Workflow B webhook: {str(err)}"
            )

    # If Workflow B URL is not yet configured in .env, acknowledge receipt
    return {
        "status": "received",
        "message": "Data received successfully. Set N8N_WORKFLOW_B_WEBHOOK_URL in .env to forward to Workflow B.",
        "payload": forward_payload
    }


# ---------------------------------------------------------------------------
# Endpoint 2: Called by Workflow B to perform Vector Search on Supabase
# ---------------------------------------------------------------------------
@app.post("/api/search")
def search_vector_db(payload: SearchRequest):
    """
    Vector similarity search endpoint for Workflow B.
    Takes a query (user need) and returns the top relevant chunks from Supabase.
    """
    try:
        results = search_documents(query=payload.query, match_count=payload.match_count)
        return {
            "query": payload.query,
            "match_count": len(results),
            "results": [
                {
                    "content": doc.get("content", ""),
                    "source": doc.get("metadata", {}).get("source", ""),
                    "page": doc.get("metadata", {}).get("page", ""),
                    "sheet": doc.get("metadata", {}).get("sheet", ""),
                    "similarity": round(doc.get("similarity", 0.0), 4)
                }
                for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
