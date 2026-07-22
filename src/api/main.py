import os
import sys
import logging
from typing import List, Dict, Any, Optional
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from pydantic import BaseModel

# Ensure src is in Python path so we can import our local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from ingestion.pipeline import RAGPipeline
from retrieval.vector import SimpleVectorRetriever
from retrieval.lexical import BM25Retriever
from retrieval.hybrid import HybridRetriever
from embeddings.factory import get_embedding_model
from ingestion.ingestion import ingest_documents

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Clinical RAG API",
    description="Enterprise-ready API for the Oncology RAG Pipeline.",
    version="1.0.0"
)

# -----------------
# Pydantic Schemas
# -----------------
class ChatMessage(BaseModel):
    role: str
    content: str
    contexts: Optional[List[Dict[str, Any]]] = None

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = []
    generator_model: Optional[str] = "mock"
    retriever_type: Optional[str] = "hybrid"

class ChatResponse(BaseModel):
    response_text: str
    rewritten_query: str
    retrieved_contexts: List[Dict[str, Any]]
    latency_ms: float

# Global pipeline cache (MVP equivalent of st.cache_resource)
PIPELINES = {}

def get_pipeline(retriever_type: str, generator_model: str) -> RAGPipeline:
    cache_key = f"{retriever_type}_{generator_model}"
    if cache_key in PIPELINES:
        return PIPELINES[cache_key]
        
    logger.info(f"Initializing new pipeline: {cache_key}")
    kb_path = os.path.join(os.path.dirname(src_dir), "datasets", "knowledge_base")
    chunks = ingest_documents(kb_path, chunk_size=800, chunk_overlap=150, chunking_enabled=True)
    
    embed = get_embedding_model("mock")
    
    if retriever_type == "vector":
        retriever = SimpleVectorRetriever(embedding_model=embed)
    elif retriever_type == "lexical":
        retriever = BM25Retriever()
    else:
        v_ret = SimpleVectorRetriever(embedding_model=embed)
        l_ret = BM25Retriever()
        retriever = HybridRetriever(vector_retriever=v_ret, lexical_retriever=l_ret)
        
    retriever.index_documents(chunks)
    pipeline = RAGPipeline(retriever=retriever, generator_model=generator_model)
    PIPELINES[cache_key] = pipeline
    return pipeline

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        pipeline = get_pipeline(request.retriever_type, request.generator_model)
        
        # Convert Pydantic models back to dictionaries for the pipeline
        history_dicts = [msg.model_dump() for msg in request.chat_history]
        
        result = pipeline.generate(request.query, k=3, chat_history=history_dicts)
        
        return ChatResponse(
            response_text=result["response_text"],
            rewritten_query=result.get("rewritten_query", request.query),
            retrieved_contexts=result.get("retrieved_contexts", []),
            latency_ms=result.get("latency", {}).get("total_ms", 0)
        )
    except Exception as e:
        logger.error(f"API Generation Error: {e}")
        # Translate the backend error (like the xAI billing error) into a clean HTTP 400/500
        error_str = str(e)
        if "Check Billing Credits" in error_str:
            raise HTTPException(status_code=402, detail="Payment Required: Your API provider account has insufficient credits.")
        elif "Model not found" in error_str or "invalid-argument" in error_str:
            raise HTTPException(status_code=400, detail=f"Bad Request: The LLM model configuration is invalid. {error_str}")
        else:
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {error_str}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "pipelines_loaded": len(PIPELINES)}
