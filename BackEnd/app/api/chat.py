"""Chat API endpoints."""

import logging
from typing import Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..core.rag_chain import get_rag_chain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=1000, description="User's question")
    conversation_id: str = Field(default="", description="Optional conversation ID for tracking")


class Source(BaseModel):
    """Source information model."""
    url: str
    title: str
    score: float


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    sources: List[Source] = []
    context_used: bool = False
    conversation_id: str = ""


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    message: str
    vector_db_stats: Dict = {}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint that processes user questions about Occam's Advisory.
    
    This endpoint:
    1. Receives a user question
    2. Uses RAG to find relevant information from Occam's Advisory website
    3. Generates a response using Groq LLM
    4. Returns the answer with source citations
    """
    try:
        logger.info(f"Received chat request: {request.message[:100]}...")
        
        # Get RAG chain instance
        rag_chain = get_rag_chain()
        
        # Process the question
        result = rag_chain.answer_question(request.message)
        
        # Format sources
        sources = [
            Source(
                url=source["url"],
                title=source["title"],
                score=source["score"]
            )
            for source in result["sources"]
        ]
        
        response = ChatResponse(
            answer=result["answer"],
            sources=sources,
            context_used=result["context_used"],
            conversation_id=request.conversation_id
        )
        
        logger.info(f"Generated response with {len(sources)} sources")
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your question. Please try again."
        )


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint to verify RAG system status.
    """
    try:
        rag_chain = get_rag_chain()
        health_status = rag_chain.health_check()
        
        if health_status["status"] == "healthy":
            return HealthResponse(
                status="healthy",
                message="RAG system is working properly",
                vector_db_stats=health_status.get("vector_db_stats", {})
            )
        else:
            return HealthResponse(
                status="unhealthy",
                message=f"RAG system issue: {health_status.get('error', 'Unknown error')}",
                vector_db_stats={}
            )
            
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )


@router.get("/stats")
async def get_system_stats() -> Dict:
    """
    Get detailed system statistics.
    """
    try:
        rag_chain = get_rag_chain()
        stats = rag_chain.embedding_manager.get_collection_stats()
        
        return {
            "system_status": "operational",
            "vector_database": stats,
            "embedding_model": rag_chain.embedding_manager.model_name,
            "llm_model": rag_chain.model_name
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve system statistics"
        )


@router.post("/reinitialize")
async def reinitialize_system() -> Dict:
    """
    Reinitialize the RAG system (admin endpoint).
    """
    try:
        from ..core.rag_chain import initialize_rag_system
        
        logger.info("Reinitializing RAG system...")
        result = initialize_rag_system()
        
        return {
            "message": "System reinitialization completed",
            "status": result["status"]
        }
        
    except Exception as e:
        logger.error(f"Error reinitializing system: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to reinitialize system"
        )