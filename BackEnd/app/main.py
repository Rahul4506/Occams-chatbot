"""FastAPI main application."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import chat
from .core.rag_chain import initialize_rag_system
from .tracing.langsmith_config import get_tracing_status
from .config import API_HOST, API_PORT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Occam's Advisory RAG Chatbot...")
    
    try:
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        rag_status = initialize_rag_system()
        
        if rag_status["status"] == "healthy":
            logger.info("✅ RAG system initialized successfully")
        else:
            logger.error(f"❌ RAG system initialization failed: {rag_status.get('error')}")
        
        # Check tracing status
        tracing_status = get_tracing_status()
        if tracing_status["langsmith_enabled"]:
            logger.info("✅ LangSmith tracing enabled")
        else:
            logger.info("ℹ️  LangSmith tracing disabled")
            
        logger.info("🚀 Application startup complete")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="Occam's Advisory RAG Chatbot",
    description="A RAG-based chatbot that answers questions about Occam's Advisory using information from their website",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])


@app.get("/")
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Occam's Advisory RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.get("/api/info")
async def get_api_info():
    """Get API information and status."""
    try:
        tracing_status = get_tracing_status()
        
        return {
            "api_name": "Occam's Advisory RAG Chatbot",
            "version": "1.0.0",
            "description": "RAG-based chatbot for Occam's Advisory information",
            "endpoints": {
                "chat": "/api/chat",
                "health": "/api/health", 
                "stats": "/api/stats",
                "docs": "/docs"
            },
            "features": {
                "web_scraping": "Playwright",
                "embeddings": "HuggingFace Sentence Transformers",
                "vector_db": "ChromaDB",
                "llm": "Groq (Llama)",
                "tracing": tracing_status["langsmith_enabled"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting API info: {str(e)}")
        return {"error": "Failed to retrieve API information"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"Starting server on {API_HOST}:{API_PORT}")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
