"""Configuration settings for the RAG chatbot."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data"
DB_DIR = BACKEND_DIR / "db"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY must be set in environment variables")

# LangSmith Configuration (Optional)
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "occams-rag-chatbot")

# ChromaDB Configuration
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(DB_DIR / "chroma_db"))

# Website Configuration
TARGET_WEBSITE_URL = os.getenv("TARGET_WEBSITE_URL", "https://occamsadvisory.com")

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# LLM Configuration

LLM_MODEL_NAME = "mistral-saba-24b"
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Frontend Configuration
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# RAG Configuration
TOP_K_RESULTS = 10
SIMILARITY_THRESHOLD = 0.2

# Scraping Configuration
SCRAPING_DELAY = 1  # seconds between requests
MAX_PAGES_TO_SCRAPE = 50

print("Configuration loaded successfully.")
