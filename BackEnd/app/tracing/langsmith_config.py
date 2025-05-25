"""LangSmith tracing configuration (optional)."""

import os
import logging
from typing import Optional

from ..config import LANGCHAIN_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_PROJECT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_langsmith_tracing() -> bool:
    """Setup LangSmith tracing if API key is available."""
    try:
        if not LANGCHAIN_API_KEY:
            logger.info("LangSmith API key not provided, skipping tracing setup")
            return False
        
        if not LANGCHAIN_TRACING_V2:
            logger.info("LangSmith tracing disabled in config")
            return False
        
        # Set environment variables for LangSmith
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
        
        # Test the connection (optional)
        try:
            from langsmith import Client
            client = Client()
            # Test if we can connect
            client.create_run(
                name="test_connection",
                run_type="tool",
                inputs={"test": "connection"}
            )
            logger.info("LangSmith tracing setup successful")
            return True
            
        except ImportError:
            logger.warning("langsmith package not installed, tracing may not work")
            return True  # Environment variables are set, might still work
            
        except Exception as e:
            logger.warning(f"LangSmith connection test failed: {str(e)}")
            return True  # Still return True as env vars are set
            
    except Exception as e:
        logger.error(f"Error setting up LangSmith tracing: {str(e)}")
        return False


def get_tracing_status() -> dict:
    """Get current tracing configuration status."""
    return {
        "langsmith_enabled": LANGCHAIN_TRACING_V2,
        "api_key_present": bool(LANGCHAIN_API_KEY),
        "project_name": LANGCHAIN_PROJECT,
        "environment_configured": all([
            os.environ.get("LANGCHAIN_API_KEY"),
            os.environ.get("LANGCHAIN_TRACING_V2"),
            os.environ.get("LANGCHAIN_PROJECT")
        ])
    }


# Initialize tracing on module import
tracing_enabled = setup_langsmith_tracing()

if __name__ == "__main__":
    status = get_tracing_status()
    print(f"Tracing Status: {status}")