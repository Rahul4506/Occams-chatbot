"""Tests for chat API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from app.main import app

client = TestClient(app)


class TestChatAPI:
    """Test cases for chat API."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_api_info_endpoint(self):
        """Test API info endpoint."""
        response = client.get("/api/info")
        assert response.status_code == 200
        data = response.json()
        assert "api_name" in data
        assert "endpoints" in data
    
    @patch('app.core.rag_chain.get_rag_chain')
    def test_health_endpoint_healthy(self, mock_get_rag_chain):
        """Test health endpoint when system is healthy."""
        # Mock RAG chain
        mock_rag_chain = Mock()
        mock_rag_chain.health_check.return_value = {
            "status": "healthy",
            "vector_db_stats": {"total_documents": 100}
        }
        mock_get_rag_chain.return_value = mock_rag_chain
        
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "vector_db_stats" in data
    
    @patch('app.core.rag_chain.get_rag_chain')
    def test_health_endpoint_unhealthy(self, mock_get_rag_chain):
        """Test health endpoint when system is unhealthy."""
        # Mock RAG chain
        mock_rag_chain = Mock()
        mock_rag_chain.health_check.return_value = {
            "status": "unhealthy",
            "error": "Database connection failed"
        }
        mock_get_rag_chain.return_value = mock_rag_chain
        
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert "Database connection failed" in data["message"]
    
    @patch('app.core.rag_chain.get_rag_chain')
    def test_chat_endpoint_success(self, mock_get_rag_chain):
        """Test successful chat interaction."""
        # Mock RAG chain
        mock_rag_chain = Mock()
        mock_rag_chain.answer_question.return_value = {
            "answer": "Occam's Advisory is a consulting firm that provides strategic advice.",
            "sources": [
                {
                    "url": "https://occamsadvisory.com/about",
                    "title": "About Us",
                    "score": 0.85
                }
            ],
            "context_used": True
        }
        mock_get_rag_chain.return_value = mock_rag_chain
        
        # Test chat request
        response = client.post("/api/chat", json={
            "message": "What is Occam's Advisory?",
            "conversation_id": "test_conv"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert data["context_used"] is True
        assert len(data["sources"]) > 0
    
    def test_chat_endpoint_invalid_input(self):
        """Test chat endpoint with invalid input."""
        # Empty message
        response = client.post("/api/chat", json={
            "message": "",
            "conversation_id": "test_conv"
        })
        assert response.status_code == 422  # Validation error
        
        # Message too long
        long_message = "x" * 1001
        response = client.post("/api/chat", json={
            "message": long_message,
            "conversation_id": "test_conv"
        })
        assert response.status_code == 422
    
    @patch('app.core.rag_chain.get_rag_chain')
    def test_chat_endpoint_error_handling(self, mock_get_rag_chain):
        """Test chat endpoint error handling."""
        # Mock RAG chain to raise exception
        mock_rag_chain = Mock()
        mock_rag_chain.answer_question.side_effect = Exception("Test error")
        mock_get_rag_chain.return_value = mock_rag_chain
        
        response = client.post("/api/chat", json={
            "message": "Test question",
            "conversation_id": "test_conv"
        })
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data["detail"].lower()
    
    @patch('app.core.rag_chain.get_rag_chain')
    def test_stats_endpoint(self, mock_get_rag_chain):
        """Test stats endpoint."""
        # Mock RAG chain and embedding manager
        mock_embedding_manager = Mock()
        mock_embedding_manager.get_collection_stats.return_value = {
            "total_documents": 150,
            "collection_name": "occams_advisory"
        }
        mock_embedding_manager.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        mock_rag_chain = Mock()
        mock_rag_chain.embedding_manager = mock_embedding_manager
        mock_rag_chain.model_name = "llama-3.1-70b-versatile"
        
        mock_get_rag_chain.return_value = mock_rag_chain
        
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert "system_status" in data
        assert "vector_database" in data
        assert "embedding_model" in data
        assert "llm_model" in data


@pytest.fixture
def mock_rag_system():
    """Fixture for mocking RAG system components."""
    with patch('app.core.rag_chain.initialize_rag_system') as mock_init:
        mock_init.return_value = {"status": "healthy"}
        yield mock_init


class TestChatIntegration:
    """Integration tests for chat functionality."""
    
    def test_chat_flow_integration(self, mock_rag_system):
        """Test complete chat flow integration."""
        # This would be an integration test that tests the complete flow
        # In a real scenario, you'd have test data and a test database
        pass
    
    def test_conversation_continuity(self, mock_rag_system):
        """Test that conversation context is maintained."""
        # Test multiple messages in same conversation
        pass


if __name__ == "__main__":
    pytest.main([__file__])