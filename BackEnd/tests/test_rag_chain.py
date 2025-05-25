"""Tests for RAG chain functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from app.core.rag_chain import OccamsRAGChain


class TestOccamsRAGChain:
    """Test cases for OccamsRAGChain."""
    
    @patch('app.core.rag_chain.EmbeddingManager')
    @patch('app.core.rag_chain.Groq')
    def setUp(self, mock_groq, mock_embedding_manager):
        """Set up test fixtures."""
        self.mock_embedding_manager = Mock()
        self.mock_groq_client = Mock()
        
        mock_embedding_manager.return_value = self.mock_embedding_manager
        mock_groq.return_value = self.mock_groq_client
        
        self.rag_chain = OccamsRAGChain()
    
    @patch('app.core.rag_chain.EmbeddingManager')
    @patch('app.core.rag_chain.Groq')
    def test_initialization(self, mock_groq, mock_embedding_manager):
        """Test RAG chain initialization."""
        rag_chain = OccamsRAGChain()
        
        assert rag_chain is not None
        assert hasattr(rag_chain, 'embedding_manager')
        assert hasattr(rag_chain, 'groq_client')
        assert hasattr(rag_chain, 'system_prompt')
    
    def test_format_context_with_documents(self):
        """Test context formatting with documents."""
        # Create test documents
        documents = [
            {
                'content': 'Occam\'s Advisory is a consulting firm.',
                'metadata': {
                    'url': 'https://occamsadvisory.com/about',
                    'title': 'About Us'
                },
                'score': 0.85
            },
            {
                'content': 'We provide strategic consulting services.',
                'metadata': {
                    'url': 'https://occamsadvisory.com/services',
                    'title': 'Our Services'
                },
                'score': 0.78
            }
        ]
        
        # Mock the RAG chain
        with patch('app.core.rag_chain.EmbeddingManager'), \
             patch('app.core.rag_chain.Groq'):
            rag_chain = OccamsRAGChain()
            context = rag_chain.format_context(documents)
        
        assert 'Document 1' in context
        assert 'Document 2' in context
        assert 'Occam\'s Advisory is a consulting firm.' in context
        assert 'We provide strategic consulting services.' in context
        assert 'https://occamsadvisory.com/about' in context
    
    def test_format_context_empty_documents(self):
        """Test context formatting with no documents."""
        with patch('app.core.rag_chain.EmbeddingManager'), \
             patch('app.core.rag_chain.Groq'):
            rag_chain = OccamsRAGChain()
            context = rag_chain.format_context([])
        
        assert 'No relevant information found' in context
    
    @patch('app.core.rag_chain.EmbeddingManager')
    @patch('app.core.rag_chain.Groq')
    def test_retrieve_relevant_documents_success(self, mock_groq, mock_embedding_manager):
        """Test document retrieval success."""
        # Mock embedding manager
        mock_embedding_instance = Mock()
        mock_embedding_instance.similarity_search.return_value = [
            {
                'content': 'Test content',
                'metadata': {'url': 'test.com', 'title': 'Test'},
                'score': 0.8
            }
        ]
        mock_embedding_manager.return_value = mock_embedding_instance
        
        rag_chain = OccamsRAGChain()
        results = rag_chain.retrieve_relevant_documents("test query")
        
        assert len(results) == 1
        assert results[0]['score'] == 0.8
        mock_embedding_instance.similarity_search.assert_called_once()
    
    @patch('app.core.rag_chain.EmbeddingManager')
    @patch('app.core.rag_chain.Groq')
    def test_retrieve_relevant_documents_filtering(self, mock_groq, mock_embedding_manager):
        """Test document retrieval with similarity filtering."""
        # Mock embedding manager with low score results
        mock_embedding_instance = Mock()
        mock_embedding_instance.similarity_search.return_value = [
            {
                'content': 'High relevance content',
                'metadata': {'url': 'test1.com', 'title': 'Test 1'},
                'score': 0.8  # Above threshold
            },
            {
                'content': 'Low relevance content',
                'metadata': {'url': 'test2.com', 'title': 'Test 2'},
                'score': 0.5  # Below threshold
            }
        ]
        mock_embedding_manager.return_value = mock_embedding_instance
        
        rag_chain = OccamsRAGChain()
        results = rag_chain.retrieve_relevant_documents("test query")
        
        # Should only return documents above similarity threshold
        assert len(results) == 1
        assert results[0]['score'] == 0.8
    
    @patch('app.core.rag_chain.EmbeddingManager')
    @patch('app.core.rag_chain.Groq')
    def test_generate_response_success(self, mock_groq, mock_embedding_manager):
        """Test response generation success."""
        # Mock Groq client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated response"
        
        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_groq_instance
        
        mock_embedding_manager.return_value = Mock()
        
        rag_chain = OccamsRAGChain()
        response = rag_chain.generate_response("test query", "test context")
        
        assert response == "Generated response"
        mock_groq_instance.chat.completions.create.assert_called_once()
    
    @patch('app.core.rag_chain.EmbeddingManager')
    @patch('app.core.rag_chain.Groq')
    def test_generate_response_error_handling(self, mock_groq, mock_embedding_manager):
        """Test response generation error handling."""
        # Mock Groq client to raise exception
        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_groq.return_value = mock_groq_instance
        
        mock_embedding_manager.return_value = Mock()
        
        rag_chain = OccamsRAGChain()
        response = rag_chain.generate_response("test query", "test context")
        
        assert "trouble generating a response" in response
    
    @patch('app.core.rag_chain.EmbeddingManager')
    @patch('app.core.rag_chain.Groq')
    def test_answer_question_complete_flow(self, mock_groq, mock_embedding_manager):
        """Test complete question answering flow."""
        # Mock embedding manager
        mock_embedding_instance = Mock()
        mock_embedding_instance.similarity_search.return_value = [
            {
                'content': 'Occam\'s Advisory provides consulting services.',
                'metadata': {
                    'url': 'https://occamsadvisory.com/services',
                    'title': 'Services'
                },
                'score': 0.9
            }
        ]
        mock_embedding_manager.return_value = mock_embedding_instance
        
        # Mock Groq client
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Occam's Advisory is a consulting firm that provides strategic advice to businesses."
        
        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_groq_instance
        
        rag_chain = OccamsRAGChain()
        result = rag_chain.answer_question("What is Occam's Advisory?")
        
        assert "answer" in result
        assert "sources" in result
        assert "context_used" in result
        assert result["context_used"] is True
        assert len(result["sources"]) == 1
        assert result["sources"][0]["url"] == "https://occamsadvisory.com/services"
    
    @patch('app.core.rag_chain.EmbeddingManager')
    @patch('app.core.rag_chain.Groq')
    def test_health_check_healthy(self, mock_groq, mock_embedding_manager):
        """Test health check when system is healthy."""
        # Mock embedding manager
        mock_embedding_instance = Mock()
        mock_embedding_instance.get_collection_stats.return_value = {
            "total_documents": 100,
            "collection_name": "occams_advisory"
        }
        mock_embedding_manager.return_value = mock_embedding_instance
        
        # Mock successful answer generation
        mock_groq_instance = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_groq_instance
        
        rag_chain = OccamsRAGChain()
        
        # Mock the retrieve method to avoid actual retrieval
        with patch.object(rag_chain, 'retrieve_relevant_documents') as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    'content': 'Test content',
                    'metadata': {'url': 'test.com', 'title': 'Test'},
                    'score': 0.8
                }
            ]
            
            health = rag_chain.health_check()
        
        assert health["status"] == "healthy"
        assert "vector_db_stats" in health
        assert health["test_query_successful"] is True
    
    @patch('app.core.rag_chain.EmbeddingManager')
    @patch('app.core.rag_chain.Groq')
    def test_health_check_unhealthy(self, mock_groq, mock_embedding_manager):
        """Test health check when system is unhealthy."""
        # Mock embedding manager to raise exception
        mock_embedding_instance = Mock()
        mock_embedding_instance.get_collection_stats.side_effect = Exception("DB Error")
        mock_embedding_manager.return_value = mock_embedding_instance
        
        mock_groq.return_value = Mock()
        
        rag_chain = OccamsRAGChain()
        health = rag_chain.health_check()
        
        assert health["status"] == "unhealthy"
        assert "error" in health


class TestRAGChainHelpers:
    """Test helper functions in RAG chain module."""
    
    @patch('app.core.rag_chain.OccamsRAGChain')
    def test_get_rag_chain_singleton(self, mock_rag_chain_class):
        """Test that get_rag_chain returns singleton instance."""
        from app.core.rag_chain import get_rag_chain
        
        # Reset global variable
        import app.core.rag_chain as rag_module
        rag_module.rag_chain = None
        
        # First call should create instance
        chain1 = get_rag_chain()
        assert chain1 is not None
        
        # Second call should return same instance
        chain2 = get_rag_chain()
        assert chain1 is chain2
    
    @patch('app.core.rag_chain.get_rag_chain')
    def test_initialize_rag_system_success(self, mock_get_rag_chain):
        """Test successful RAG system initialization."""
        from app.core.rag_chain import initialize_rag_system
        
        # Mock RAG chain
        mock_chain = Mock()
        mock_chain.health_check.return_value = {"status": "healthy", "vector_db_stats": {}}
        mock_get_rag_chain.return_value = mock_chain
        
        result = initialize_rag_system()
        
        assert result["status"] == "healthy"
        mock_chain.health_check.assert_called_once()
    
    @patch('app.core.rag_chain.get_rag_chain')
    def test_initialize_rag_system_failure(self, mock_get_rag_chain):
        """Test RAG system initialization failure."""
        from app.core.rag_chain import initialize_rag_system
        
        # Mock RAG chain to raise exception
        mock_get_rag_chain.side_effect = Exception("Initialization failed")
        
        result = initialize_rag_system()
        
        assert result["status"] == "failed"
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__])