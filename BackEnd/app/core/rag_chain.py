"""RAG chain implementation with Groq LLM."""

import logging
from typing import List, Dict, Optional
from groq import Groq

from .embedding_utils import EmbeddingManager
from ..config import (
    GROQ_API_KEY, LLM_MODEL_NAME, LLM_TEMPERATURE, 
    MAX_TOKENS, TOP_K_RESULTS, SIMILARITY_THRESHOLD
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OccamsRAGChain:
    """RAG chain for Occam's Advisory chatbot."""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.model_name = LLM_MODEL_NAME
        self.temperature = LLM_TEMPERATURE
        self.max_tokens = MAX_TOKENS
        
        print(f"Using LLM model: {LLM_MODEL_NAME}")

        
        # System prompt to ensure responses are strictly based on website content
        self.system_prompt = """You are a helpful assistant that answers questions about Occam's Advisory based on the information provided in the context below. 

    IMPORTANT GUIDELINES:
1. use information mentioned in the provided context
2. If the context not found anywhere then mention: "I don't have information about that in the Occam's Advisory materials provided to me."
3. Exract important information like names, addresses, and other details from the context and provide them in the answer.
4. When answering, you may reference that the information comes from Occam's Advisory website
5. Be helpful and conversational
6. If user says hii, greet with for first time only "Hello Rahul, how can I help you today?"
7. If you can partially answer a question based on the context, do so but clearly indicate what information is available and what is not
Context from Occam's Advisory website:
{context}

Question: {question}

Answer based only on the context provided above:"""
    
    def retrieve_relevant_documents(self, query: str) -> List[Dict]:
        """Retrieve relevant documents for the query."""
        try:
            results = self.embedding_manager.similarity_search(
                query=query,
                top_k=TOP_K_RESULTS
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in results 
                if result['score'] >= SIMILARITY_THRESHOLD
            ]
            
            logger.info(f"Retrieved {len(filtered_results)} relevant documents for query")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant information found in Occam's Advisory materials."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc['metadata']
            content = doc['content']
            
            context_part = f"""
Document {i} (Source: {metadata.get('url', 'N/A')}):
Title: {metadata.get('title', 'N/A')}
Content: {content}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using Groq LLM."""
        try:
            # Format the prompt
            prompt = self.system_prompt.format(context=context, question=query)
            
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."
    
    def answer_question(self, question: str) -> Dict:
        """Complete RAG pipeline: retrieve, format, generate."""
        try:
            # Step 1: Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_documents(question)
            
            # Step 2: Format context
            context = self.format_context(relevant_docs)
            
            # Step 3: Generate response
            response = self.generate_response(question, context)
            
            # Return structured response
            return {
                "answer": response,
                "sources": [
                    {
                        "url": doc['metadata'].get('url', 'N/A'),
                        "title": doc['metadata'].get('title', 'N/A'),
                        "score": round(doc['score'], 3)
                    }
                    for doc in relevant_docs
                ],
                "context_used": len(relevant_docs) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "sources": [],
                "context_used": False
            }
    
    def health_check(self) -> Dict:
        """Check if RAG system is working properly."""
        try:
            # Check vector database
            stats = self.embedding_manager.get_collection_stats()
            
            # Test query
            test_response = self.answer_question("What is Occam's Advisory?")
            
            return {
                "status": "healthy",
                "vector_db_stats": stats,
                "test_query_successful": len(test_response["answer"]) > 0
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }


# Global RAG chain instance
rag_chain = None


def get_rag_chain() -> OccamsRAGChain:
    """Get or create RAG chain instance."""
    global rag_chain
    if rag_chain is None:
        rag_chain = OccamsRAGChain()
    return rag_chain


def initialize_rag_system():
    """Initialize the RAG system."""
    try:
        chain = get_rag_chain()
        health = chain.health_check()
        
        if health["status"] == "healthy":
            logger.info("RAG system initialized successfully")
            logger.info(f"Vector DB stats: {health['vector_db_stats']}")
        else:
            logger.error(f"RAG system initialization failed: {health.get('error')}")
            
        return health
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return {"status": "failed", "error": str(e)}


if __name__ == "__main__": 
    # Test the RAG chain
    initialize_rag_system()
    
    chain = get_rag_chain()
    
    # Test questions
    test_questions = [
        # "What is Occam's Advisory?",
        "What services does Occam's Advisory provide?",
        # "Head office of Occams Advisory?",
        # "information about Leadership Teams ?",
        # "Top 3 Awards in 2025?",    
        # "How can I contact Occam's Advisory?"
            
    ]
    
    for question in test_questions:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print(f"{'='*50}")
        
        result = chain.answer_question(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['sources'])}")
        for source in result['sources']:
            print(f"  - {source['title']} (Score: {source['score']})")