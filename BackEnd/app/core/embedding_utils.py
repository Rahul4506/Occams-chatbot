"""Embedding utilities using HuggingFace models."""

import logging
from typing import List
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
import uuid

import torch
# print(torch.__version__)
# print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


from ..config import EMBEDDING_MODEL_NAME, CHROMA_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages embeddings and vector database operations."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize embedding model and ChromaDB client."""
        try:
            # Initialize SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully")
            
            # Initialize ChromaDB
            logger.info(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
            self.chroma_client = chromadb.PersistentClient(
                path=CHROMA_DB_PATH,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Create or get collection
            self.collection = self.chroma_client.get_or_create_collection(
                
                
                
                name="occams_advisory",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_documents(self, documents: List[Document]) -> None:
        """Embed documents and store in ChromaDB."""
        if not documents:
            logger.warning("No documents to embed")
            return
        
        logger.info(f"Embedding {len(documents)} documents...")
        
        # Prepare data for embedding
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(uuid.uuid4()) for _ in documents]
        
        # Generate embeddings in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            try:
                # Generate embeddings
                batch_embeddings = self.embed_texts(batch_texts)
                
                # Store in ChromaDB
                self.collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )
                
                logger.info(f"Embedded batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Error embedding batch {i//batch_size + 1}: {str(e)}")
                continue
        
        logger.info(f"Successfully embedded {len(documents)} documents")
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[dict]:
        """Perform similarity search in the vector database."""
        try:
            # Generate query embedding
            query_embedding = self.embed_texts([query])[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    formatted_results.append({
                        'content': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'score': 1 - results['distances'][0][i]  # Convert distance to similarity
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': self.collection.name,
                'embedding_model': self.model_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {}
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            # Delete the collection
            self.chroma_client.delete_collection(name="occams_advisory")
            
            # Recreate the collection
            self.collection = self.chroma_client.get_or_create_collection(
                name="occams_advisory",
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info("Collection cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            raise
    
    def check_if_collection_exists(self) -> bool:
        """Check if collection has any documents."""
        try:
            return self.collection.count() > 0
        except Exception as e:
            logger.error(f"Error checking collection: {str(e)}")
            return False


def create_vector_database(documents: List[Document]) -> EmbeddingManager:
    """Create vector database from documents."""
    embedding_manager = EmbeddingManager()
    
    # Clear existing collection if needed
    if embedding_manager.check_if_collection_exists():
        logger.info("Collection already exists. Clearing...")
        embedding_manager.clear_collection()
    
    # Embed documents
    embedding_manager.embed_documents(documents)
    
    # Print stats
    stats = embedding_manager.get_collection_stats()
    logger.info(f"Vector database created: {stats}")
    
    return embedding_manager


if __name__ == "__main__":
    # Test embedding functionality
    from .chunking_utils import create_chunks_from_scraped_data
    
    chunks = create_chunks_from_scraped_data()
    embedding_manager = create_vector_database(chunks)