"""Text chunking utilities using LangChain TextSplitters."""

import json
import logging
from typing import List, Dict
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ..config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentChunker:
    """Handles text chunking for RAG pipeline."""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_scraped_data(self) -> List[Dict]:
        """Load scraped data from JSON file."""
        data_file = DATA_DIR / "scraped_data.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Scraped data file not found: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
               
        
        logger.info(f"Loaded {len(data)} scraped pages")
        return data
    
    def create_documents(self, scraped_data: List[Dict]) -> List[Document]:
        """Convert scraped data to LangChain Documents."""
        documents = []
    

        for item in scraped_data:
            # Combine title and content for better context
            content = f"Title: {item['title']}\n\n{item['content']}"
            
            # Skip empty content
            if not content.strip() or len(content.strip()) < 50:
                continue
            
            document = Document(
                page_content=content,
                metadata={
                    'url': item['url'],
                    'title': item['title'],
                    'meta_description': item.get('meta_description', ''),
                    'scraped_at': item.get('scraped_at'),
                    'source': 'occams_advisory'
                }
            )
            documents.append(document)
        
        logger.info(f"Created {len(documents)} documents")
        return documents
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        chunks = []
        
        for doc in documents:
            # Split the document
            doc_chunks = self.text_splitter.split_documents([doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    'chunk_id': f"{hash(doc.metadata['url'])}_{i}",
                    'chunk_index': i,
                    'total_chunks': len(doc_chunks),
                    'chunk_size': len(chunk.page_content)
                })
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def save_chunks(self, chunks: List[Document]) -> None:
        """Save chunks to JSON file for inspection."""
        chunks_data = []
        
        for chunk in chunks:
            chunks_data.append({
                'content': chunk.page_content,
                'metadata': chunk.metadata
            })
        
        output_file = DATA_DIR / "chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")
    
    def process_scraped_data(self) -> List[Document]:
        """Complete pipeline: load data -> create documents -> chunk."""
        # Load scraped data
        scraped_data = self.load_scraped_data()
        
        # Create documents
        documents = self.create_documents(scraped_data)
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Save chunks for inspection
        self.save_chunks(chunks)
        
        return chunks
    
    def get_chunk_stats(self, chunks: List[Document]) -> Dict:
        """Get statistics about the chunks."""
        if not chunks:
            return {}
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        unique_urls = set(chunk.metadata['url'] for chunk in chunks)
        
        stats = {
            'total_chunks': len(chunks),
            'unique_pages': len(unique_urls),
            'avg_chunk_size': sum(chunk_sizes) / len(chunk_sizes),
            'min_chunk_size': min(chunk_sizes),
            'max_chunk_size': max(chunk_sizes),
            'total_content_length': sum(chunk_sizes)
        }
        
        return stats


def create_chunks_from_scraped_data() -> List[Document]:
    """Main function to create chunks from scraped data."""
    chunker = DocumentChunker()
    chunks = chunker.process_scraped_data()
    
    # Print statistics
    stats = chunker.get_chunk_stats(chunks)
    logger.info(f"Chunking Statistics: {stats}")
    
    return chunks


if __name__ == "__main__":
    chunks = create_chunks_from_scraped_data()
