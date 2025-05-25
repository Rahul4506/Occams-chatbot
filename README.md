# Occam's Advisory RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system that provides accurate answers about Occam's Advisory based on scraped website content.

## 🚀 Features

- **Web Scraping**: Automatically scrapes Occam's Advisory website content
- **Vector Database**: Stores content in ChromaDB with semantic embeddings
- **Smart Retrieval**: Uses similarity search to find relevant information
- **LLM Integration**: Powered by Groq's fast inference API
- **Debug Mode**: Comprehensive debugging tools to troubleshoot issues
- **API & Frontend**: FastAPI backend with Streamlit frontend

## 📋 Prerequisites

- Python 3.8+
- Groq API Key
- Internet connection for web scraping

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd occams-rag-chatbot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env file with your API keys
   ```

   Required environment variables:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   TARGET_WEBSITE_URL=https://occamsadvisory.com
   ```

## ⚙️ Configuration

Key configuration options in `config.py`:

```python
# LLM Settings
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1
MAX_TOKENS = 2048

# RAG Settings
TOP_K_RESULTS = 20
SIMILARITY_THRESHOLD = 0.25
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Scraping Settings
MAX_PAGES_TO_SCRAPE = 100
SCRAPING_DELAY = 1
```

## 🚀 Quick Start

### 1. Initialize the System

```bash
# Run web scraping
python -m backend.scraper.web_scraper

# Initialize RAG system with debugging
python -m backend.rag.rag_chain
```

### 2. Start the API Server

```bash
python -m backend.api.main
```

### 3. Launch Frontend

```bash
streamlit run frontend/app.py
```

## 🔧 Usage

### Command Line Testing

```python
from backend.rag.rag_chain import get_rag_chain

# Initialize RAG chain
chain = get_rag_chain()

# Ask questions
result = chain.answer_question("Who is the CEO of Occam's Advisory?")
print(result['answer'])
```

### API Endpoints

- `POST /ask` - Ask a question
- `GET /health` - Health check
- `GET /stats` - System statistics

Example API request:
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What services does Occam's Advisory provide?"}'
```

## 🐛 Debugging

### Common Issues & Solutions

1. **"I don't have information about that"**
   ```bash
   # Run with debugging to inspect database content
   python -m backend.rag.rag_chain
   
   # Check if content was scraped properly
   ls backend/data/
   
   # Lower similarity threshold in config.py
   SIMILARITY_THRESHOLD = 0.2
   ```

2. **No CEO/Leadership information found**
   - Verify scraping captured leadership pages
   - Check if content is JavaScript-rendered
   - Increase `MAX_PAGES_TO_SCRAPE`

3. **Poor answer quality**
   - Adjust `LLM_TEMPERATURE` (lower = more consistent)
   - Increase `TOP_K_RESULTS` for more context
   - Try different `LLM_MODEL_NAME`

### Debug Mode

Enable comprehensive debugging:

```python
# In config.py
DEBUG_MODE = True

# Run with debugging
chain = get_rag_chain()
result = chain.answer_question("your question", debug_mode=True)
```

This will show:
- Documents retrieved from vector database
- Similarity scores
- Context sent to LLM
- Database inspection results

## 📁 Project Structure

```
occams-rag-chatbot/
├── backend/
│   ├── config/
│   │   └── __init__.py          # Configuration settings
│   ├── scraper/
│   │   ├── web_scraper.py       # Website scraping logic
│   │   └── embedding_utils.py   # Vector database management
│   ├── rag/
│   │   └── rag_chain.py         # RAG implementation
│   ├── api/
│   │   └── main.py              # FastAPI server
│   ├── data/                    # Scraped content storage
│   └── db/                      # ChromaDB storage
├── frontend/
│   └── app.py                   # Streamlit interface
├── requirements.txt
├── .env.example
└── README.md
```

## 🔄 Updating Content

To refresh the knowledge base:

```bash
# Re-scrape website
python -m backend.scraper.web_scraper

# Rebuild vector database
python -m backend.rag.rag_chain
```

## 🎯 Supported Questions

The chatbot can answer questions about:
- Company information and history
- Leadership team and executives
- Services and offerings
- Contact information and locations
- Awards and recognition
- Mission and vision

## 📊 Performance Tuning

### For Better Accuracy:
- Lower `SIMILARITY_THRESHOLD` (0.2-0.3)
- Increase `TOP_K_RESULTS` (15-25)
- Use more stable LLM models
- Improve chunking strategy

### For Faster Responses:
- Use smaller, faster models (`llama-3.1-8b-instant`)
- Reduce `MAX_TOKENS`
- Decrease `TOP_K_RESULTS`

## 🚨 Troubleshooting

### Environment Issues
```bash
# Check Python version
python --version  # Should be 3.8+

# Verify dependencies
pip list | grep -E "(groq|chromadb|sentence-transformers)"

# Test API key
python -c "import os; print('GROQ_API_KEY' in os.environ)"
```

### Database Issues
```bash
# Reset vector database
rm -rf backend/db/chroma_db
python -m backend.rag.rag_chain
```

### Scraping Issues
```bash
# Check scraped content
ls -la backend/data/
head -n 20 backend/data/scraped_content.json
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.



