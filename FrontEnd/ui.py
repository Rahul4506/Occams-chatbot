"""Streamlit frontend for Occam's Advisory RAG Chatbot."""

import streamlit as st
import requests
import time
from typing import Dict, List
import json

# Configuration
API_BASE_URL = "http://127.0.0.1:8080/api"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
STATS_ENDPOINT = f"{API_BASE_URL}/stats"


def check_api_health() -> Dict:
    """Check if the API is healthy."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "unhealthy", "message": str(e)}


def get_system_stats() -> Dict:
    """Get system statistics."""
    try:
        response = requests.get(STATS_ENDPOINT, timeout=5)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def send_chat_message(message: str) -> Dict:
    """Send chat message to API."""
    try:
        payload = {
            "message": message,
            "conversation_id": st.session_state.get("conversation_id", "")
        }
        print(f"Sending payload: {payload}")  # DEBUG
        response = requests.post(
            CHAT_ENDPOINT,
            json=payload,
            timeout=30
        )
        
        print(f"Response status: {response.status_code}")  # DEBUG
        print(f"Response text: {response.text}")  # DEBUG
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": f"API Error: {response.status_code}",
                "details": response.text
            }
            
    except Exception as e:
        return {"error": str(e)}


def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = f"conv_{int(time.time())}"
    
    if "api_healthy" not in st.session_state:
        st.session_state.api_healthy = False


def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Occam's Advisory Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid #1f4e79;
    }
    
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
        color: black;
    }
    
    .assistant-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
        color: black;
    }
    
    .source-item {
        background-color: #f5f5f5;
        padding: 0.5rem;
        margin: 0.2rem 0;
        border-radius: 5px;
        font-size: 0.8rem;
        color: black;
    }
    
    .status-healthy {
        color: #4caf50;
        font-weight: bold;
    }
    
    .status-unhealthy {
        color: #f44336;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Occam's Advisory Chatbot</h1>
        <p>Ask me anything about Occam's Advisory services, team, and expertise</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        # Check API health
        health_status = check_api_health()
        
        if health_status["status"] == "healthy":
            st.markdown('<p class="status-healthy">✅ System Healthy</p>', unsafe_allow_html=True)
            st.session_state.api_healthy = True
        else:
            st.markdown('<p class="status-unhealthy">❌ System Unhealthy</p>', unsafe_allow_html=True)
            st.error(f"Error: {health_status.get('message', 'Unknown error')}")
            st.session_state.api_healthy = False
        
        # System Statistics
        if st.session_state.api_healthy:
            st.subheader("System Info")
            stats = get_system_stats()
            
            if "error" not in stats:
                st.metric("Documents in DB", stats.get("vector_database", {}).get("total_documents", "N/A"))
                st.text(f"Embedding Model: {stats.get('embedding_model', 'N/A')}")
                st.text(f"LLM Model: {stats.get('llm_model', 'N/A')}")
            
        # Instructions
        st.subheader("How to Use")
        st.markdown("""
        1. Type your question about Occam's Advisory
        2. The bot will search through website content
        3. Get answers with source citations
        4. Ask follow-up questions for more details
        
        **Example Questions:**
        - What is Occam's Advisory?
        - Information about Leadership Teams ?
        - What services does Occam's Advisory offer?
        - Head office of Occams Advisory??
        - Top 3 Awards in 2025??
        - How can I contact Occam's Advisory?
        """)
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    if not st.session_state.api_healthy:
        st.error("⚠️ The chatbot is currently unavailable. Please check the system status in the sidebar.")
        st.stop()
    
    # Display chat messages
    chat_container = st.container()
    
    with chat_container:
        # Display existing messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    st.markdown("**Sources:**")
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"""
                        <div class="source-item">
                            {i}. <a href="{source["url"]}" target="_blank">{source["title"]}</a> 
                            (Relevance: {source["score"]:.2f})
                        </div>
                        """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask me about Occam's Advisory..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with chat_container:
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {prompt}
            </div>
            """, unsafe_allow_html=True)
        
        # Show thinking indicator
        with st.spinner("Thinking..."):
            # Send message to API
            response = send_chat_message(prompt)
        
        # Handle response
        if "error" in response:
            error_msg = f"Sorry, I encountered an error: {response['error']}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.error(error_msg)
        else:
            # Add assistant response to chat
            assistant_message = {
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", [])
            }
            st.session_state.messages.append(assistant_message)
            
            # Display assistant response
            with chat_container:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {response["answer"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Display sources
                if response.get("sources"):
                    st.markdown("**Sources:**")
                    for i, source in enumerate(response["sources"][:5], 1):
                        st.markdown(f"""
                        <div class="source-item">
                            {i}. <a href="{source["url"]}" target="_blank">{source["title"]}</a> 
                            (Relevance: {source["score"]:.2f})
                        </div>
                        """, unsafe_allow_html=True)
        
        # Rerun to update the display
        # st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        Powered by Groq LLM, ChromaDB, and HuggingFace Embeddings<br>
        Built with ❤️ by Rahul
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()