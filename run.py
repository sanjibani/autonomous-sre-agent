#!/usr/bin/env python3
"""
Autonomous SRE Agent - Application Entry Point
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from app.routes import main
from src.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG


def create_app():
    """Create and configure the Flask application"""
    app = Flask(
        __name__,
        template_folder='app/templates',
        static_folder='app/static'
    )
    
    # Register blueprint
    app.register_blueprint(main)
    
    return app


def init_system():
    """Initialize system components"""
    print("\n" + "="*60)
    print("🤖 Autonomous SRE Agent - Starting Up")
    print("="*60 + "\n")
    
    # Initialize embedding model (this preloads it)
    print("📦 Loading embedding model...")
    from src.embeddings import get_embedding_service
    get_embedding_service()
    
    # Initialize vector database
    print("🗄️  Initializing vector database...")
    from src.vectordb import get_vectordb_service
    vectordb = get_vectordb_service()
    stats = vectordb.get_collection_stats()
    print(f"   - Runbooks: {stats['runbooks']}")
    print(f"   - Feedback entries: {stats['feedback']}")
    
    # Check Ollama
    print("🧠 Checking LLM (Ollama)...")
    try:
        import ollama
        models = ollama.list()
        available_models = [m['name'] for m in models.get('models', [])]
        if available_models:
            print(f"   - Available models: {', '.join(available_models[:3])}")
        else:
            print("   - No models found. Run: ollama pull llama3.2:3b")
    except Exception as e:
        print(f"   - Ollama not available: {e}")
        print("   - Using rule-based fallback mode")
    
    print("\n" + "="*60)
    print(f"✅ Server ready at http://{FLASK_HOST}:{FLASK_PORT}")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Initialize components
    init_system()
    
    # Create and run app
    app = create_app()
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG
    )
