"""
Configuration settings for Autonomous SRE Agent
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
RUNBOOKS_DIR = DATA_DIR / "runbooks"
CHROMADB_DIR = BASE_DIR / "chromadb_data"

# Ensure directories exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RUNBOOKS_DIR.mkdir(parents=True, exist_ok=True)
CHROMADB_DIR.mkdir(parents=True, exist_ok=True)

# Embedding settings
# Options: "all-MiniLM-L6-v2" (local, free) or "text-embedding-3-small" (OpenAI, paid)
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

if USE_OPENAI_EMBEDDINGS:
    EMBEDDING_MODEL = "text-embedding-3-small"
    EMBEDDING_DIMENSION = 1536
else:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384

# Clustering settings
DBSCAN_EPS = 0.5  # Maximum distance between samples
DBSCAN_MIN_SAMPLES = 3  # Minimum samples in a cluster

# LLM Provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # "openai" or "ollama"

# OpenAI settings
OPENAI_MODEL = "gpt-3.5-turbo"  # Cost-effective and fast

# Ollama settings
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ChromaDB collections
COLLECTION_RUNBOOKS = "runbooks"
COLLECTION_FEEDBACK = "feedback"
COLLECTION_INCIDENTS = "incidents"

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5001
FLASK_DEBUG = True

# Severity thresholds
SEVERITY_THRESHOLDS = {
    "high": 0.8,
    "medium": 0.5,
    "low": 0.0
}
