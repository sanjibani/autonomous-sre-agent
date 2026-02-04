# Autonomous SRE Agent

An intelligent agent that ingests raw logs, clusters them into incidents, investigates root causes using RAG, and drafts fixes for human approval.

## Features

- **Log Ingestion & Clustering**: Uses DBSCAN to group similar log entries
- **RAG-based Investigation**: Searches runbooks and past incidents for context
- **Human-in-the-Loop Learning**: Learns from user feedback to improve over time
- **Web Dashboard**: View alerts, provide feedback, and monitor incidents

## Tech Stack

- **Embeddings** (Hybrid): 
    - Default: **OpenAI `text-embedding-3-small`** (Cloud, 1536-dim)
    - Fallback: `all-MiniLM-L6-v2` (Local, 384-dim)
- **Vector DB**: ChromaDB (Local persistence)
- **LLM**: Ollama (Local Llama 3.2, privacy-focused)
- **Web Framework**: Flask
- **Clustering**: DBSCAN (scikit-learn)

## Architecture

```mermaid
graph TD
    User[User / System Logs] -->|Raw Logs| Ingest[Ingestion Service]
    
    subgraph "Perception Layer (Cloud)"
        Ingest -->|Batch API Call| OpenAI[OpenAI Embeddings API]
        OpenAI -->|Vectors (1536-dim)| VectorDB[(ChromaDB - Local)]
    end
    
    subgraph "Reasoning Layer (Local)"
        VectorDB -->|Vectors| Clustering[DBSCAN Clustering]
        Clustering -->|Cluster Info| Agent[SRE Agent Logic]
        VectorDB -.->|Context (Runbooks)| Agent
        Agent -->|Prompt| Ollama[Ollama LLM (Llama 3.2)]
        Ollama -->|Recommendation| Dashboard[Web Dashboard]
    end
    
    Dashboard -->|Feedback| VectorDB
```

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed locally:
   ```bash
   # macOS
   brew install ollama
   
   # Or download from https://ollama.ai
   ```

3. **Pull the LLM model**:
   ```bash
   ollama pull llama3.2:3b
   ```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Start the application
python run.py

# Open browser at http://localhost:5000
```

## Project Structure

```
├── data/
│   ├── logs/           # Log files to analyze
│   └── runbooks/       # Synthetic runbooks for RAG
├── src/
│   ├── embeddings.py   # Log embedding with MiniLM
│   ├── clustering.py   # DBSCAN clustering
│   ├── vectordb.py     # ChromaDB operations
│   ├── rag.py          # RAG retrieval logic
│   └── agent.py        # Main agent logic
├── app/
│   ├── routes.py       # Flask API routes
│   └── templates/      # HTML templates
└── tests/              # Unit tests
```

## Data Source

Uses [Loghub](https://github.com/logpai/loghub) HDFS dataset for real production logs.

## License

MIT
