# Autonomous SRE Agent

An intelligent agent that ingests raw logs, clusters them into incidents, investigates root causes using RAG, and drafts fixes for human approval.

## Features

- **Log Ingestion & Clustering**: Uses DBSCAN to group similar log entries
- **RAG-based Investigation**: Searches runbooks and past incidents for context
- **Human-in-the-Loop Learning**: Learns from user feedback to improve over time
- **Web Dashboard**: View alerts, provide feedback, and monitor incidents

## Tech Stack

- **Embeddings**: **OpenAI `text-embedding-3-small`** (Cloud, 1536-dim)
- **Vector DB**: ChromaDB (Local persistence)
- **LLM**: **OpenAI `gpt-3.5-turbo`** (Cloud)
- **Web Framework**: **Streamlit** (Python-only UI)
- **Clustering**: DBSCAN (scikit-learn)

## Architecture

```mermaid
graph TD
    User[User / System Logs] -->|Raw Logs| Ingest[Ingestion Service]
    
    subgraph "Perception Layer"
        Ingest -->|Batch API Call| OpenAI[OpenAI Embeddings API]
        OpenAI -->|Vectors (1536-dim)| VectorDB[(ChromaDB - Local)]
    end
    
    subgraph "Reasoning Layer"
        VectorDB -->|Vectors| Clustering[DBSCAN Clustering]
        Clustering -->|Cluster Info| Agent[SRE Agent Logic]
        VectorDB -.->|Context (Runbooks)| Agent
        Agent -->|Prompt| GPT[OpenAI GPT-3.5]
        GPT -->|Recommendation| Dashboard[Streamlit Dashboard]
    end
    
    Dashboard -->|Chat & Feedback| VectorDB
```

## Prerequisites

1. **Python 3.10+**
2. **OpenAI API Key** (Set in `.env`)

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
streamlit run streamlit_app.py
# Opens browser at http://localhost:8501
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
