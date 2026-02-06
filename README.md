# Autonomous SRE Agent

An intelligent agent that ingests raw logs, clusters them into incidents, investigates root causes using RAG, and drafts fixes for human approval.

## Features

- **Log Ingestion & Clustering**: Uses DBSCAN to group similar log entries
- **RAG-based Investigation**: Searches runbooks and past incidents for context
- **Human-in-the-Loop Learning**: Learns from user feedback to improve over time
- **Web Dashboard**: View alerts, provide feedback, and monitor incidents

## Tech Stack

- **Embeddings**: **OpenAI `text-embedding-3-small`** (Cloud, 1536-dim)
- **Vector DB**: ChromaDB (Time-partitioned, local persistence)
- **LLM**: **OpenAI `gpt-3.5-turbo`** (Cloud)
- **Web Framework**: **Streamlit** (Python-only UI)
- **Clustering**: DBSCAN (scikit-learn)
- **Streaming**: Kafka (Optional, real-time ingestion)
- **Caching**: Redis (Optional, 40-60% fewer DB queries)

## Architecture

```mermaid
graph TD
    Logs[Log Sources] -->|Stream| Kafka[Kafka Topic]
    Kafka -->|Consume| Consumer[Kafka Consumer]
    Consumer -->|Batch| Embed[Embedding Service]
    
    subgraph "Caching Layer"
        Embed -->|Check| Redis[(Redis Cache)]
        Redis -->|Miss| OpenAI[OpenAI Embeddings]
        OpenAI -->|Store| Redis
    end
    
    subgraph "Storage Layer - Time Partitioned"
        Embed -->|Write| Today[(logs_2026_02_06)]
        Embed -->|Write| Yesterday[(logs_2026_02_05)]
        Embed -->|Write| Older[(logs_2026_02_...)]
    end
    
    subgraph "Reasoning Layer"
        Today -->|Query| Clustering[DBSCAN Clustering]
        Clustering -->|Cluster Info| Agent[SRE Agent]
        Agent -->|RAG| Runbooks[(Runbooks Collection)]
        Agent -->|Prompt| GPT[OpenAI GPT-3.5]
        GPT -->|Recommendation| Dashboard[Streamlit Dashboard]
    end
    
    Dashboard -->|Feedback| Agent
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
│   ├── logs/              # Log files to analyze
│   └── runbooks/          # Runbooks for RAG
├── src/
│   ├── embeddings.py      # OpenAI/Local embedding service
│   ├── clustering.py      # DBSCAN clustering
│   ├── vectordb.py        # ChromaDB with caching + partitioning
│   ├── cache.py           # Redis caching layer
│   ├── kafka_consumer.py  # Kafka streaming consumer
│   ├── rag.py             # RAG retrieval logic
│   └── agent.py           # Main agent logic
├── evals/                 # Ragas/DeepEval benchmarks
├── streamlit_app.py       # Main Streamlit application
└── requirements.txt
```

## Data Source

Uses [Loghub](https://github.com/logpai/loghub) HDFS dataset for real production logs.

## License

MIT
