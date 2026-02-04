# CLAUDE.md - Project Rules for Autonomous SRE Agent

## Development Rules

1. **Plan Before Code**: Before writing any code, describe your approach and wait for approval. Always ask clarifying questions before writing any code if requirements are ambiguous.

2. **Small Task Scope**: If a task requires changes to more than 3 files, stop and break it into smaller tasks first.

3. **Post-Code Analysis**: After writing code, list what could break and suggest tests to cover it.

4. **Test-Driven Bug Fixes**: When there's a bug, start by writing a test that reproduces it, then fix it until the test passes.

5. **Continuous Learning**: Every time the user corrects me, add a new rule to this CLAUDE.md file so it never happens again.

---

## Project Overview

**Autonomous SRE Agent** - An intelligent system that:
- Ingests raw logs and clusters them into incidents
- Investigates root causes using RAG (Retrieval Augmented Generation)
- Drafts human-readable fixes for approval
- Learns from human feedback (HITL)

## Tech Stack

- **Language**: Python
- **Vector Database**: ChromaDB
- **Embeddings**: all-MiniLM-L6-v2 (CPU-friendly)
- **Clustering**: DBSCAN
- **Interface**: Web Dashboard (Flask/FastAPI)
- **Data Source**: Loghub (HDFS dataset)

## Architecture

```
Logs → Embedding → DBSCAN Clustering → Agent Analysis → RAG Lookup → Recommendation → Human Feedback → Learning Loop
```
