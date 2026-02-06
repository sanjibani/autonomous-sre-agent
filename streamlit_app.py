import streamlit as st
import pandas as pd
import json
from pathlib import Path
import time

from src.config import LOGS_DIR, RUNBOOKS_DIR, OLLAMA_MODEL, USE_OPENAI_EMBEDDINGS
from src.agent import get_agent, Recommendation
from src.clustering import ClusterInfo

# Page Config
st.set_page_config(
    page_title="Autonomous SRE Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Agent (Cached)
@st.cache_resource
def load_agent():
    return get_agent()

agent = load_agent()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/bot.png", width=64)
    st.title("Autonomous SRE")
    st.caption(f"Powered by {OLLAMA_MODEL} + RAG")
    
    st.divider()
    
    st.subheader("System Status")
    st.success("● Agent Online")
    if USE_OPENAI_EMBEDDINGS:
        st.info("🧠 Brain: OpenAI Embeddings")
    else:
        st.info("🧠 Brain: Local MiniLM")
        
    st.divider()
    
    st.subheader("Actions")
    if st.button("Initialize Runbooks"):
        with st.spinner("Indexing Knowledge Base..."):
            try:
                # Load runbooks
                runbooks_path = RUNBOOKS_DIR / "runbooks.json"
                if runbooks_path.exists():
                    with open(runbooks_path, "r") as f:
                        runbooks = json.load(f)
                    agent.rag_service.vectordb.add_runbooks(runbooks)
                    st.toast(f"Result: Indexed {len(runbooks)} runbooks", icon="✅")
                else:
                    st.error("runbooks.json not found!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("Clear History"):
        st.session_state.messages = []
        st.session_state.incidents = []
        st.rerun()

# State Management
if "incidents" not in st.session_state:
    st.session_state.incidents = []
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your SRE Assistant. I can analyze logs or answer questions about your infrastructure."}]

# Main Dashboard
st.title("🛡️ SRE Command Center")

# Metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Active Incidents", len(st.session_state.incidents))
with col2:
    stats = agent.rag_service.vectordb.get_collection_stats()
    st.metric("Indexed Runbooks", stats.get('runbooks', 0))
with col3:
    st.metric("Monitoring Status", "Healthy")
with col4:
    if st.button("🚨 Load Sample Logs", type="primary"):
        with st.spinner("Analyzing Logs..."):
            # Simulate log ingestion
            with open(LOGS_DIR / "hdfs_sample.log", "r") as f:
                logs = f.readlines()
            
            # Simple simulation of clustering
            error_logs = [l.strip() for l in logs if "ERROR" in l][:20]
            cluster = ClusterInfo(
                cluster_id=1,
                size=len(error_logs),
                representative_logs=error_logs,
                error_keywords=["disk", "full", "datanode", "error"],
                severity_hint="high",
                has_anomalies=True,
                anomaly_count=1
            )
            
            rec = agent.analyze_incident(cluster)
            st.session_state.incidents.append(rec)
            st.rerun()

# Tabs
tab1, tab2 = st.tabs(["🔥 Active Incidents", "💬 Chat Assistant"])

with tab1:
    if not st.session_state.incidents:
        st.info("No active incidents. System is healthy.")
        st.caption("Click 'Load Sample Logs' to simulate an outage.")
    
    for inc in st.session_state.incidents:
        with st.expander(f"[{inc.severity}] {inc.root_cause} ({inc.incident_id})", expanded=True):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"### 🔍 Root Cause Analysis\n{inc.root_cause}")
                st.markdown(f"### 🛠️ Recommendation\n{inc.recommendation}")
                if inc.retrieved_context:
                    st.markdown("### 📚 Runbook Context")
                    for ctx in inc.retrieved_context:
                        st.info(ctx)
            with c2:
                st.markdown("### 📝 Evidence")
                st.code("\n".join(inc.evidence[:5]), language="text")
                st.caption(f"Confidence: {inc.confidence * 100:.1f}%")
                
                if getattr(inc, 'has_anomalies', False):
                    st.warning(f"⚠️ **Anomaly Detected**: Found {getattr(inc, 'anomaly_count', 0)} outliers that may be the root cause.")

with tab2:
    # Chat Interface
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about runbooks (e.g., 'How do I fix disk full?'):"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use RAG to answer
                response = "I couldn't find a runbook for that."
                matches = agent.rag_service.vectordb.search_runbooks(prompt, n_results=1)
                if matches:
                    doc = matches[0]['document']
                    meta = matches[0]['metadata']
                    response = f"**Based on {meta.get('title', 'Runbook')}:**\n\n{doc}"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
