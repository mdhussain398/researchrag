"""
ResearchRAG: Autonomous AI Research Report Generator
Main Streamlit Application Entrypoint.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from app.utils.config import logger, SAMPLE_DIR
from app.models.schemas import ResearchConfig
from app.ingestion.manager import IngestionManager
from app.retrieval.vector_store import FaissVectorStore
from app.retrieval.retriever import EvidenceRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.generation.report_generator import ResearchReportGenerator
from app.evaluation.evaluator import ResearchEvaluator
from app.frontend.styles import CUSTOM_CSS
from app.frontend.views import (
    render_home,
    render_documents,
    render_setup,
    render_generate_report,
    render_evidence_explorer,
    render_paper_comparison,
    render_evaluation,
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="ResearchRAG — AI Research Report Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply CSS Design System
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_ingestion_manager():
    """Initializes and caches the IngestionManager singleton."""
    return IngestionManager()


@st.cache_resource
def get_vector_store():
    """Initializes and caches the FaissVectorStore singleton."""
    vs = FaissVectorStore(index_name="research_store")
    vs.load()
    return vs


@st.cache_resource
def get_reranker():
    """Initializes and caches the CrossEncoderReranker singleton."""
    return CrossEncoderReranker()


@st.cache_resource
def get_retriever():
    """Initializes and caches the EvidenceRetriever singleton."""
    vs = get_vector_store()
    reranker = get_reranker()
    return EvidenceRetriever(vector_store=vs, reranker=reranker)


@st.cache_resource
def get_report_generator():
    """Initializes and caches the ResearchReportGenerator singleton."""
    return ResearchReportGenerator()


@st.cache_resource
def get_evaluator():
    """Initializes and caches the ResearchEvaluator singleton."""
    return ResearchEvaluator()


def init_session_state():
    """Initializes Streamlit session state variables."""
    if "config" not in st.session_state:
        st.session_state["config"] = ResearchConfig(
            research_question="How does dense passage retrieval compare with traditional sparse BM25 retrieval for open-domain QA?",
            sub_objectives=[
                "Methodology and dual-encoder architecture",
                "Evaluation benchmarks and retrieval accuracy metrics",
                "Generalization limitations and computational trade-offs",
            ],
            top_k_chunks=8,
            similarity_threshold=0.15,
            enable_reranking=True,
            llm_provider="gemini",
        )
    if "current_report" not in st.session_state:
        st.session_state["current_report"] = None
    if "current_metrics" not in st.session_state:
        st.session_state["current_metrics"] = None
    if "benchmark_summary" not in st.session_state:
        st.session_state["benchmark_summary"] = None


def main():
    init_session_state()

    ingestion_mgr = get_ingestion_manager()
    vector_store = get_vector_store()
    retriever = get_retriever()
    report_gen = get_report_generator()
    evaluator = get_evaluator()

    # Sidebar Navigation & System Status
    st.sidebar.markdown("# 🔬 ResearchRAG")
    st.sidebar.caption("Autonomous AI Research Synthesis & RAG Engine")
    st.sidebar.markdown("---")

    nav_options = [
        "🏠 Home",
        "📄 Documents",
        "⚙️ Research Setup",
        "📊 Generate Report",
        "🔍 Evidence Explorer",
        "📑 Paper Comparison",
        "📈 Evaluation",
    ]

    selected_view = st.sidebar.radio(
        "Navigation",
        options=nav_options,
        index=0,
        label_visibility="collapsed",
    )

    # Corpus Status in Sidebar
    docs_count = len(ingestion_mgr.get_all_documents())
    chunks_count = len(ingestion_mgr.get_all_chunks())
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📦 Knowledge Base Status")
    st.sidebar.markdown(f"- **Papers Loaded**: `{docs_count}`")
    st.sidebar.markdown(f"- **Indexed Chunks**: `{chunks_count}`")
    st.sidebar.markdown(f"- **FAISS Index**: `{'Active' if vector_store.count() > 0 else 'Empty'}`")

    current_cfg: ResearchConfig = st.session_state["config"]
    st.sidebar.markdown(f"- **Current LLM**: `{current_cfg.llm_provider}`")

    st.sidebar.markdown("---")

    # Route Views
    if selected_view == "🏠 Home":
        render_home()
    elif selected_view == "📄 Documents":
        render_documents(ingestion_mgr, vector_store)
    elif selected_view == "⚙️ Research Setup":
        render_setup(st.session_state)
    elif selected_view == "📊 Generate Report":
        render_generate_report(st.session_state, ingestion_mgr, retriever, report_gen, evaluator)
    elif selected_view == "🔍 Evidence Explorer":
        render_evidence_explorer(st.session_state)
    elif selected_view == "📑 Paper Comparison":
        render_paper_comparison(st.session_state, ingestion_mgr)
    elif selected_view == "📈 Evaluation":
        render_evaluation(st.session_state, evaluator)


if __name__ == "__main__":
    main()
