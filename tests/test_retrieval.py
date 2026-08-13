"""
Unit tests for Retrieval, Embeddings, Vector Store, and Reranker.
"""

import pytest
import numpy as np
from app.models.schemas import DocumentChunk, ResearchConfig
from app.retrieval.embeddings import EmbeddingEngine
from app.retrieval.vector_store import FaissVectorStore
from app.retrieval.query_processor import QueryProcessor
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import EvidenceRetriever


def test_embedding_engine_dimension_and_norm():
    engine = EmbeddingEngine()
    assert engine.dimension > 0
    
    vecs = engine.encode(["Dense retrieval with transformers", "BM25 sparse baseline"])
    assert vecs.shape == (2, engine.dimension)
    
    # Check L2 normalization
    norm1 = np.linalg.norm(vecs[0])
    norm2 = np.linalg.norm(vecs[1])
    assert pytest.approx(norm1, abs=1e-3) == 1.0
    assert pytest.approx(norm2, abs=1e-3) == 1.0


def test_vector_store_add_and_search(tmp_path):
    store = FaissVectorStore(index_name="test_store")
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            filename="dpr_paper.pdf",
            page_number=1,
            section="Methodology",
            text="Dense Passage Retrieval uses dual-encoders with in-batch negatives.",
        ),
        DocumentChunk(
            chunk_id="c2",
            document_id="d2",
            filename="beir_paper.pdf",
            page_number=2,
            section="Results",
            text="Cross-Encoder reranking significantly improves NDCG@10 on out-of-domain search.",
        ),
        DocumentChunk(
            chunk_id="c3",
            document_id="d3",
            filename="rag_paper.pdf",
            page_number=3,
            section="Discussion",
            text="Lost in the middle phenomenon causes long-context LLMs to fail at retrieving middle facts.",
        ),
    ]

    added = store.add_chunks(chunks)
    assert added == 3
    assert store.count() == 3

    # Search for DPR
    results = store.search(query="Dense Passage Retrieval dual-encoders", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk.chunk_id == "c1"
    assert results[0].similarity_score > 0.0

    # Test persistence
    store.save(tmp_path)
    new_store = FaissVectorStore(index_name="test_store")
    loaded = new_store.load(tmp_path)
    assert loaded is True
    assert new_store.count() == 3


def test_query_processor():
    qp = QueryProcessor()
    subqueries = qp.generate_subqueries(
        main_query="How does RAG prevent hallucinations?",
        custom_objectives=["Analyze latency overheads"]
    )
    assert len(subqueries) >= 5
    facets = [sq["facet"] for sq in subqueries]
    assert "Primary" in facets
    assert "User Objective" in facets
    assert "Methodology" in facets
    assert "Results" in facets


def test_retriever_deduplication():
    store = FaissVectorStore(index_name="dedup_store")
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="d1",
            filename="paper1.pdf",
            page_number=1,
            text="Retrieval augmented generation mitigates hallucination by providing grounded context.",
        ),
        DocumentChunk(
            chunk_id="c2",
            document_id="d1",
            filename="paper1.pdf",
            page_number=1,
            text="Retrieval augmented generation mitigates hallucination by providing grounded context snippets.",
        ),
    ]
    store.add_chunks(chunks)
    retriever = EvidenceRetriever(vector_store=store)

    config = ResearchConfig(
        research_question="How does RAG mitigate hallucination?",
        top_k_chunks=5,
        deduplication_threshold=0.75,
        enable_reranking=False,
    )
    retrieved = retriever.retrieve_evidence(config)
    # Deduplication should collapse the two nearly identical passages
    assert len(retrieved) == 1
