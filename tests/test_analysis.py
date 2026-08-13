"""
Unit tests for Research Analysis, Paper Comparison, Contradictions, and Research Gaps.
"""

import pytest
from app.models.schemas import DocumentMetadata, DocumentChunk, RetrievedChunk
from app.analysis.extractor import ResearchExtractor
from app.analysis.comparator import PaperComparator
from app.analysis.contradiction import ContradictionDetector
from app.analysis.research_gaps import ResearchGapDetector


def test_research_extractor():
    extractor = ResearchExtractor()
    sample_text = """
    We evaluate Dense Passage Retrieval on Natural Questions and TriviaQA datasets.
    DPR achieves 78.4% Top-20 retrieval accuracy and 0.768 NDCG@10, outperforming BM25.
    Our approach is limited by out-of-domain vocabulary degradation.
    """
    metrics = extractor.extract_metrics(sample_text)
    datasets = extractor.extract_datasets(sample_text)
    methods = extractor.extract_methods(sample_text)

    assert len(metrics) >= 1
    assert "Natural Questions" in datasets
    assert "Triviaqa" in datasets or "TriviaQA" in [d.upper() for d in datasets]
    assert any("Dense Passage Retrieval" in m for m in methods)


def test_paper_comparator():
    comparator = PaperComparator()
    meta = DocumentMetadata(
        document_id="doc_1",
        filename="dpr.pdf",
        title="Dense Passage Retrieval",
        authors=["V. Karpukhin", "D. Chen"],
        year=2020,
        abstract="Open-domain QA relies on efficient passage retrieval.",
    )
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            document_id="doc_1",
            filename="dpr.pdf",
            page_number=1,
            section="Results",
            text="DPR achieves 78.4% Top-20 accuracy on Natural Questions with BM25 comparison.",
        )
    ]
    rows = comparator.build_comparison_matrix([meta], {"doc_1": chunks})
    assert len(rows) == 1
    assert rows[0].paper_title == "Dense Passage Retrieval"
    assert rows[0].year == "2020"
    
    df = comparator.to_dataframe(rows)
    assert len(df) == 1
    assert "Paper Title" in df.columns

    md_table = comparator.to_markdown_table(rows)
    assert "| **Dense Passage Retrieval** |" in md_table


def test_contradiction_detector():
    detector = ContradictionDetector()
    chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id="c1",
                document_id="doc_dpr",
                filename="dpr_paper.pdf",
                page_number=1,
                text="Our findings establish that dense retrieval uniformly outperforms traditional sparse BM25 across benchmarks.",
            ),
            similarity_score=0.88,
        ),
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id="c2",
                document_id="doc_beir",
                filename="beir_paper.pdf",
                page_number=2,
                text="Contradicting earlier claims, sparse retrieval BM25 outperforms dense retrieval on zero-shot out-of-domain datasets.",
            ),
            similarity_score=0.85,
        ),
    ]

    contradictions = detector.detect_contradictions(chunks)
    assert len(contradictions) >= 1
    c = contradictions[0]
    assert c.source_a != c.source_b
    assert c.confidence_score > 0.5


def test_research_gap_detector():
    gap_detector = ResearchGapDetector()
    chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id="c1",
                document_id="doc_1",
                filename="study1.pdf",
                page_number=4,
                section="Limitations",
                text="A major drawback of our approach is the high inference latency of cross-encoder rerankers on CPU.",
            ),
            similarity_score=0.82,
        )
    ]
    gaps = gap_detector.detect_gaps(chunks)
    assert len(gaps) >= 1
    types = [g.gap_type for g in gaps]
    assert "Explicitly stated by authors" in types
