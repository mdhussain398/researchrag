"""
Unit tests for Report Generation, Citation Validation, and Grounding Audits.
"""

import pytest
from app.models.schemas import ResearchConfig, RetrievedChunk, DocumentChunk, DocumentMetadata
from app.generation.citation_validator import CitationValidator
from app.generation.report_generator import ResearchReportGenerator
from app.generation.llm_client import LLMClient


def test_citation_validator():
    validator = CitationValidator()
    docs = [
        DocumentMetadata(
            document_id="doc_1",
            filename="dpr_paper.pdf",
            page_count=6,
            title="DPR Paper",
        )
    ]
    chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id="c1",
                document_id="doc_1",
                filename="dpr_paper.pdf",
                page_number=4,
                section="Results",
                text="DPR achieves 78.4% Top-20 retrieval accuracy on Natural Questions.",
            ),
            similarity_score=0.92,
        )
    ]

    report_text = "Empirical evaluations show DPR achieves 78.4% Top-20 retrieval accuracy on Natural Questions [1]."
    cleaned, citations, metrics = validator.validate_report_citations(report_text, chunks, docs)

    assert len(citations) == 1
    assert citations[0].is_verified is True
    assert citations[0].filename == "dpr_paper.pdf"
    assert citations[0].page_number == 4
    assert metrics["validity_rate"] == 1.0
    assert metrics["hallucination_claim_rate"] == 0.0


def test_citation_validator_hallucinated_index():
    validator = CitationValidator()
    docs = [DocumentMetadata(document_id="doc_1", filename="dpr.pdf", page_count=5)]
    chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id="c1",
                document_id="doc_1",
                filename="dpr.pdf",
                page_number=1,
                text="Dense passage retrieval.",
            ),
            similarity_score=0.8,
        )
    ]
    # Citation [99] does not exist in retrieved chunks
    report_text = "DPR is scalable [99]."
    cleaned, citations, metrics = validator.validate_report_citations(report_text, chunks, docs)

    assert len(citations) == 1
    assert citations[0].is_verified is False
    assert metrics["hallucination_claim_rate"] == 1.0


def test_research_report_generator_offline():
    generator = ResearchReportGenerator()
    docs = [
        DocumentMetadata(
            document_id="doc_1",
            filename="rag_study.pdf",
            page_count=4,
            title="RAG Empirical Study",
        )
    ]
    chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id="c1",
                document_id="doc_1",
                filename="rag_study.pdf",
                page_number=2,
                section="Results",
                text="RAG reduces hallucination rates by 3.4x compared to 128k long-context LLMs on multi-document QA.",
            ),
            similarity_score=0.91,
        )
    ]
    config = ResearchConfig(
        research_question="How does RAG compare to Long-Context LLMs?",
        llm_provider="local",
    )

    report = generator.generate_report(config, chunks, docs)

    assert report.title != ""
    assert report.executive_summary != ""
    assert report.background != ""
    assert len(report.key_findings) >= 1
    assert report.conclusion != ""
    assert len(report.citations) >= 1
    assert report.citations[0].is_verified is True
