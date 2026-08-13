"""
Unit tests for Quantitative Evaluation and Multi-Format Export (Markdown, PDF, CSV).
"""

import pytest
from app.models.schemas import ResearchReport, Citation, ComparisonRow, RetrievedChunk, DocumentChunk, DocumentMetadata
from app.evaluation.evaluator import ResearchEvaluator
from app.utils.export import ReportExporter


def test_research_evaluator():
    evaluator = ResearchEvaluator()
    docs = [DocumentMetadata(document_id="d1", filename="paper1.pdf", page_count=5)]
    chunks = [
        RetrievedChunk(
            chunk=DocumentChunk(
                chunk_id="c1",
                document_id="d1",
                filename="paper1.pdf",
                page_number=1,
                text="Evidence text.",
            ),
            similarity_score=0.88,
        )
    ]
    report = ResearchReport(
        report_id="rep_1",
        research_question="What are RAG advantages?",
        title="RAG Synthesis",
        executive_summary="Executive summary.",
        background="Background text.",
        key_findings=["Finding 1"],
        methodology_comparison="Methodology text.",
        evidence_synthesis="Synthesis text.",
        conclusion="Conclusion text.",
        citations=[
            Citation(
                citation_id=1,
                document_id="d1",
                filename="paper1.pdf",
                page_number=1,
                is_verified=True,
                verification_confidence=0.95,
            )
        ],
        retrieved_chunks=chunks,
    )

    metrics = evaluator.evaluate_report(report, docs, execution_time=1.2)
    assert metrics.retrieval_count == 1
    assert metrics.source_coverage_ratio == 1.0
    assert metrics.citation_validity_rate == 1.0
    assert metrics.semantic_grounding_score > 0.8
    assert metrics.hallucination_claim_rate == 0.0


def test_export_markdown_pdf_csv():
    report = ResearchReport(
        report_id="rep_test",
        research_question="Comparative RAG Evaluation",
        title="State-of-the-art in RAG",
        executive_summary="Summary text.",
        background="Background text.",
        key_findings=["Key finding A"],
        methodology_comparison="Dual encoders with FAISS.",
        evidence_synthesis="Cross-study evaluation.",
        conclusion="Final conclusion.",
        citations=[
            Citation(
                citation_id=1,
                document_id="d1",
                filename="study1.pdf",
                page_number=2,
                section="Results",
                is_verified=True,
            )
        ],
        comparison_table=[
            ComparisonRow(
                paper_title="DPR Paper",
                filename="dpr.pdf",
                authors="Karpukhin et al.",
                year="2020",
                methodology="Dual Encoder",
            )
        ],
        retrieved_chunks=[
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="c1",
                    document_id="d1",
                    filename="study1.pdf",
                    page_number=2,
                    text="Sample evidence text.",
                ),
                similarity_score=0.89,
            )
        ],
    )

    # 1. Markdown Export
    md = ReportExporter.to_markdown(report)
    assert "# State-of-the-art in RAG" in md
    assert "[1] **study1.pdf**" in md

    # 2. PDF Export
    pdf_bytes = ReportExporter.to_pdf_bytes(report)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")

    # 3. CSV Exports
    matrix_csv = ReportExporter.to_csv_bytes(report.comparison_table) if hasattr(ReportExporter, "to_csv_bytes") else ReportExporter.comparison_to_csv(report.comparison_table)
    assert "DPR Paper" in matrix_csv
    assert "Karpukhin et al." in matrix_csv

    evidence_csv = ReportExporter.evidence_to_csv(report.retrieved_chunks)
    assert "study1.pdf" in evidence_csv
