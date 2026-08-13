"""
Unit tests for Document Ingestion, PDF Parsing, Cleaning, and Section-Aware Chunking.
"""

import pytest
from pathlib import Path
from app.ingestion.cleaner import TextCleaner
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import AcademicChunker
from app.ingestion.manager import IngestionManager
from app.models.schemas import DocumentMetadata
from app.utils.config import SAMPLE_DIR


def test_cleaner_ligatures_and_hyphenation():
    raw = "The transfor-\nmers architecture achieves ﬁne-grained visual-linguistic alignment."
    cleaned = TextCleaner.clean(raw)
    assert "transformers" in cleaned
    assert "fine-grained" in cleaned


def test_cleaner_headers_and_whitespace():
    raw = "Page 1 of 12\narXiv:2301.12345v1 [cs.CL] 15 Jan 2023\n\n\n\nIntroduction to RAG\n\n\nSystems."
    cleaned = TextCleaner.clean(raw)
    assert "Page 1 of 12" not in cleaned
    assert "arXiv:2301.12345" not in cleaned
    assert "Introduction to RAG\n\nSystems." in cleaned


def test_pdf_parser_sample_paper():
    parser = PDFParser()
    sample_pdf = list(SAMPLE_DIR.glob("*.pdf"))[0]
    metadata, pages = parser.parse_pdf(str(sample_pdf))

    assert metadata.page_count >= 1
    assert metadata.title is not None
    assert len(metadata.detected_sections) >= 3
    assert metadata.sha256_hash != ""
    assert len(pages) == metadata.page_count
    assert len(pages[0]["cleaned_text"]) > 50


def test_pdf_parser_invalid_file(tmp_path):
    parser = PDFParser()
    empty_file = tmp_path / "empty.pdf"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError):
        parser.parse_pdf(str(empty_file))

    non_existent = tmp_path / "does_not_exist.pdf"
    with pytest.raises(FileNotFoundError):
        parser.parse_pdf(str(non_existent))


def test_academic_chunker():
    chunker = AcademicChunker(target_chunk_size=300, chunk_overlap=50, min_chunk_size=50)
    meta = DocumentMetadata(
        document_id="doc_test123",
        filename="test_paper.pdf",
        page_count=1,
        title="Test Paper",
        detected_sections=["Introduction", "Methodology"],
        sha256_hash="abc123hash",
    )
    pages_data = [{
        "page_number": 1,
        "raw_text": "Introduction\n\nRetrieval-Augmented Generation combines retrieval models with generative LLMs. This architecture effectively reduces hallucinations. Furthermore, experimental evaluations demonstrate strong gains on QA benchmarks.\n\nMethodology\n\nWe utilize dense passage representations and FAISS indexing. Inner product similarity facilitates fast MIPS search.",
        "cleaned_text": "Introduction\n\nRetrieval-Augmented Generation combines retrieval models with generative LLMs. This architecture effectively reduces hallucinations. Furthermore, experimental evaluations demonstrate strong gains on QA benchmarks.\n\nMethodology\n\nWe utilize dense passage representations and FAISS indexing. Inner product similarity facilitates fast MIPS search.",
        "sections": ["Introduction", "Methodology"],
    }]

    chunks = chunker.chunk_document(meta, pages_data)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.document_id == "doc_test123"
        assert c.filename == "test_paper.pdf"
        assert c.page_number == 1
        assert len(c.text) >= 50
        assert c.token_count_est > 0


def test_ingestion_manager_caching(tmp_path):
    mgr = IngestionManager()
    sample_pdf = list(SAMPLE_DIR.glob("*.pdf"))[0]
    
    # Process first time
    meta1, chunks1 = mgr.process_file(str(sample_pdf))
    assert meta1.document_id in mgr.documents
    assert len(chunks1) > 0

    # Process second time (should hit cache)
    meta2, chunks2 = mgr.process_file(str(sample_pdf), force_reprocess=False)
    assert meta1.sha256_hash == meta2.sha256_hash
    assert len(chunks1) == len(chunks2)
