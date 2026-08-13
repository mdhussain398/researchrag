"""
Data models and schemas for ResearchRAG.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timezone


def get_utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentMetadata(BaseModel):
    """Metadata extracted from a research paper or PDF document."""
    document_id: str
    filename: str
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    page_count: int = 0
    title: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    detected_sections: List[str] = Field(default_factory=list)
    sha256_hash: str = ""
    created_at: datetime = Field(default_factory=get_utc_now)


class DocumentChunk(BaseModel):
    """A granular chunk of text extracted from a research paper."""
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    section: str = "General"
    char_start: int = 0
    char_end: int = 0
    token_count_est: int = 0
    text: str
    sha256_hash: str = ""


class RetrievedChunk(BaseModel):
    """A retrieved document chunk with relevance and rerank scores."""
    chunk: DocumentChunk
    similarity_score: float = 0.0
    rerank_score: Optional[float] = None
    subtopic_matched: Optional[str] = None


class Citation(BaseModel):
    """A citation mapping a factual statement to its source document and page."""
    citation_id: int
    document_id: str
    filename: str
    page_number: int
    section: str = "General"
    source_chunk_id: Optional[str] = None
    quoted_evidence: str = ""
    is_verified: bool = True
    verification_confidence: float = 1.0
    validation_note: Optional[str] = None


class ComparisonRow(BaseModel):
    """A row in the structured paper comparison matrix."""
    paper_title: str
    filename: str
    authors: str = "N/A / Not found in provided source"
    year: str = "N/A / Not found in provided source"
    problem_statement: str = "N/A / Not found in provided source"
    methodology: str = "N/A / Not found in provided source"
    dataset_used: str = "N/A / Not found in provided source"
    evaluation_metrics: str = "N/A / Not found in provided source"
    key_results: str = "N/A / Not found in provided source"
    strengths: str = "N/A / Not found in provided source"
    limitations: str = "N/A / Not found in provided source"


class ContradictionItem(BaseModel):
    """Detected disagreement or contradiction between studies."""
    topic: str
    claim_a: str
    source_a: str
    page_a: int
    claim_b: str
    source_b: str
    page_b: int
    explanation: str
    confidence_score: float = 0.85


class ResearchGapItem(BaseModel):
    """Identified research gap or unaddressed challenge."""
    category: str  # e.g., 'Methodological', 'Empirical', 'Evaluation', 'Domain Generalization'
    description: str
    source_paper: Optional[str] = None
    gap_type: str = "Explicitly stated by authors"  # or 'Potential gap inferred from evidence'
    suggested_future_work: str = ""


class ResearchReport(BaseModel):
    """Complete structured research report."""
    report_id: str
    research_question: str
    sub_objectives: List[str] = Field(default_factory=list)
    title: str
    executive_summary: str
    background: str
    key_findings: List[str] = Field(default_factory=list)
    methodology_comparison: str
    evidence_synthesis: str
    agreements: List[str] = Field(default_factory=list)
    contradictions: List[ContradictionItem] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    research_gaps: List[ResearchGapItem] = Field(default_factory=list)
    conclusion: str
    citations: List[Citation] = Field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = Field(default_factory=list)
    comparison_table: List[ComparisonRow] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=get_utc_now)
    model_provider_used: str = "Local Synthesizer"
    raw_markdown: str = ""


class EvaluationMetrics(BaseModel):
    """Quantitative evaluation metrics for retrieval and generation."""
    query: str
    retrieval_count: int = 0
    mean_similarity_score: float = 0.0
    source_coverage_ratio: float = 0.0  # ratio of uploaded papers with retrieved chunks
    citation_validity_rate: float = 1.0  # ratio of citations that exist in source docs
    page_match_rate: float = 1.0  # ratio of citations with valid page numbers
    semantic_grounding_score: float = 0.0  # semantic overlap between claim and evidence
    hallucination_claim_rate: float = 0.0  # unsupported claim rate
    retrieval_mrr: Optional[float] = None
    retrieval_precision_at_k: Optional[float] = None
    retrieval_recall_at_k: Optional[float] = None
    execution_time_seconds: float = 0.0
    evaluation_notes: List[str] = Field(default_factory=list)


class ResearchConfig(BaseModel):
    """User configuration for research retrieval and generation."""
    research_question: str = ""
    sub_objectives: List[str] = Field(default_factory=list)
    top_k_chunks: int = 8
    similarity_threshold: float = 0.25
    enable_reranking: bool = True
    llm_provider: str = "gemini"  # "gemini", "groq", "openai", "ollama", "local"
    llm_model: str = "gemini-1.5-flash"
    temperature: float = 0.2
    max_tokens: int = 4096
    deduplication_threshold: float = 0.88
