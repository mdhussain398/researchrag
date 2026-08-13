"""
Structured evidence extraction engine across research paper chunks.
"""

import re
from typing import List, Dict, Any, Optional
from app.models.schemas import DocumentChunk, RetrievedChunk, DocumentMetadata, ComparisonRow


class ResearchExtractor:
    """Extracts structured research dimensions (methods, datasets, metrics, results, limitations)."""

    METRIC_PATTERNS = [
        r"\b(?:accuracy|acc|f1|mrr|ndcg|bleu|rouge|hit|recall|precision)\b[:\s=]*([\d\.]+%?)",
        r"([\d\.]+%?)\s+(?:top-\d+|accuracy|acc|f1|mrr|ndcg|exact match|em|improvement|gain|increase|reduction)",
        r"\b(?:ndcg(?:@\d+)?|mrr(?:@\d+)?|f1|hit(?:@\d+)?)\s*(?:of|is|achieves|reaches)?\s*[:\s=]*([\d\.]+)",
        r"\b(?:latency|throughput)\b[:\s=]*([\d\.]+\s*(?:ms|s|tokens/s|queries/sec)?)",
    ]

    DATASET_KEYWORDS = [
        "natural questions", "triviaqa", "ms marco", "ms-marco", "hotpotqa",
        "squad", "bioasq", "pubmed", "fever", "beir", "coqa", "drop",
        "mmlu", "gsm8k", "humaneval", "synthetic", "custom benchmark"
    ]

    METHOD_KEYWORDS = [
        "dense passage retrieval", "dpr", "bm25", "hybrid retrieval", "sparse-dense",
        "colbert", "reranking", "cross-encoder", "bi-encoder", "in-context learning",
        "vector database", "faiss", "reciprocal rank fusion", "hierarchical chunking",
        "fine-tuning", "lora", "retrieval-augmented", "rag"
    ]

    def extract_metrics(self, text: str) -> List[str]:
        """Extracts quantitative metric statements from text."""
        findings = []
        for pattern in self.METRIC_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for m in matches:
                start = max(0, m.start() - 35)
                end = min(len(text), m.end() + 35)
                snippet = text[start:end].strip().replace("\n", " ")
                findings.append(snippet)
        return list(dict.fromkeys(findings))[:3]

    def extract_datasets(self, text: str) -> List[str]:
        """Identifies datasets referenced in the text."""
        found = []
        lower = text.lower()
        for ds in self.DATASET_KEYWORDS:
            if ds in lower:
                found.append(ds.title())
        return list(dict.fromkeys(found))

    def extract_methods(self, text: str) -> List[str]:
        """Identifies methodologies and architectures mentioned in text."""
        found = []
        lower = text.lower()
        for method in self.METHOD_KEYWORDS:
            if method in lower:
                found.append(method.title())
        return list(dict.fromkeys(found))

    def extract_paper_summary_row(
        self, metadata: DocumentMetadata, chunks: List[DocumentChunk]
    ) -> ComparisonRow:
        """Constructs a structured ComparisonRow for a single paper based on its chunks and metadata."""
        paper_text = " ".join([c.text for c in chunks])
        lower_full = paper_text.lower()

        # Authors & Year
        authors_str = ", ".join(metadata.authors) if metadata.authors else "N/A / Not found in provided source"
        year_str = str(metadata.year) if metadata.year else "N/A / Not found in provided source"

        # Problem Statement
        problem = "N/A / Not found in provided source"
        if metadata.abstract:
            # First 1-2 sentences of abstract usually state the problem
            sentences = [s.strip() for s in metadata.abstract.split(". ") if s.strip()]
            if sentences:
                problem = sentences[0] + ("." if not sentences[0].endswith(".") else "")

        # Methodology
        methods = self.extract_methods(paper_text)
        method_str = ", ".join(methods) if methods else "N/A / Not found in provided source"

        # Datasets
        datasets = self.extract_datasets(paper_text)
        dataset_str = ", ".join(datasets) if datasets else "N/A / Not found in provided source"

        # Evaluation Metrics & Results
        metrics = self.extract_metrics(paper_text)
        metric_str = "; ".join(metrics) if metrics else "N/A / Not found in provided source"

        # Strengths & Limitations from section chunks
        strengths = []
        limitations = []

        for c in chunks:
            sec_lower = c.section.lower()
            if any(k in sec_lower for k in ["limitation", "threat", "drawback", "failure"]):
                limitations.append(c.text[:200].strip())
            elif any(k in sec_lower for k in ["result", "conclusion", "contribution", "strength"]):
                strengths.append(c.text[:200].strip())

        strength_str = "; ".join(strengths[:2]) if strengths else "N/A / Not found in provided source"
        limitation_str = "; ".join(limitations[:2]) if limitations else "N/A / Not found in provided source"

        return ComparisonRow(
            paper_title=metadata.title or metadata.filename,
            filename=metadata.filename,
            authors=authors_str,
            year=year_str,
            problem_statement=problem,
            methodology=method_str,
            dataset_used=dataset_str,
            evaluation_metrics=metric_str,
            key_results=strength_str if strength_str != "N/A / Not found in provided source" else "See full report synthesis",
            strengths=strength_str,
            limitations=limitation_str,
        )
