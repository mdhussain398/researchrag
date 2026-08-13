"""
Paper comparison matrix builder and cross-study synthesizer.
"""

from typing import List, Dict, Any, Optional
import pandas as pd

from app.models.schemas import ComparisonRow, DocumentMetadata, DocumentChunk
from app.analysis.extractor import ResearchExtractor
from app.utils.config import logger


class PaperComparator:
    """Builds and formats multi-paper structured comparison matrices."""

    def __init__(self, extractor: Optional[ResearchExtractor] = None):
        self.extractor = extractor or ResearchExtractor()

    def build_comparison_matrix(
        self,
        documents: List[DocumentMetadata],
        chunks_by_doc: Dict[str, List[DocumentChunk]],
    ) -> List[ComparisonRow]:
        """Constructs a list of ComparisonRow objects for all documents."""
        rows: List[ComparisonRow] = []
        for doc in documents:
            doc_chunks = chunks_by_doc.get(doc.document_id, [])
            row = self.extractor.extract_paper_summary_row(doc, doc_chunks)
            rows.append(row)
        return rows

    def to_dataframe(self, comparison_rows: List[ComparisonRow]) -> pd.DataFrame:
        """Converts comparison rows to a pandas DataFrame with user-friendly headers."""
        data = []
        for r in comparison_rows:
            data.append({
                "Paper Title": r.paper_title,
                "Authors": r.authors,
                "Year": r.year,
                "Problem Addressed": r.problem_statement,
                "Methodology / Architecture": r.methodology,
                "Datasets Used": r.dataset_used,
                "Evaluation Metrics & Results": r.evaluation_metrics,
                "Key Strengths": r.strengths,
                "Limitations": r.limitations,
            })
        return pd.DataFrame(data)

    def to_markdown_table(self, comparison_rows: List[ComparisonRow]) -> str:
        """Generates a GitHub-flavored Markdown table for the report."""
        if not comparison_rows:
            return "_No paper comparisons available._\n"

        headers = ["Paper", "Year", "Methodology", "Datasets", "Key Findings & Metrics", "Limitations"]
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for r in comparison_rows:
            clean_title = r.paper_title.replace("|", "-")[:45] + ("..." if len(r.paper_title) > 45 else "")
            clean_year = str(r.year)
            clean_method = r.methodology.replace("|", "-")[:35]
            clean_ds = r.dataset_used.replace("|", "-")[:30]
            clean_results = (r.evaluation_metrics if r.evaluation_metrics != "N/A / Not found in provided source" else r.key_results).replace("|", "-")[:60]
            clean_lim = r.limitations.replace("|", "-")[:40]
            lines.append(f"| **{clean_title}** | {clean_year} | {clean_method} | {clean_ds} | {clean_results} | {clean_lim} |")

        return "\n".join(lines) + "\n"
