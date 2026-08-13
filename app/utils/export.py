"""
Multi-format export engine for Research Reports, Paper Comparison matrices, and Evidence Ledgers.
Supports Markdown, PDF (ReportLab), and CSV.
"""

import io
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from app.models.schemas import ResearchReport, ComparisonRow, RetrievedChunk
from app.utils.config import REPORTS_DIR, logger

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        KeepTogether,
        HRFlowable,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ReportExporter:
    """Exports generated research reports to Markdown, styled PDF, and CSV formats."""

    @staticmethod
    def to_markdown(report: ResearchReport) -> str:
        """Returns the full GitHub-flavored Markdown text of the report."""
        if report.raw_markdown:
            return report.raw_markdown

        md = []
        md.append(f"# {report.title}\n")
        md.append(f"**Research Topic:** {report.research_question}\n")
        md.append(f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')} | **Model:** {report.model_provider_used}\n")
        md.append("\n---\n")

        md.append("## 1. Executive Summary\n")
        md.append(f"{report.executive_summary}\n\n")

        md.append("## 2. Background & Context\n")
        md.append(f"{report.background}\n\n")

        md.append("## 3. Key Findings\n")
        for f in report.key_findings:
            md.append(f"- {f}\n")
        md.append("\n")

        md.append("## 4. Methodology Comparison\n")
        md.append(f"{report.methodology_comparison}\n\n")

        md.append("## 5. Evidence Synthesis & Cross-Paper Analysis\n")
        md.append(f"{report.evidence_synthesis}\n\n")

        md.append("## 6. Agreements Between Studies\n")
        for a in report.agreements:
            md.append(f"- {a}\n")
        md.append("\n")

        md.append("## 7. Contradictions & Disagreements\n")
        if report.contradictions:
            for c in report.contradictions:
                md.append(f"- **{c.topic}**: {c.claim_a} *({c.source_a}, p. {c.page_a})* vs. {c.claim_b} *({c.source_b}, p. {c.page_b})*. *Explanation: {c.explanation}*\n")
        else:
            md.append("- _No direct contradictions detected in the provided evidence._\n")
        md.append("\n")

        md.append("## 8. Limitations\n")
        for lim in report.limitations:
            md.append(f"- {lim}\n")
        md.append("\n")

        md.append("## 9. Research Gaps & Future Directions\n")
        if report.research_gaps:
            for g in report.research_gaps:
                md.append(f"- **{g.category}** ({g.gap_type}): {g.description} _(Suggested Future Work: {g.suggested_future_work})_\n")
        else:
            md.append("- _No specific research gaps identified._\n")
        md.append("\n")

        md.append("## 10. Conclusion\n")
        md.append(f"{report.conclusion}\n\n")

        md.append("## 11. References & Citations\n")
        for c in report.citations:
            verify_badge = " [Verified Grounded]" if c.is_verified else " [Partial Context]"
            md.append(f"[{c.citation_id}] **{c.filename}**, p. {c.page_number} (Section: {c.section}){verify_badge}\n")
            if c.quoted_evidence:
                md.append(f"> \"{c.quoted_evidence}\"\n\n")

        return "".join(md)

    @staticmethod
    def to_pdf_bytes(report: ResearchReport) -> bytes:
        """Generates a styled, publication-ready PDF document bytes using ReportLab."""
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("ReportLab is not installed.")

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=45,
            leftMargin=45,
            topMargin=45,
            bottomMargin=45,
        )

        styles = getSampleStyleSheet()
        
        # Custom Academic Typography Styles
        title_style = ParagraphStyle(
            "AcademicTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1e293b"),
            spaceAfter=8,
        )
        meta_style = ParagraphStyle(
            "AcademicMeta",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=14,
        )
        h1_style = ParagraphStyle(
            "AcademicH1",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=14,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "AcademicBody",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=8,
        )
        bullet_style = ParagraphStyle(
            "AcademicBullet",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#334155"),
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=4,
        )
        citation_style = ParagraphStyle(
            "AcademicCitation",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11.5,
            textColor=colors.HexColor("#475569"),
            leftIndent=12,
            spaceAfter=4,
        )

        story = []

        # Header Title
        clean_title = report.title.replace("#", "").strip()
        story.append(Paragraph(clean_title, title_style))
        story.append(Paragraph(
            f"Research Question: {report.research_question} | Model: {report.model_provider_used} | Date: {report.generated_at.strftime('%Y-%m-%d')}",
            meta_style
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

        # Sections
        sections = [
            ("1. Executive Summary", report.executive_summary),
            ("2. Background & Context", report.background),
            ("3. Methodology Comparison", report.methodology_comparison),
            ("4. Evidence Synthesis", report.evidence_synthesis),
            ("5. Conclusion", report.conclusion),
        ]

        for sec_title, sec_content in sections:
            story.append(Paragraph(sec_title, h1_style))
            # Clean markdown formatting from text for ReportLab
            clean_body = sec_content.replace("**", "").replace("*", "").replace("`", "")
            for p in clean_body.split("\n\n"):
                if p.strip():
                    story.append(Paragraph(p.strip(), body_style))

        # Key Findings
        if report.key_findings:
            story.append(Paragraph("6. Key Findings", h1_style))
            for f in report.key_findings:
                clean_f = f.replace("**", "").replace("*", "")
                story.append(Paragraph(f"• {clean_f}", bullet_style))

        # Disagreements & Contradictions
        if report.contradictions:
            story.append(Paragraph("7. Contradictions & Disagreements", h1_style))
            for c in report.contradictions:
                story.append(Paragraph(f"• <b>{c.topic}</b>: {c.explanation} (Sources: {c.source_a} vs {c.source_b})", bullet_style))

        # Limitations & Gaps
        if report.limitations:
            story.append(Paragraph("8. Limitations", h1_style))
            for lim in report.limitations:
                clean_lim = lim.replace("**", "").replace("*", "")
                story.append(Paragraph(f"• {clean_lim}", bullet_style))

        # References
        if report.citations:
            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
            story.append(Paragraph("References & Grounded Citations", h1_style))
            for c in report.citations:
                story.append(Paragraph(f"[{c.citation_id}] <b>{c.filename}</b>, p. {c.page_number} (Section: {c.section})", citation_style))

        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data

    @staticmethod
    def comparison_to_csv(comparison_rows: List[ComparisonRow]) -> str:
        """Converts comparison rows into a standard CSV string."""
        data = []
        for r in comparison_rows:
            data.append({
                "Paper Title": r.paper_title,
                "Authors": r.authors,
                "Year": r.year,
                "Problem Addressed": r.problem_statement,
                "Methodology": r.methodology,
                "Datasets": r.dataset_used,
                "Evaluation Metrics & Results": r.evaluation_metrics,
                "Key Strengths": r.strengths,
                "Limitations": r.limitations,
            })
        df = pd.DataFrame(data)
        return df.to_csv(index=False)

    @staticmethod
    def evidence_to_csv(retrieved_chunks: List[RetrievedChunk]) -> str:
        """Converts retrieved evidence ledger into CSV."""
        data = []
        for i, item in enumerate(retrieved_chunks, 1):
            c = item.chunk
            data.append({
                "Index": i,
                "Document": c.filename,
                "Page": c.page_number,
                "Section": c.section,
                "Similarity Score": round(item.similarity_score, 4),
                "Rerank Score": round(item.rerank_score, 4) if item.rerank_score is not None else "N/A",
                "Subtopic": item.subtopic_matched or "Primary",
                "Evidence Text": c.text,
            })
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
