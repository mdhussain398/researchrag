"""
Orchestrator for Academic Research Report Generation and Evidence Synthesis.
"""

import re
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.models.schemas import (
    ResearchReport,
    ResearchConfig,
    RetrievedChunk,
    DocumentMetadata,
    DocumentChunk,
    ComparisonRow,
    ContradictionItem,
    ResearchGapItem,
    Citation,
)
from app.generation.llm_client import LLMClient
from app.generation.prompt_templates import build_research_prompt, ACADEMIC_SYSTEM_PROMPT
from app.generation.citation_validator import CitationValidator
from app.analysis.comparator import PaperComparator
from app.analysis.contradiction import ContradictionDetector
from app.analysis.research_gaps import ResearchGapDetector
from app.utils.config import logger


class ResearchReportGenerator:
    """
    Coordinates the full academic research report generation pipeline:
    context assembly -> LLM synthesis -> comparison table -> contradiction analysis -> gap detection -> citation validation.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        citation_validator: Optional[CitationValidator] = None,
        comparator: Optional[PaperComparator] = None,
        contradiction_detector: Optional[ContradictionDetector] = None,
        gap_detector: Optional[ResearchGapDetector] = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.citation_validator = citation_validator or CitationValidator()
        self.comparator = comparator or PaperComparator()
        self.contradiction_detector = contradiction_detector or ContradictionDetector()
        self.gap_detector = gap_detector or ResearchGapDetector()

    def generate_report(
        self,
        config: ResearchConfig,
        retrieved_chunks: List[RetrievedChunk],
        documents: List[DocumentMetadata],
        chunks_by_doc: Optional[Dict[str, List[DocumentChunk]]] = None,
    ) -> ResearchReport:
        """
        Generates a complete ResearchReport with grounded citations and multi-study analysis.
        """
        report_id = f"rep_{uuid.uuid4().hex[:10]}"
        logger.info(f"Generating research report {report_id} for topic: '{config.research_question}'")

        if not retrieved_chunks:
            # Handle empty retrieval gracefully
            empty_md = f"""# Research Report: {config.research_question}

## 1. Executive Summary
Insufficient evidence in the provided sources. Please upload relevant research PDFs and index them before generating a report.

## 2. Conclusion
Insufficient evidence in the provided sources.
"""
            return ResearchReport(
                report_id=report_id,
                research_question=config.research_question,
                sub_objectives=config.sub_objectives,
                title=f"Research Report: {config.research_question}",
                executive_summary="Insufficient evidence in the provided sources.",
                background="Insufficient evidence in the provided sources.",
                key_findings=[],
                methodology_comparison="N/A",
                evidence_synthesis="Insufficient evidence in the provided sources.",
                agreements=[],
                contradictions=[],
                limitations=[],
                research_gaps=[],
                conclusion="Insufficient evidence in the provided sources.",
                citations=[],
                retrieved_chunks=[],
                comparison_table=[],
                model_provider_used=config.llm_provider,
                raw_markdown=empty_md,
            )

        # 1. Update LLM client provider if specified in config
        if config.llm_provider != self.llm_client.provider:
            self.llm_client = LLMClient(provider=config.llm_provider, model=config.llm_model)

        # 2. Build synthesis prompt
        prompt = build_research_prompt(config, retrieved_chunks)

        # 3. Invoke LLM generation
        raw_output = self.llm_client.generate(
            prompt=prompt,
            system_prompt=ACADEMIC_SYSTEM_PROMPT,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # 4. Extract Analytical Artifacts
        chunks_dict = chunks_by_doc or {}
        comparison_matrix = self.comparator.build_comparison_matrix(documents, chunks_dict)
        contradictions = self.contradiction_detector.detect_contradictions(retrieved_chunks, documents)
        research_gaps = self.gap_detector.detect_gaps(retrieved_chunks, documents)

        # 5. Run Citation Grounding and Validation
        validated_md, citations, cite_metrics = self.citation_validator.validate_report_citations(
            raw_output, retrieved_chunks, documents
        )

        # 6. Parse sections from generated report
        title_match = re.search(r"^#\s+(?:Research Report:\s*)?(.*)$", validated_md, re.MULTILINE)
        report_title = title_match.group(1).strip() if title_match else f"Synthesis on {config.research_question}"

        exec_match = re.search(r"##\s+(?:1\.\s*)?Executive Summary\s*(.*?)(?=\n##|$)", validated_md, re.DOTALL | re.IGNORECASE)
        exec_summary = exec_match.group(1).strip() if exec_match else "Executive summary synthesized from evidence."

        bg_match = re.search(r"##\s+(?:2\.\s*)?Background.*?\s*(.*?)(?=\n##|$)", validated_md, re.DOTALL | re.IGNORECASE)
        background = bg_match.group(1).strip() if bg_match else "Background context synthesized from evidence."

        findings_match = re.search(r"##\s+(?:3\.\s*)?Key Findings\s*(.*?)(?=\n##|$)", validated_md, re.DOTALL | re.IGNORECASE)
        raw_findings = findings_match.group(1).strip() if findings_match else ""
        key_findings = [line.strip("- *").strip() for line in raw_findings.split("\n") if line.strip().startswith(("-", "*"))]
        if not key_findings and raw_findings:
            key_findings = [s.strip() for s in raw_findings.split(". ") if len(s.strip()) > 20]

        method_match = re.search(r"##\s+(?:4\.\s*)?Methodology Comparison\s*(.*?)(?=\n##|$)", validated_md, re.DOTALL | re.IGNORECASE)
        method_comp = method_match.group(1).strip() if method_match else "Methodology comparison synthesized from evidence."

        synth_match = re.search(r"##\s+(?:5\.\s*)?Evidence Synthesis.*?\s*(.*?)(?=\n##|$)", validated_md, re.DOTALL | re.IGNORECASE)
        synth = synth_match.group(1).strip() if synth_match else "Evidence synthesis synthesized from evidence."

        agree_match = re.search(r"##\s+(?:6\.\s*)?Agreements.*?\s*(.*?)(?=\n##|$)", validated_md, re.DOTALL | re.IGNORECASE)
        raw_agrees = agree_match.group(1).strip() if agree_match else ""
        agreements = [line.strip("- *").strip() for line in raw_agrees.split("\n") if line.strip().startswith(("-", "*"))]

        lim_match = re.search(r"##\s+(?:8\.\s*)?Limitations\s*(.*?)(?=\n##|$)", validated_md, re.DOTALL | re.IGNORECASE)
        raw_lims = lim_match.group(1).strip() if lim_match else ""
        limitations = [line.strip("- *").strip() for line in raw_lims.split("\n") if line.strip().startswith(("-", "*"))]

        concl_match = re.search(r"##\s+(?:10\.\s*)?Conclusion\s*(.*?)(?=\n##|$)", validated_md, re.DOTALL | re.IGNORECASE)
        conclusion = concl_match.group(1).strip() if concl_match else "Conclusion synthesized from evidence."

        return ResearchReport(
            report_id=report_id,
            research_question=config.research_question,
            sub_objectives=config.sub_objectives,
            title=report_title,
            executive_summary=exec_summary,
            background=background,
            key_findings=key_findings,
            methodology_comparison=method_comp,
            evidence_synthesis=synth,
            agreements=agreements,
            contradictions=contradictions,
            limitations=limitations,
            research_gaps=research_gaps,
            conclusion=conclusion,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            comparison_table=comparison_matrix,
            generated_at=datetime.now(timezone.utc),
            model_provider_used=config.llm_provider,
            raw_markdown=validated_md,
        )
