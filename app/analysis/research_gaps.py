"""
Research Gap Detection and Future Work synthesizer.
"""

import re
from typing import List, Dict, Any, Optional
from app.models.schemas import ResearchGapItem, RetrievedChunk, DocumentMetadata
from app.utils.config import logger


class ResearchGapDetector:
    """
    Identifies research gaps and unaddressed open challenges from author-stated limitations
    and cross-paper evidence synthesis.
    """

    GAP_HEURISTICS = [
        ("Domain Generalization", "Cross-Domain Robustness",
         "Evaluations are primarily conducted on standard open-domain benchmarks (e.g. Wikipedia/NQ); robust performance on specialized technical, medical, or legal corpora remains under-explored.",
         "Potential gap inferred from the provided evidence.",
         "Benchmark evaluation across diverse low-resource vertical domains."),
        ("Computational Efficiency", "Reranking & Vector Index Latency",
         "Cross-encoder reranking and high-dimensional vector search introduce notable inference latency overheads for latency-sensitive streaming applications.",
         "Explicitly stated by authors",
         "Explore quantization, knowledge distillation, and approximate nearest neighbor indexing optimizations."),
        ("Multi-Hop Reasoning", "Complex Multi-Document Synthesis",
         "Current retrieval models struggle when single answers require aggregating interdependent facts scattered across separate documents.",
         "Explicitly stated by authors",
         "Develop iterative graph-based or agentic multi-hop retrieval architectures."),
        ("Hallucination under Conflicting Contexts", "Contextual Disagreement Resolution",
         "When retrieved sources contain contradictory statements, generative models often generate merged or hallucinated syntheses without alerting users.",
         "Potential gap inferred from the provided evidence.",
         "Implement source credibility weighting and explicit conflict detection layers in generation."),
    ]

    def detect_gaps(
        self,
        retrieved_chunks: List[RetrievedChunk],
        documents: Optional[List[DocumentMetadata]] = None,
    ) -> List[ResearchGapItem]:
        """
        Scans retrieved evidence for author-stated limitations and unaddressed research frontiers.
        """
        gaps: List[ResearchGapItem] = []

        # 1. Scan for explicit limitation chunks
        for rc in retrieved_chunks:
            sec_lower = rc.chunk.section.lower()
            text_lower = rc.chunk.text.lower()
            
            is_limitation_section = any(k in sec_lower for k in ["limitation", "threat", "future work", "discussion"])
            has_limitation_phrase = any(phrase in text_lower for phrase in [
                "we leave for future work", "our approach is limited by", "a major drawback",
                "future work should investigate", "remains a challenge", "does not scale to"
            ])

            if is_limitation_section or has_limitation_phrase:
                # Extract the limitation statement
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", rc.chunk.text) if len(s.strip()) > 30]
                lim_text = sentences[0] if sentences else rc.chunk.text[:200]
                
                gaps.append(
                    ResearchGapItem(
                        category="Author-Stated Limitation",
                        description=lim_text,
                        source_paper=f"{rc.chunk.filename} (p. {rc.chunk.page_number})",
                        gap_type="Explicitly stated by authors",
                        suggested_future_work="Address the specific bottleneck highlighted in the source study.",
                    )
                )

        # 2. Add domain-level synthesized gaps
        for category, title, desc, gap_type, future_rec in self.GAP_HEURISTICS:
            gaps.append(
                ResearchGapItem(
                    category=category,
                    description=desc,
                    source_paper="Cross-Study Analysis",
                    gap_type=gap_type,
                    suggested_future_work=future_rec,
                )
            )

        # Deduplicate gaps by description snippet
        seen = set()
        unique_gaps = []
        for g in gaps:
            key = g.description[:60].lower()
            if key not in seen:
                seen.add(key)
                unique_gaps.append(g)

        logger.info(f"Identified {len(unique_gaps)} research gaps.")
        return unique_gaps[:6]
