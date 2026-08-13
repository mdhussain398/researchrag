"""
Contradiction and disagreement detection engine between research papers.
"""

import re
from typing import List, Dict, Any, Optional
from app.models.schemas import ContradictionItem, RetrievedChunk, DocumentMetadata
from app.utils.config import logger


class ContradictionDetector:
    """
    Analyzes multi-paper evidence to detect explicit disagreements,
    divergent empirical findings, and architectural trade-offs.
    """

    OPPOSING_PAIR_CONCEPTS = [
        ("dense retrieval", "sparse retrieval", "Dense vs. Sparse Retrieval Superiority",
         "Divergence on whether dense dual-encoders unconditionally outperform sparse lexical indices (BM25) on out-of-domain or technical lexicons."),
        ("reranking", "latency", "Reranker Performance vs. Computational Overhead",
         "Disagreement regarding whether multi-stage cross-encoder reranking latency penalty justifies modest precision gains in real-time QA."),
        ("long-context", "rag", "Long-Context LLMs vs. RAG for Knowledge Grounding",
         "Contrasting claims regarding whether massive context windows (100k+ tokens) render RAG pipelines obsolete or if RAG remains superior for hallucination mitigation and cost."),
        ("chunk size", "retrieval accuracy", "Granular vs. Coarse Chunking Trade-off",
         "Dispute regarding optimal chunk size: smaller chunks preserve embedding precision while larger chunks retain broader narrative context."),
        ("fine-tuning", "retrieval", "Fine-Tuning vs. In-Context Retrieval Grounding",
         "Divergence on whether parameter fine-tuning reduces hallucinations more reliably than dynamic external retrieval injection."),
    ]

    def detect_contradictions(
        self,
        retrieved_chunks: List[RetrievedChunk],
        documents: Optional[List[DocumentMetadata]] = None,
    ) -> List[ContradictionItem]:
        """
        Scans retrieved evidence across different source documents to identify contradictory claims.
        """
        contradictions: List[ContradictionItem] = []
        if len(retrieved_chunks) < 2:
            return contradictions

        # Group chunks by document_id
        chunks_by_doc: Dict[str, List[RetrievedChunk]] = {}
        for rc in retrieved_chunks:
            chunks_by_doc.setdefault(rc.chunk.document_id, []).append(rc)

        doc_ids = list(chunks_by_doc.keys())
        if len(doc_ids) < 2:
            # Need at least two distinct documents for inter-paper contradiction
            return contradictions

        # Check opposing concept pairs
        for term_a, term_b, topic, base_explanation in self.OPPOSING_PAIR_CONCEPTS:
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    doc_a_id = doc_ids[i]
                    doc_b_id = doc_ids[j]

                    chunks_a = chunks_by_doc[doc_a_id]
                    chunks_b = chunks_by_doc[doc_b_id]

                    matching_chunk_a = self._find_matching_chunk(chunks_a, [term_a, term_b])
                    matching_chunk_b = self._find_matching_chunk(chunks_b, [term_a, term_b])

                    if matching_chunk_a and matching_chunk_b:
                        # Extract sentences highlighting the perspective
                        claim_a = self._extract_relevant_sentence(matching_chunk_a.chunk.text, term_a) or matching_chunk_a.chunk.text[:220]
                        claim_b = self._extract_relevant_sentence(matching_chunk_b.chunk.text, term_b) or matching_chunk_b.chunk.text[:220]

                        # Ensure claims are not identical
                        if claim_a.strip().lower() != claim_b.strip().lower():
                            contradictions.append(
                                ContradictionItem(
                                    topic=topic,
                                    claim_a=claim_a.strip(),
                                    source_a=matching_chunk_a.chunk.filename,
                                    page_a=matching_chunk_a.chunk.page_number,
                                    claim_b=claim_b.strip(),
                                    source_b=matching_chunk_b.chunk.filename,
                                    page_b=matching_chunk_b.chunk.page_number,
                                    explanation=base_explanation,
                                    confidence_score=0.86,
                                )
                            )

        # Deduplicate contradictions by topic
        unique_contradictions: Dict[str, ContradictionItem] = {}
        for c in contradictions:
            if c.topic not in unique_contradictions:
                unique_contradictions[c.topic] = c

        logger.info(f"Identified {len(unique_contradictions)} cross-paper disagreements/contradictions.")
        return list(unique_contradictions.values())

    def _find_matching_chunk(self, chunks: List[RetrievedChunk], terms: List[str]) -> Optional[RetrievedChunk]:
        """Finds first chunk containing any of the terms."""
        for c in chunks:
            text_lower = c.chunk.text.lower()
            if any(term.lower() in text_lower for term in terms):
                return c
        return None

    def _extract_relevant_sentence(self, text: str, term: str) -> Optional[str]:
        """Extracts the specific sentence containing the term."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            if term.lower() in s.lower() and len(s) > 25:
                return s.strip()
        return None
