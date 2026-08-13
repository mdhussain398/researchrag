"""
Grounded Citation Validator and Hallucination Verification Engine.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from app.models.schemas import Citation, RetrievedChunk, DocumentMetadata
from app.utils.config import logger


class CitationValidator:
    """
    Validates, grounds, and audits citations within generated research reports.
    Prevents hallucinated references and verifies factual grounding against source chunks.
    """

    @staticmethod
    def _compute_overlap(claim: str, evidence: str) -> float:
        """Calculates token overlap ratio between claim sentence and evidence chunk, filtering common stopwords."""
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
            "by", "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "from", "up", "down", "in", "out", "of", "off",
            "over", "under", "again", "further", "then", "once", "here", "there", "when",
            "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "this", "that", "these", "those", "it", "its"
        }
        claim_words = {w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", claim.lower()) if w not in stopwords}
        evidence_words = {w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", evidence.lower()) if w not in stopwords}
        if not claim_words:
            return 1.0
        overlap = claim_words.intersection(evidence_words)
        return len(overlap) / len(claim_words)

    def validate_report_citations(
        self,
        report_text: str,
        retrieved_chunks: List[RetrievedChunk],
        documents: List[DocumentMetadata],
    ) -> Tuple[str, List[Citation], Dict[str, float]]:
        """
        Validates all citation indices in report_text against retrieved_chunks.
        Returns:
            - cleaned_report_text
            - list of Citation objects
            - metrics dict (validity_rate, page_match_rate, grounding_score, hallucination_rate)
        """
        doc_registry = {d.filename.lower(): d for d in documents}
        for d in documents:
            doc_registry[d.document_id.lower()] = d

        # Map 1-based index to retrieved chunk
        chunk_map = {i + 1: rc for i, rc in enumerate(retrieved_chunks)}

        # Separate body text from References section
        ref_split = re.split(r"##\s+(?:\d+\.\s*)?References.*", report_text, flags=re.IGNORECASE)
        body_text = ref_split[0] if ref_split else report_text

        citations_found: Dict[int, Citation] = {}
        total_citations_count = 0
        valid_citations_count = 0
        page_matches_count = 0
        grounding_scores: List[float] = []
        hallucinated_citations = 0

        # Scan each sentence in the body of the report
        sentences = re.split(r"(?<=[.!?])\s+", body_text)

        for sentence in sentences:
            brackets = re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", sentence)
            for b in brackets:
                indices = [int(idx.strip()) for idx in b.split(",") if idx.strip().isdigit()]
                for idx in indices:
                    total_citations_count += 1
                    
                    if idx in chunk_map:
                        rc = chunk_map[idx]
                        c = rc.chunk
                        
                        # Verify document exists
                        doc_exists = c.filename.lower() in doc_registry or c.document_id.lower() in doc_registry
                        # Verify page exists
                        matched_doc = doc_registry.get(c.filename.lower()) or doc_registry.get(c.document_id.lower())
                        page_valid = matched_doc is not None and (1 <= c.page_number <= matched_doc.page_count)
                        
                        # Verify grounding
                        overlap = self._compute_overlap(sentence, c.text)
                        grounding_scores.append(overlap)

                        is_verified = doc_exists and page_valid and (overlap >= 0.15)
                        confidence = round(min(1.0, 0.4 + overlap), 3)

                        if is_verified:
                            valid_citations_count += 1
                        if page_valid:
                            page_matches_count += 1
                        if not is_verified and overlap < 0.08:
                            hallucinated_citations += 1

                        if idx not in citations_found:
                            citations_found[idx] = Citation(
                                citation_id=idx,
                                document_id=c.document_id,
                                filename=c.filename,
                                page_number=c.page_number,
                                section=c.section,
                                source_chunk_id=c.chunk_id,
                                quoted_evidence=c.text[:240].strip() + ("..." if len(c.text) > 240 else ""),
                                is_verified=is_verified,
                                verification_confidence=confidence,
                                validation_note="Verified grounded citation" if is_verified else "Partial context grounding",
                            )
                        else:
                            # Update with highest confidence seen across sentences
                            if is_verified:
                                citations_found[idx].is_verified = True
                                citations_found[idx].validation_note = "Verified grounded citation"
                            if confidence > citations_found[idx].verification_confidence:
                                citations_found[idx].verification_confidence = confidence
                    else:
                        # Index out of bounds (hallucinated citation index)
                        hallucinated_citations += 1
                        if idx not in citations_found:
                            citations_found[idx] = Citation(
                                citation_id=idx,
                                document_id="unknown",
                                filename="Unknown Source",
                                page_number=0,
                                is_verified=False,
                                verification_confidence=0.0,
                                validation_note="Citation index not found in retrieved evidence context",
                            )

        # Compute summary metrics
        total = max(1, total_citations_count)
        validity_rate = round(valid_citations_count / total, 4) if total_citations_count > 0 else 1.0
        page_match_rate = round(page_matches_count / total, 4) if total_citations_count > 0 else 1.0
        avg_grounding = round(sum(grounding_scores) / len(grounding_scores), 4) if grounding_scores else 0.85
        hallucination_rate = round(hallucinated_citations / total, 4) if total_citations_count > 0 else 0.0

        metrics = {
            "total_citations": total_citations_count,
            "valid_citations": valid_citations_count,
            "validity_rate": validity_rate,
            "page_match_rate": page_match_rate,
            "semantic_grounding_score": avg_grounding,
            "hallucination_claim_rate": hallucination_rate,
        }

        # Sort citations list by ID
        citation_list = [citations_found[k] for k in sorted(citations_found.keys())]

        logger.info(
            f"Citation Validation complete: {valid_citations_count}/{total_citations_count} verified. "
            f"Validity Rate: {validity_rate * 100:.1f}%, Grounding: {avg_grounding:.3f}"
        )

        return report_text, citation_list, metrics
