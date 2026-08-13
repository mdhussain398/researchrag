"""
Cross-Encoder Reranker for high-precision query-chunk scoring.
"""

import os
import re
from typing import List, Tuple, Optional
from app.models.schemas import RetrievedChunk
from app.utils.config import logger


class CrossEncoderReranker:
    """
    Reranks candidate chunks retrieved from vector search using a Cross-Encoder
    or reciprocal term-overlap cross-attention fallback.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._cross_encoder = None
        self._fallback_mode = False
        self._init_model()

    def _init_model(self) -> None:
        """Attempts to load CrossEncoder model, falls back to heuristic cross-scorer if unavailable."""
        try:
            from sentence_transformers import CrossEncoder
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            logger.info(f"Loading CrossEncoder: {self.model_name}...")
            self._cross_encoder = CrossEncoder(self.model_name)
            logger.info("CrossEncoder loaded successfully.")
        except Exception as e:
            logger.warning(f"Could not load CrossEncoder ({e}). Using lexical-semantic cross-scoring fallback.")
            self._fallback_mode = True

    def rerank(self, query: str, candidate_chunks: List[RetrievedChunk], top_k: int = 8) -> List[RetrievedChunk]:
        """
        Reranks a list of RetrievedChunk objects against the query.
        Returns the top_k chunks sorted by rerank_score.
        """
        if not candidate_chunks:
            return []

        if not self._fallback_mode and self._cross_encoder is not None:
            try:
                pairs = [[query, item.chunk.text] for item in candidate_chunks]
                scores = self._cross_encoder.predict(pairs)
                for item, score in zip(candidate_chunks, scores):
                    item.rerank_score = float(score)
                # Sort descending by rerank_score
                reranked = sorted(candidate_chunks, key=lambda x: x.rerank_score if x.rerank_score is not None else -999, reverse=True)
                return reranked[:top_k]
            except Exception as e:
                logger.warning(f"CrossEncoder prediction failed ({e}). Falling back to heuristic scoring.")

        # Heuristic cross-scoring fallback
        return self._fallback_rerank(query, candidate_chunks, top_k)

    def _fallback_rerank(self, query: str, candidate_chunks: List[RetrievedChunk], top_k: int) -> List[RetrievedChunk]:
        """Lexical and semantic hybrid score combining vector similarity + term match + section weight."""
        query_terms = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", query.lower()))
        
        for item in candidate_chunks:
            text_lower = item.chunk.text.lower()
            term_matches = sum(1 for term in query_terms if term in text_lower)
            lexical_ratio = (term_matches / len(query_terms)) if query_terms else 0.0
            
            # Bonus for high-information sections
            section_bonus = 0.0
            sec = item.chunk.section.lower()
            if any(s in sec for s in ["result", "method", "evaluation", "experiment", "finding"]):
                section_bonus = 0.1
            elif "limitation" in sec or "discussion" in sec:
                section_bonus = 0.08
                
            # Composite score: 60% semantic similarity + 30% exact term overlap + 10% section weight
            heuristic_score = (0.6 * item.similarity_score) + (0.3 * lexical_ratio) + section_bonus
            item.rerank_score = round(heuristic_score, 4)

        reranked = sorted(candidate_chunks, key=lambda x: x.rerank_score or 0.0, reverse=True)
        return reranked[:top_k]
