"""
Academic evidence retriever coordinating multi-aspect semantic search, deduplication, and reranking.
"""

import re
from typing import List, Dict, Any, Optional
import numpy as np

from app.models.schemas import RetrievedChunk, ResearchConfig
from app.retrieval.vector_store import FaissVectorStore
from app.retrieval.query_processor import QueryProcessor
from app.retrieval.reranker import CrossEncoderReranker
from app.utils.config import logger


class EvidenceRetriever:
    """
    Retriever that conducts faceted research queries, deduplicates near-identical passages,
    and reranks evidence for grounded synthesis.
    """

    def __init__(
        self,
        vector_store: FaissVectorStore,
        reranker: Optional[CrossEncoderReranker] = None,
        query_processor: Optional[QueryProcessor] = None,
    ):
        self.vector_store = vector_store
        self.reranker = reranker or CrossEncoderReranker()
        self.query_processor = query_processor or QueryProcessor()

    @staticmethod
    def _compute_jaccard_similarity(text_a: str, text_b: str) -> float:
        """Computes word-level Jaccard similarity between two texts for rapid deduplication."""
        words_a = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", text_a.lower()))
        words_b = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a.intersection(words_b))
        union = len(words_a.union(words_b))
        return intersection / union if union > 0 else 0.0

    def _deduplicate_chunks(
        self, candidate_chunks: List[RetrievedChunk], threshold: float = 0.80
    ) -> List[RetrievedChunk]:
        """Filters out near-duplicate chunks based on text similarity."""
        unique_chunks: List[RetrievedChunk] = []
        for cand in candidate_chunks:
            is_dup = False
            for accepted in unique_chunks:
                # Same doc and adjacent page or high word overlap
                sim = self._compute_jaccard_similarity(cand.chunk.text, accepted.chunk.text)
                if sim >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique_chunks.append(cand)
        return unique_chunks

    def retrieve_evidence(
        self,
        config: ResearchConfig,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Executes faceted evidence retrieval for a research question.
        Returns deduplicated, reranked, high-relevance chunks.
        """
        if self.vector_store.count() == 0:
            logger.warning("Vector store is empty. No evidence can be retrieved.")
            return []

        # 1. Expand query and generate subtopic facets
        subqueries = self.query_processor.generate_subqueries(
            config.research_question, config.sub_objectives
        )

        all_candidates: Dict[str, RetrievedChunk] = {}

        # 2. Retrieve candidates for each facet
        per_query_k = max(4, config.top_k_chunks // len(subqueries) + 3)
        for sq in subqueries:
            q_text = sq["query"]
            facet = sq["facet"]
            matches = self.vector_store.search(
                query=q_text,
                top_k=per_query_k,
                min_similarity=config.similarity_threshold,
                filter_doc_ids=filter_doc_ids,
            )
            for m in matches:
                m.subtopic_matched = facet
                cid = m.chunk.chunk_id
                if cid not in all_candidates or m.similarity_score > all_candidates[cid].similarity_score:
                    all_candidates[cid] = m

        candidate_list = list(all_candidates.values())
        logger.info(f"Retrieved {len(candidate_list)} raw candidates across {len(subqueries)} query facets.")

        if not candidate_list:
            # If no matches above threshold, retry primary query with lowered threshold
            fallback_matches = self.vector_store.search(
                query=config.research_question,
                top_k=config.top_k_chunks,
                min_similarity=0.05,
                filter_doc_ids=filter_doc_ids,
            )
            candidate_list = fallback_matches

        # 3. Deduplicate
        deduped = self._deduplicate_chunks(candidate_list, threshold=config.deduplication_threshold)
        logger.info(f"Deduplicated to {len(deduped)} distinct candidate chunks.")

        # 4. Rerank if enabled
        if config.enable_reranking and deduped:
            final_results = self.reranker.rerank(
                query=config.research_question,
                candidate_chunks=deduped,
                top_k=config.top_k_chunks,
            )
        else:
            # Sort by similarity score
            sorted_chunks = sorted(deduped, key=lambda x: x.similarity_score, reverse=True)
            final_results = sorted_chunks[:config.top_k_chunks]

        logger.info(f"Final retrieved evidence pool: {len(final_results)} chunks.")
        return final_results
