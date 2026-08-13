"""
Query analysis, subtopic facet decomposition, and academic query expansion.
"""

import re
from typing import List, Dict, Any


class QueryProcessor:
    """
    Decomposes a primary research question into targeted sub-queries
    to ensure high multi-aspect recall across methodology, results, limitations, and comparisons.
    """

    SUBTOPIC_TEMPLATES = [
        ("{query} methodology algorithm framework architecture approach", "Methodology"),
        ("{query} experimental results benchmark performance metrics datasets", "Results"),
        ("{query} limitations weaknesses drawbacks computational trade-offs failure modes", "Limitations"),
        ("{query} comparison baseline state of the art vs disagreement contradiction", "Comparison"),
    ]

    EXPANSION_KEYWORDS = {
        "rag": ["retrieval-augmented generation", "dense retrieval", "vector search", "knowledge injection"],
        "retrieval": ["dense passage retrieval", "sparse retrieval", "BM25", "semantic search"],
        "hallucination": ["factual inconsistency", "groundedness", "faithfulness", "unsupported generation"],
        "evaluation": ["benchmarking", "ground truth", "exact match", "F1 score", "human evaluation"],
        "transformer": ["self-attention", "multi-head attention", "encoder-decoder", "large language model"],
        "llm": ["large language model", "generative pre-trained", "in-context learning", "prompting"],
    }

    def expand_query(self, query: str) -> str:
        """Expands query with domain-specific research synonyms."""
        lower_q = query.lower()
        expansions = []
        for term, syns in self.EXPANSION_KEYWORDS.items():
            if re.search(r"\b" + re.escape(term) + r"\b", lower_q):
                for s in syns[:2]:
                    if s.lower() not in lower_q:
                        expansions.append(s)
        if expansions:
            return f"{query} ({', '.join(expansions)})"
        return query

    def generate_subqueries(self, main_query: str, custom_objectives: List[str] = None) -> List[Dict[str, str]]:
        """
        Decomposes research question into sub-queries.
        Returns list of dicts: [{"query": str, "facet": str}]
        """
        subqueries = [{"query": main_query.strip(), "facet": "Primary"}]

        # Add custom sub-objectives if user provided them
        if custom_objectives:
            for obj in custom_objectives:
                if obj.strip() and obj.strip().lower() != main_query.strip().lower():
                    subqueries.append({"query": obj.strip(), "facet": "User Objective"})

        # Add automated faceted subqueries
        clean_q = re.sub(r"[?!.,]", "", main_query).strip()
        for template, facet in self.SUBTOPIC_TEMPLATES:
            subqueries.append({
                "query": template.format(query=clean_q),
                "facet": facet
            })

        return subqueries
