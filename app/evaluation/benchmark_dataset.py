"""
Benchmark evaluation dataset containing curated research questions and ground-truth targets.
"""

from typing import List, Dict, Any

BENCHMARK_SUITE: List[Dict[str, Any]] = [
    {
        "benchmark_id": "bm_01",
        "question": "How does dense passage retrieval (DPR) compare with traditional sparse BM25 retrieval for open-domain question answering?",
        "expected_subtopics": ["dual-encoder architecture", "BM25 lexical baseline", "exact match accuracy", "hybrid sparse-dense"],
        "target_metrics": ["Top-20 retrieval accuracy", "MRR@10"],
        "expected_contradiction_topic": "Dense vs. Sparse Retrieval Superiority",
    },
    {
        "benchmark_id": "bm_02",
        "question": "What is the impact of multi-stage cross-encoder reranking on retrieval precision and latency?",
        "expected_subtopics": ["cross-encoder cross-attention", "reranking latency overhead", "precision gains"],
        "target_metrics": ["Latency ms", "MRR@10", "NDCG@10"],
        "expected_contradiction_topic": "Reranker Performance vs. Computational Overhead",
    },
    {
        "benchmark_id": "bm_03",
        "question": "How do Retrieval-Augmented Generation (RAG) pipelines mitigate hallucinations compared to long-context LLMs?",
        "expected_subtopics": ["parametric vs non-parametric memory", "factual grounding", "needle-in-a-haystack", "hallucination mitigation"],
        "target_metrics": ["Faithfulness", "Citation Recall", "Hallucination Rate"],
        "expected_contradiction_topic": "Long-Context LLMs vs. RAG for Knowledge Grounding",
    },
    {
        "benchmark_id": "bm_04",
        "question": "What are the primary computational limitations and research gaps in deploying production RAG systems?",
        "expected_subtopics": ["vector search scaling", "cross-domain degradation", "multi-hop reasoning", "inference latency"],
        "target_metrics": ["Throughput", "Index memory"],
        "expected_contradiction_topic": "Granular vs. Coarse Chunking Trade-off",
    },
]
