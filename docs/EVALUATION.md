# ResearchRAG: Quantitative Evaluation & Quality Benchmark

## 1. Evaluation Methodology

Automated evaluation of Retrieval-Augmented Generation (RAG) systems requires assessing both the **retrieval quality** (whether relevant evidence is surfaced) and the **generation fidelity** (whether claims are grounded without hallucinations).

ResearchRAG uses a quantitative evaluation suite executed via `scripts/evaluate.py`.

### 1.1 Metrics Defined

1. **Mean Reciprocal Rank (MRR)**:
   Measures the ranking quality of retrieved evidence:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
2. **Retrieval Precision@K**:
   Fraction of the top-$K$ retrieved chunks that surpass the relevance similarity threshold ($\ge 0.25$).
3. **Retrieval Recall@K**:
   Coverage of essential query facets across the top-$K$ candidate pool.
4. **Source Document Coverage**:
   Fraction of ingested papers actively contributing to the synthesized evidence pool.
5. **Citation Validity Rate**:
   Percentage of in-text citations $[N]$ that map to an authentic chunk ID, verified document name, and valid page number within the uploaded corpus.
6. **Semantic Grounding Score**:
   Average token overlap ratio (filtering stopwords) between each claim sentence and the cited chunk context.
7. **Hallucination / Unsupported Claim Rate**:
   Percentage of in-text citations that fail verification checks or lack contextual support in the retrieved evidence snippets.
8. **End-to-End Pipeline Latency**:
   Total wall-clock duration (seconds) required for multi-query retrieval, deduplication, reranking, synthesis, analytical extraction, and citation auditing.

---

## 2. Benchmark Evaluation Results

The evaluation suite was executed across 4 standardized research questions on the curated academic corpus (DPR, BEIR, and Long-Context RAG papers).

### Aggregate Summary Table

| Metric | Measured Value | Target Standard | Status |
| :--- | :--- | :--- | :--- |
| **Mean Reciprocal Rank (MRR)** | **0.4567** | $\ge 0.400$ | ✅ Exceeds Target |
| **Retrieval Precision@K** | **96.9%** | $\ge 85.0\%$ | ✅ Exceeds Target |
| **Retrieval Recall@K** | **93.8%** | $\ge 80.0\%$ | ✅ Exceeds Target |
| **Source Document Coverage** | **66.7%** | $\ge 50.0\%$ | ✅ Exceeds Target |
| **Citation Validity Rate** | **79.5%** | $\ge 75.0\%$ | ✅ Exceeds Target |
| **Semantic Grounding Score** | **0.8051** | $\ge 0.700$ | ✅ Exceeds Target |
| **Hallucination Rate** | **20.5%** | $\le 25.0\%$ | ✅ Within Bound |
| **Average Report Latency** | **0.21s** | $\le 5.00s$ | ✅ High Speed |

---

## 3. Individual Benchmark Breakdown

### Benchmark 1: Dense Passage Retrieval vs BM25
- **Research Question**: *"How does dense passage retrieval (DPR) compare with traditional sparse BM25 retrieval for open-domain question answering?"*
- **Retrieved Chunks**: 8 | **Mean Similarity**: 0.632 | **Precision@K**: 100.0%
- **Citation Validity**: 83.3% | **Grounding Score**: 0.821 | **Latency**: 0.33s
- **Analytical Output**: 1 Contradiction detected (Dense vs Sparse Out-of-Domain Generalization), 6 Research Gaps identified.

### Benchmark 2: Cross-Encoder Reranker Impact
- **Research Question**: *"What is the impact of multi-stage cross-encoder reranking on retrieval precision and latency?"*
- **Retrieved Chunks**: 6 | **Mean Similarity**: 0.548 | **Precision@K**: 100.0%
- **Citation Validity**: 83.3% | **Grounding Score**: 0.768 | **Latency**: 0.18s
- **Analytical Output**: 6 Research Gaps identified (Computational latency bottlenecks).

### Benchmark 3: RAG vs Long-Context LLMs
- **Research Question**: *"How do Retrieval-Augmented Generation (RAG) pipelines mitigate hallucinations compared to long-context LLMs?"*
- **Retrieved Chunks**: 8 | **Mean Similarity**: 0.414 | **Precision@K**: 100.0%
- **Citation Validity**: 80.0% | **Grounding Score**: 0.830 | **Latency**: 0.15s
- **Analytical Output**: 2 Contradictions detected (Lost in the middle phenomenon), 4 Research Gaps identified.

### Benchmark 4: Production RAG Bottlenecks
- **Research Question**: *"What are the primary computational limitations and research gaps in deploying production RAG systems?"*
- **Retrieved Chunks**: 8 | **Mean Similarity**: 0.320 | **Precision@K**: 87.5%
- **Citation Validity**: 71.4% | **Grounding Score**: 0.801 | **Latency**: 0.18s
- **Analytical Output**: 2 Contradictions detected, 6 Research Gaps identified.

---

## 4. How to Run the Benchmark Suite

Run the automated CLI evaluation runner from your terminal:

```bash
# Activate virtual environment
source .venv/bin/activate

# Execute evaluation suite
python scripts/evaluate.py
```

The benchmark results will be saved to `data/processed/evaluation_results.json`.
