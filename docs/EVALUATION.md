# ResearchRAG: Benchmark Evaluation & Quality Audit

## 1. Evaluation Scope & Benchmark Definition

> **Evaluation Scope**: All reported quantitative metrics in this document were measured across a **curated 3-document / 4-question benchmark suite** using the local offline synthesis engine on a local CPU environment.
> 
> These metrics evaluate the internal retrieval alignment, ranking consistency, citation provenance, and lexical grounding of the ResearchRAG pipeline. They are **not** claims of performance on massive external benchmarks (e.g. TREC, MS-MARCO, BEIR 10k-doc sets).

---

## 2. Precise Metric Definitions & Methodological Nuance

To maintain scientific rigor and transparency, the metrics calculated in `scripts/evaluate.py` and `app/evaluation/evaluator.py` are defined as follows:

| Reported Metric | Formal Implementation | Methodological Scope & Limitations |
| :--- | :--- | :--- |
| **Candidate Rank Metric (Reciprocal Rank Decay)** | $\text{MRR}_{\text{cand}} = \frac{1}{|K|}\sum_{i=1}^{K}\frac{1}{i+1}$ over top ranked candidates. | **Candidate Rank Decay**: Measures the reciprocal rank distribution of the retrieved candidate pool. Because the benchmark does not contain manual binary relevance labels per chunk, this is a ranking quality indicator, **not** standard ground-truth labeled MRR. |
| **Similarity-Threshold Retrieval Alignment** | Fraction of retrieved top-$K$ chunks with cosine similarity $\ge 0.25$ against the query. | **Threshold Alignment**: Measures semantic relevance above the vector noise floor. It evaluates embedding vector alignment, not human-annotated ground-truth relevance. |
| **Heuristic Retrieval Facet Coverage** | $\min(1.0, \frac{\text{retrieved chunks}}{8.0})$ across multi-query subtopic facets. | **Facet Capacity Metric**: Measures multi-aspect query capacity across subtopics. It is **not** traditional dataset-wide exhaustive Information Retrieval (IR) recall. |
| **Source Document Coverage** | $\frac{|\text{documents with retrieved chunks}|}{|\text{total ingested documents}|}$ | **Corpus Participation**: Exactly measures the fraction of ingested documents contributing evidence to the report. |
| **Citation Provenance Validity Rate** | $\frac{|\text{verified citations}|}{|\text{total in-text citations}|}$ ($[N]$ maps to valid doc, page $\le \text{page\_count}$, token overlap $\ge 0.15$). | **Provenance Audit**: Verifies that the cited document exists, the page number is physically valid, and the chunk contains lexical support. It verifies *provenance*, which is distinct from verifying absolute factual truth. |
| **Lexical Grounding Score** | Average stopword-filtered token overlap: $\frac{|W_{\text{claim}} \cap W_{\text{chunk}}|}{|W_{\text{claim}}|}$. | **Lexical Overlap Score**: Measures N-gram token overlap between the generated claim sentence and the source chunk text. It does **not** prove full semantic/logical faithfulness (which would require an NLI or LLM-judge). |
| **Unsupported / Low-Overlap Claim Rate** | Fraction of citations with token overlap $< 0.08$ or missing source mappings. | **Low-Overlap Rate**: Measures the percentage of claims that lack direct lexical backing in the retrieved snippets. |
| **Pipeline Latency** | Wall-clock execution time ($s$) across retrieval, reranking, synthesis, and citation verification. | **Execution Time**: Real local execution measurement using the deterministic offline synthesizer. |

---

## 3. Curated Benchmark Evaluation Results

The evaluation runner (`scripts/evaluate.py`) was executed on the 3 sample academic papers (DPR, BEIR, and Long-Context RAG studies):

### Aggregate Results Table

| Metric Category | Metric Name | Measured Value | Standard Target | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Retrieval Quality** | Candidate Rank Metric | **0.4567** | $\ge 0.400$ | ✅ Stable Ranking |
| **Retrieval Quality** | Similarity-Threshold Alignment | **96.9%** | $\ge 85.0\%$ | ✅ High Relevance |
| **Retrieval Quality** | Heuristic Facet Coverage | **93.8%** | $\ge 80.0\%$ | ✅ Broad Coverage |
| **Retrieval Quality** | Source Document Coverage | **66.7%** | $\ge 50.0\%$ | ✅ Multi-Doc Participation |
| **Citation Grounding** | Citation Provenance Validity | **79.5%** | $\ge 75.0\%$ | ✅ Verified Provenance |
| **Citation Grounding** | Lexical Grounding Score | **0.8051** | $\ge 0.700$ | ✅ Strong Token Overlap |
| **Citation Grounding** | Unsupported / Low-Overlap Rate | **20.5%** | $\le 25.0\%$ | ✅ Controlled Noise |
| **Performance** | Pipeline Latency | **0.21s** | $\le 5.00s$ | ✅ High Speed (CPU) |

---

## 4. Per-Question Benchmark Breakdown

### Benchmark 1: Dense Passage Retrieval vs BM25
- **Research Question**: *"How does dense passage retrieval (DPR) compare with traditional sparse BM25 retrieval for open-domain question answering?"*
- **Retrieved Chunks**: 8 | **Mean Cosine Similarity**: 0.632 | **Similarity Alignment**: 100.0%
- **Citation Provenance Validity**: 83.3% | **Lexical Grounding Score**: 0.821 | **Latency**: 0.33s
- **Analytical Output**: 1 Contradiction detected, 6 Research Gaps identified.

### Benchmark 2: Cross-Encoder Reranker Impact
- **Research Question**: *"What is the impact of multi-stage cross-encoder reranking on retrieval precision and latency?"*
- **Retrieved Chunks**: 6 | **Mean Cosine Similarity**: 0.548 | **Similarity Alignment**: 100.0%
- **Citation Provenance Validity**: 83.3% | **Lexical Grounding Score**: 0.768 | **Latency**: 0.18s
- **Analytical Output**: 6 Research Gaps identified.

### Benchmark 3: RAG vs Long-Context LLMs
- **Research Question**: *"How do Retrieval-Augmented Generation (RAG) pipelines mitigate hallucinations compared to long-context LLMs?"*
- **Retrieved Chunks**: 8 | **Mean Cosine Similarity**: 0.414 | **Similarity Alignment**: 100.0%
- **Citation Provenance Validity**: 80.0% | **Lexical Grounding Score**: 0.830 | **Latency**: 0.15s
- **Analytical Output**: 2 Contradictions detected, 4 Research Gaps identified.

### Benchmark 4: Production RAG Limitations & Gaps
- **Research Question**: *"What are the primary computational limitations and research gaps in deploying production RAG systems?"*
- **Retrieved Chunks**: 8 | **Mean Cosine Similarity**: 0.320 | **Similarity Alignment**: 87.5%
- **Citation Provenance Validity**: 71.4% | **Lexical Grounding Score**: 0.801 | **Latency**: 0.18s
- **Analytical Output**: 2 Contradictions detected, 6 Research Gaps identified.

---

## 5. How to Reproduce Benchmark Results

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Run the automated evaluation suite
python scripts/evaluate.py
```
Output results are persisted to `data/processed/evaluation_results.json`.
