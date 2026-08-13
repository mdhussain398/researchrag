# ResearchRAG: Technical Interview Preparation Guide

This comprehensive guide prepares you for machine learning, software engineering, and AI system design interviews covering RAG architectures, information retrieval, vector search, and evaluation.

---

## 1. Why RAG instead of Fine-Tuning?
- **Simple Answer**: Fine-tuning teaches a model a new style or task format, but is expensive, static, and prone to hallucinations on factual queries. RAG allows dynamic retrieval from fresh, auditable external documents with exact source citations.
- **Technical Answer**: Parameter weights in pre-trained LLMs store non-parametric knowledge implicitly via loss minimization, which decays and cannot provide verifiable source provenance. Fine-tuning updates parameter weights but costs significant GPU compute and suffers from catastrophic forgetting. RAG decouples knowledge storage (external vector index) from parametric reasoning (the generator), enabling $O(1)$ knowledge updates without retraining, sub-second citation verification, and strict access control.
- **Interviewer Follow-up**: *"When would you combine fine-tuning with RAG?"*
  - **Answer**: You fine-tune the generator or embedding model on domain-specific vocabulary (e.g. biomedical, legal) to improve retrieval sensitivity and report tone, while using RAG at inference time for factual retrieval and grounded citations.

---

## 2. How do Embeddings and Dense Vector Search Work?
- **Simple Answer**: An embedding model converts a piece of text into a list of numbers (a vector) capturing its semantic meaning. Similar texts end up close together in vector space, allowing us to find relevant passages using mathematical distance formulas.
- **Technical Answer**: Bi-encoders (like `all-MiniLM-L6-v2`) process text through a Transformer encoder and apply mean pooling across token hidden states to produce a fixed-dimensional dense vector $\mathbf{v} \in \mathbb{R}^{384}$. By $L_2$-normalizing vectors ($\|\mathbf{v}\|_2 = 1$), cosine similarity between query $\mathbf{q}$ and passage $\mathbf{p}$ reduces to the dot product $\mathbf{q} \cdot \mathbf{p}$. In FAISS, `IndexFlatIP` computes maximum inner product search (MIPS) across vectors in sub-milliseconds.
- **Interviewer Follow-up**: *"What is the computational complexity of exact vector search, and how do you scale it to billions of vectors?"*
  - **Answer**: Exact search has $O(N \cdot d)$ complexity where $N$ is the number of vectors and $d$ is dimension. To scale to billions of vectors, we transition to Approximate Nearest Neighbor (ANN) indexing like Inverted File with Product Quantization (IVF-PQ) or Hierarchical Navigable Small World (HNSW) graphs, trading <2% recall for 100x speed and memory reduction.

---

## 3. What is your Chunking Strategy, and why not use fixed 500-token chunks?
- **Simple Answer**: Academic papers have clear sections (like Abstract, Methodology, Results). Slicing blindly by 500 characters breaks paragraphs mid-sentence and mixes unrelated sections. We use section-aware recursive chunking that respects sentence boundaries and section headers.
- **Technical Answer**: Fixed-size windowing introduces boundary truncation artifacts—splitting critical mathematical formulas or experimental metrics across chunk boundaries. Our `AcademicChunker` detects canonical section headers, protects abbreviations (`et al.`, `i.e.`) from sentence splitting, applies recursive paragraph-to-sentence sliding windows (750 chars with 150-char overlap), and tags each chunk with document, page, and section metadata for provenance tracking.
- **Interviewer Follow-up**: *"How does chunk size affect retrieval accuracy versus generation context?"*
  - **Answer**: Smaller chunks (200-400 tokens) maximize embedding vector specificity (less noise), while larger chunks (800-1200 tokens) provide broader contextual coherence for the LLM. Using hierarchical chunking (small chunks for vector retrieval, parent chunk injection into the prompt) gives the optimal trade-off.

---

## 4. Why use a Two-Stage Retrieval Pipeline (Bi-Encoder + Cross-Encoder)?
- **Simple Answer**: Bi-encoders are fast but less accurate because they compare text vectors separately. Cross-encoders are much more accurate because they read the query and document together, but are too slow to run on all documents. We use bi-encoders to find top 20 candidates, and a cross-encoder to rerank the top 8.
- **Technical Answer**: Bi-encoders project query $\mathbf{q}$ and passage $\mathbf{p}$ into isolated embedding spaces, computing similarity via dot product without cross-token interaction ($O(N)$ dot products). A Cross-Encoder processes the concatenated sequence `[CLS] q [SEP] p` through all self-attention layers, allowing every query token to attend to every passage token. Since cross-attention is computationally heavy ($O(K \cdot L^2)$), we use a two-stage funnel: FAISS bi-encoder for high-recall top-20 candidate generation in <1ms, followed by `ms-marco-MiniLM` cross-encoder for high-precision top-8 reranking in ~25ms.
- **Interviewer Follow-up**: *"What is ColBERT, and how does it compare to Cross-Encoders?"*
  - **Answer**: ColBERT uses 'Late Interaction'—it computes token-level multi-vector embeddings for query and document, and scores relevance using MaxSim operations across token pairs. It provides 95% of cross-encoder accuracy with sub-10ms retrieval speeds.

---

## 5. How do you prevent and audit Hallucinations in Research Reports?
- **Simple Answer**: We force the LLM to cite numbered evidence chunks `[N]`, and then run an automated citation validator that checks if the document, page number, and evidence text actually support the claim.
- **Technical Answer**: Hallucination mitigation is handled across three layers:
  1. **Prompt Grounding**: System prompts strictly enforce that every factual claim must be accompanied by `[N]` referencing injected context chunks, with negative constraints ("Insufficient evidence in provided sources").
  2. **Automated Provenance Audit**: `CitationValidator` parses all `[N]` brackets, verifying that index $N$ exists in the retrieved pool and the page number matches document limits.
  3. **Semantic N-gram Overlap**: Measures stopword-filtered token intersection between the claim sentence and chunk text ($\ge 0.15$), calculating a quantitative Grounding Score and flagging unsupported citations.
- **Interviewer Follow-up**: *"How do you handle cases where two uploaded papers directly contradict each other?"*
  - **Answer**: `ContradictionDetector` identifies opposing claims on identical technical topics (e.g. dense vs sparse retrieval superiority), highlights both viewpoints with their respective citations, and surfaces them in a dedicated 'Contradictions & Disagreements' report section rather than allowing the LLM to blend contradictory facts.

---

## 6. How do you evaluate RAG Systems in Production?
- **Simple Answer**: We evaluate both retrieval (did we find the right chunks?) and generation (did the model write a faithful, well-cited report without making things up?).
- **Technical Answer**: We employ component-wise evaluation metrics:
  - **Retrieval Metrics**: Mean Reciprocal Rank (MRR), Precision@K, Recall@K across faceted subqueries.
  - **Generation Metrics**: Citation Validity Rate, Semantic Grounding Score, Source Coverage Ratio, and Hallucination Claim Rate.
  - **Automated Benchmarks**: We maintain `scripts/evaluate.py` with golden QA pairs to detect retrieval regression across code changes.
- **Interviewer Follow-up**: *"What is the difference between Faithfulness and Answer Relevance in Ragas?"*
  - **Answer**: *Faithfulness* measures whether the answer can be inferred purely from the retrieved context (groundedness). *Answer Relevance* measures whether the generated answer directly addresses the original user question without extraneous fluff.
