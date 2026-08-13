# Technical Decisions & Engineering Trade-Offs

This document details the architectural decisions, trade-offs, and design rationales made during the development of **ResearchRAG**.

---

## 1. Document Ingestion: PyMuPDF (`fitz`) vs. `pypdf` vs. `pdfminer`

| Decision | Chosen: **PyMuPDF (`fitz`)** |
| :--- | :--- |
| **Alternatives Evaluated** | `pypdf`, `pdfplumber`, `pdfminer.six`, `Unstructured` |
| **Rationale** | 1. **Performance**: PyMuPDF is written in C (MuPDF engine) and processes 50-page academic papers in under 100ms, whereas pure-Python parsers take 2-4 seconds.<br>2. **Layout Order**: PyMuPDF's `get_text("blocks")` extracts blocks ordered by coordinate hierarchy, preventing column-crossing text interleaving in 2-column academic papers.<br>3. **Provenance**: PyMuPDF provides exact page indexing and bounding coordinates required for grounded citation tracking. |
| **Trade-Off** | Requires compiled binary wheels (supported on all major OS platforms including macOS ARM64 and Linux x86_64). |

---

## 2. Chunking Strategy: Academic Section-Aware Recursive Chunking vs. Fixed-Size Windowing

| Decision | Chosen: **Academic Section-Aware Recursive Chunking** |
| :--- | :--- |
| **Alternatives Evaluated** | Fixed Character Windowing (500 chars), Naive Recursive Splitting, Sentence-only Chunking |
| **Rationale** | Academic research papers have distinct semantic boundaries (e.g. Abstract vs Methodology vs Results vs Limitations). Mixing abstract claims with methodology parameters creates noisy context vectors.<br>Our chunker: <br>• Detects section headers via regex.<br>• Preserves sentence boundaries with academic abbreviation protection (`et al.`, `i.e.`, `Fig.`).<br>• Uses a 750-character sliding window with a 150-character overlap.<br>• Attaches section name, document ID, and page number to each chunk. |
| **Trade-Off** | Slightly higher preprocessing logic complexity than naive fixed chunking. |

---

## 3. Vector Database: Local FAISS vs. ChromaDB vs. Pinecone

| Decision | Chosen: **FAISS (`faiss-cpu`) with In-Memory / Disk Persistence** |
| :--- | :--- |
| **Alternatives Evaluated** | ChromaDB, Qdrant, Weaviate, Pinecone, pgvector |
| **Rationale** | 1. **Zero External Infrastructure**: FAISS runs locally in-process without requiring background Docker containers, network connections, or cloud credentials.<br>2. **Exact Cosine Similarity**: Using `IndexFlatIP` on $L_2$-normalized vectors provides exact (non-approximate) cosine similarity for corpora of thousands of chunks in <1ms.<br>3. **Simplicity & Portability**: Serializes cleanly to a single binary `.faiss` file and JSON metadata ledger. |
| **Trade-Off** | For corpora exceeding 50M vectors, approximate indexing (IVF-PQ or HNSW) and distributed vector databases would be needed. |

---

## 4. Embeddings: `all-MiniLM-L6-v2` vs. OpenAI `text-embedding-3-small`

| Decision | Chosen: **`sentence-transformers/all-MiniLM-L6-v2` (Local)** |
| :--- | :--- |
| **Alternatives Evaluated** | `BAAI/bge-small-en-v1.5`, OpenAI `text-embedding-3-small`, Cohere Embed |
| **Rationale** | 1. **Zero API Cost & Unlimited Throughput**: Runs locally on CPU/Apple Silicon with zero API rate limits.<br>2. **Speed & Size**: 384-dimensional dense vectors with a 90MB model footprint, enabling sub-millisecond batch encoding.<br>3. **Deterministic Fallback**: System includes a pure-Python hash vectorizer fallback for air-gapped systems. |
| **Trade-Off** | Slightly lower MTEB retrieval benchmark score than 1536-dim OpenAI embeddings (compensated by Cross-Encoder reranking). |

---

## 5. Reranking: Multi-Stage Cross-Encoder vs. Single-Stage Bi-Encoder Retrieval

| Decision | Chosen: **Two-Stage Retrieval (Bi-Encoder Retrieval + Cross-Encoder Reranker)** |
| :--- | :--- |
| **Alternatives Evaluated** | Single-stage Bi-Encoder cosine retrieval, BM25-only, ColBERT (Late Interaction) |
| **Rationale** | Bi-encoders project query and passage into separate vectors, losing fine-grained cross-token attention. Cross-encoders (`cross-encoder/ms-marco-MiniLM-L-6-v2`) process query and passage jointly through all Transformer attention layers, achieving much higher top-1 precision.<br>The two-stage pipeline uses FAISS for rapid candidate generation ($K=20$) and Cross-Encoder for precision reranking ($K=8$), balancing speed and precision. |
| **Trade-Off** | Cross-Encoder adds ~15-30ms CPU inference latency per query. |

---

## 6. Citation Grounding & Hallucination Prevention

| Decision | Chosen: **Multi-Tier Citation Grounding with Automated Verification Audit** |
| :--- | :--- |
| **Alternatives Evaluated** | Raw Prompted Citations, Post-hoc String Matching, Strict Regex Extraction |
| **Rationale** | Generative LLMs frequently hallucinate fake citations or map real claims to wrong document pages. ResearchRAG enforces a 3-layer audit:<br>1. **Prompt Constraint**: Every claim must cite `[N]` from numbered context snippets.<br>2. **Provenance Verification**: Validates that chunk $N$ exists in the corpus and page number matches document length.<br>3. **N-Gram Overlap Check**: Measures token overlap between claim sentence and source chunk, flagging unverified citations. |
| **Trade-Off** | Adds an automated post-generation verification step, but guarantees transparent auditability for enterprise and academic use. |
