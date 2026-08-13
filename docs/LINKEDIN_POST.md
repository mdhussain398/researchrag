# LinkedIn Launch Post (Full Version)

Most "Chat with your PDF" projects are just simple wrappers that pass random chunks into an LLM and hope for the best.

When you're dealing with academic research papers, that approach falls apart:
❌ Paragraphs get sliced mid-equation.
❌ Factual claims are hallucinated with zero provenance.
❌ Conflicting empirical findings between studies get silently blurred together.

To solve this, I built **ResearchRAG** — an autonomous AI Research Report Generator powered by Retrieval-Augmented Generation (RAG). 🔬⚡

Instead of building another chatbot, I engineered a structured research-analysis product that ingests multiple papers and automatically produces publication-grade synthesis reports with strict citation provenance.

---

### 🧠 What's happening under the hood:

1. **Academic Section-Aware Parsing**: PyMuPDF extracts multi-column text, cleans ligatures/hyphenations, and splits documents along canonical section headers (Abstract, Methodology, Results, Limitations) to preserve semantic coherence.
2. **Dense Vector Search with FAISS**: Chunks are embedded into 384-dimensional dense vectors using `all-MiniLM-L6-v2` with exact $L_2$-normalized cosine similarity (`IndexFlatIP`).
3. **Two-Stage Retrieval & Reranking**: The research question is decomposed into 4 faceted sub-queries (Methodology, Results, Limitations, Comparisons), deduplicated via word-level Jaccard similarity, and reranked using a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`).
4. **Cross-Study Contradiction & Gap Detection**: The engine automatically identifies conflicting empirical benchmarks between papers (e.g. dense vs sparse retrieval superiority out-of-domain) and highlights unaddressed research gaps.
5. **Automated Citation Provenance Audit**: Every claim is cited `[N]` and verified against underlying chunk text and page bounds using token overlap, achieving an **80.5% lexical grounding score** and **96.9% similarity-threshold alignment** on our curated benchmark suite.
6. **Multi-Format Export**: Generates styled publication-ready PDFs via ReportLab, Markdown reports, and downloadable comparison CSVs.

Built with Python, Streamlit, FAISS, Sentence-Transformers, PyMuPDF, ReportLab, and modular LLM support (Gemini, Groq, OpenAI, Ollama, + an offline deterministic synthesizer that works 100% locally with zero API keys).

Check out the full repository and architecture diagrams below! 👇

🔗 GitHub: https://github.com/your-username/researchrag

#ArtificialIntelligence #MachineLearning #RAG #NLP #Python #OpenSource #SoftwareEngineering #DeepLearning #LLMs
