# LinkedIn Launch Post (Short Version)

I got tired of generic "chat with PDF" wrappers that hallucinate citations and blur research findings together.

So I built **ResearchRAG** — an autonomous AI Research Report Generator! 🔬🚀

Instead of a chat interface, ResearchRAG takes multiple research PDFs and automatically generates structured 13-section literature reviews, 8-dimension comparison matrices, contradiction alerts, and citation provenance verification.

Key Technical Highlights:
🔹 **Ingestion**: PyMuPDF + section-aware recursive chunking (preserves equations & sections)
🔹 **Vector Search**: FAISS (`IndexFlatIP`) + `all-MiniLM-L6-v2` dense embeddings
🔹 **Reranking**: Two-stage retrieval with `cross-encoder/ms-marco-MiniLM`
🔹 **Provenance**: Automated citation validator verifying document, page, and N-gram overlap (80.5% lexical grounding score, 96.9% similarity-threshold alignment)
🔹 **Export**: Publication-ready PDF, Markdown, and CSV matrices

Full code, architecture breakdown, evaluation benchmarks, and sample dataset available on GitHub:
👉 https://github.com/your-username/researchrag

#MachineLearning #RAG #AI #Python #VectorSearch #OpenSource #NLP
