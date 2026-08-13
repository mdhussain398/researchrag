# ResearchRAG: 3-Minute Video / Live Demo Script

Use this structured script to present or record a high-impact demo of **ResearchRAG**.

---

## ⏱️ Timeline & Step-by-Step Walkthrough

### 0:00 - 0:30 | The Problem & The Solution
> **Speaker**: "Hey everyone! Most AI PDF tools are just generic chatbots—you ask a question and get a loose summary with zero citation verification or cross-paper synthesis.
> Today, I'm excited to demonstrate **ResearchRAG**, an autonomous AI Research Report Generator powered by Retrieval-Augmented Generation. Instead of chatting, it ingests multi-page research PDFs, builds a FAISS vector index, performs cross-study analysis, detects contradictions, identifies research gaps, and generates a structured 13-section report where every factual claim is strictly verified against source chunks."

---

### 0:30 - 1:15 | Document Ingestion & Vector Indexing
> **Action**: Navigate to `📄 Documents` view in Streamlit.
> **Speaker**: "Let's head over to the **Documents** tab. We can drag and drop custom PDFs, or click **'Load Sample Research Papers'** to instantly load three peer-reviewed AI papers on Dense Passage Retrieval, BEIR benchmarks, and Long-Context LLMs."
> **Action**: Click **'Load Sample Research Papers'**. Show the parsed metadata table.
> **Speaker**: "In seconds, PyMuPDF parsed the multi-column text, cleaned ligatures and hyphenations, detected section headers (Abstract, Methodology, Results, Limitations), and segmented the corpus into 18 section-aware chunks indexed in FAISS with 384-dimensional `all-MiniLM-L6-v2` embeddings."

---

### 1:15 - 2:00 | Research Setup & Autonomous Synthesis
> **Action**: Navigate to `⚙️ Research Setup` and then `📊 Generate Report`.
> **Speaker**: "Now, let's configure our research topic: *'How does dense passage retrieval (DPR) compare with traditional sparse BM25 retrieval for open-domain question answering?'*
> Notice we can customize top-K chunks, enable Cross-Encoder reranking, and select our LLM provider—including Google Gemini, Groq, OpenAI, or our offline deterministic synthesizer that works with zero API keys."
> **Action**: Click **'Generate Autonomous Research Report'**. Show the real-time progress steps.
> **Speaker**: "Watch the autonomous pipeline execute: multi-query faceted retrieval, word-level deduplication, cross-encoder reranking, analytical extraction, structured report generation, and automated citation auditing."

---

### 2:00 - 2:40 | Inspecting Analytical Artifacts & Citations
> **Action**: Scroll down the generated report. Click citation chips.
> **Speaker**: "Here is our 13-section research report. Notice:
> 1. **Citation Grounding**: Every claim has bracketed citations like `[1]` or `[2]`. When we expand a citation chip, we see the exact source document, page number, and quoted evidence with a verified grounding score of 80%+.
> 2. **Paper Comparison Matrix**: In the **Paper Comparison** tab, we have a structured 8-dimension table comparing Problem, Methodology, Datasets, and Metrics side-by-side.
> 3. **Contradictions & Gaps**: The system automatically identified that Paper 2 contradicts Paper 1 regarding whether dense retrieval outperforms BM25 on zero-shot out-of-domain benchmarks."

---

### 2:40 - 3:00 | Evaluation Dashboard & Export
> **Action**: Switch to `📈 Evaluation` view and show the download buttons.
> **Speaker**: "Finally, in the **Evaluation** tab, we calculate real quantitative metrics: Mean Reciprocal Rank (MRR), Precision@K, citation validity, and hallucination rates. We can export the entire publication-ready report as Markdown, styled PDF, or CSV.
> ResearchRAG bridges the gap between raw research literature and verifiable, grounded scientific synthesis. Thank you!"
