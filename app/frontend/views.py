"""
Streamlit view renderers for ResearchRAG.
Implements the 7 core views: Home, Documents, Setup, Report, Evidence, Comparison, Evaluation.
"""

import time
import pandas as pd
import streamlit as st
from typing import Dict, Any, List

from app.models.schemas import ResearchConfig, ResearchReport, DocumentMetadata, RetrievedChunk
from app.utils.config import SAMPLE_DIR, logger
from app.utils.export import ReportExporter
from app.generation.llm_client import LLMClient
from app.evaluation.benchmark_dataset import BENCHMARK_SUITE


def render_home():
    """Renders the Home & Overview page."""
    st.markdown("## 📚 Welcome to ResearchRAG")
    st.markdown(
        "**ResearchRAG** is an autonomous AI Research Report Generator powered by **Retrieval-Augmented Generation (RAG)**. "
        "Unlike generic 'chat with PDFs' tools, ResearchRAG performs multi-aspect cross-study synthesis, empirical comparison, "
        "contradiction detection, research gap identification, and strict grounded citation validation."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-title">Architecture</div>
            <div class="metric-card-value">Dense RAG</div>
            <div class="metric-card-subtitle">FAISS + MiniLM + Reranker</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-title">Citations</div>
            <div class="metric-card-value">Citation Grounded</div>
            <div class="metric-card-subtitle">Evidence-backed reports with citation validation</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-title">Analysis</div>
            <div class="metric-card-value">Multi-Paper</div>
            <div class="metric-card-subtitle">Contradictions & Gaps</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-title">Export</div>
            <div class="metric-card-value">MD, PDF, CSV</div>
            <div class="metric-card-subtitle">Publication-ready reports</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🔄 Autonomous Research Workflow")
    st.markdown("""
```mermaid
graph LR
    A[📄 Upload PDFs] --> B[✂️ Section-Aware Chunking]
    B --> C[🧠 Dense Embeddings & FAISS Index]
    C --> D[🎯 Faceted Multi-Query Retrieval]
    D --> E[⚡ Cross-Encoder Reranking & Dedup]
    E --> F[🔬 Synthesis & Contradiction Detection]
    F --> G[🛡️ Citation Grounding & Hallucination Audit]
    G --> H[📊 Publication Report, Matrix & PDF]
```
    """)

    st.markdown("### 🚀 Quick Start Guide")
    st.markdown("""
    1. **Go to 📄 Documents**: Upload your research PDFs, or click **'Load Sample Research Papers'** to test immediately with curated open-access AI papers.
    2. **Go to ⚙️ Research Setup**: Enter your research question (e.g. *'How does dense retrieval compare with BM25 across domains?'*) and configure retrieval depth.
    3. **Go to 📊 Generate Report**: Click **'Generate Autonomous Research Report'** to watch the multi-stage synthesis and citation verification in real time.
    4. **Inspect Analytical Views**: Explore **🔍 Evidence Explorer** for chunk provenance, **📑 Paper Comparison** for structured matrices, and **📈 Evaluation** for groundedness metrics.
    """)


def render_documents(ingestion_mgr, vector_store):
    """Renders the Document Ingestion & Management view."""
    st.markdown("## 📄 Document Management & Ingestion")
    st.markdown("Upload multiple research PDFs or load pre-packaged benchmark papers.")

    col1, col2 = st.columns([3, 2])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload Research Papers (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload one or more academic research papers in PDF format.",
        )

        if uploaded_files:
            if st.button("🚀 Process & Ingest Uploaded PDFs", type="primary"):
                saved_paths = []
                with st.spinner("Saving uploaded files..."):
                    for uf in uploaded_files:
                        saved_path = ingestion_mgr.save_uploaded_file(uf.getvalue(), uf.name)
                        saved_paths.append(saved_path)

                with st.spinner("Extracting text, detecting sections, and chunking..."):
                    result = ingestion_mgr.process_files(saved_paths)
                    new_chunks = result["chunks"]
                    vector_store.add_chunks(new_chunks)
                    vector_store.save()
                    st.success(f"Successfully processed {len(result['processed_documents'])} papers ({len(new_chunks)} chunks indexed in FAISS).")
                    st.rerun()

    with col2:
        st.markdown("#### Sample Benchmark Papers")
        st.info("No papers ready? Click below to instantly load 3 peer-reviewed open-access synthetic AI papers (DPR, BEIR, and Long-Context vs RAG).")
        if st.button("📥 Load Sample Research Papers", use_container_width=True):
            with st.spinner("Loading and indexing sample research papers..."):
                from scripts.generate_sample_papers import generate_sample_papers
                sample_paths = generate_sample_papers()
                result = ingestion_mgr.process_files(sample_paths)
                vector_store.clear()
                vector_store.add_chunks(result["chunks"])
                vector_store.save()
                st.success(f"Loaded 3 sample research papers ({len(result['chunks'])} chunks indexed).")
                st.rerun()

        if st.button("🗑️ Clear All Ingested Papers & Index", use_container_width=True):
            ingestion_mgr.clear_all()
            vector_store.clear()
            st.warning("Cleared all documents and vector index.")
            st.rerun()

    # Documents Registry Table
    docs = ingestion_mgr.get_all_documents()
    chunks = ingestion_mgr.get_all_chunks()

    st.markdown("---")
    st.markdown(f"### Ingested Document Registry ({len(docs)} papers, {len(chunks)} chunks)")

    if not docs:
        st.info("No documents currently ingested. Please upload PDFs or load the sample papers above.")
        return

    doc_rows = []
    for d in docs:
        doc_chunks = ingestion_mgr.chunks_by_doc.get(d.document_id, [])
        doc_rows.append({
            "Document ID": d.document_id,
            "Filename": d.filename,
            "Title": d.title or d.filename,
            "Authors": ", ".join(d.authors[:2]) + ("..." if len(d.authors) > 2 else "") if d.authors else "N/A",
            "Year": d.year or "N/A",
            "Pages": d.page_count,
            "Chunks": len(doc_chunks),
            "Sections Detected": ", ".join(d.detected_sections[:3]) + ("..." if len(d.detected_sections) > 3 else ""),
        })

    st.dataframe(pd.DataFrame(doc_rows), use_container_width=True)

    # Document Detail Accordion
    with st.expander("🔍 Inspect Document Details & Chunks"):
        selected_doc_id = st.selectbox("Select Paper", options=[d.document_id for d in docs], format_func=lambda x: next((d.filename for d in docs if d.document_id == x), x))
        if selected_doc_id:
            sel_doc = next(d for d in docs if d.document_id == selected_doc_id)
            st.markdown(f"**Title**: {sel_doc.title}")
            st.markdown(f"**Abstract**: {sel_doc.abstract or 'N/A'}")
            st.markdown(f"**Detected Sections**: `{', '.join(sel_doc.detected_sections)}`")
            doc_chunks = ingestion_mgr.chunks_by_doc.get(selected_doc_id, [])
            st.markdown(f"#### Chunks ({len(doc_chunks)} total)")
            for i, c in enumerate(doc_chunks[:4]):
                st.markdown(f"""
                <div class="evidence-card">
                    <div class="evidence-header">
                        <span>Chunk #{i+1} | Page {c.page_number} | Section: {c.section}</span>
                        <span>{c.token_count_est} tokens</span>
                    </div>
                    <div class="evidence-text">{c.text}</div>
                </div>
                """, unsafe_allow_html=True)


def render_setup(state: Dict[str, Any]):
    """Renders the Research Setup & Configuration view."""
    st.markdown("## ⚙️ Research Setup & Configuration")
    st.markdown("Formulate your research question, set sub-objectives, and tune retrieval parameters.")

    config: ResearchConfig = state.get("config", ResearchConfig())

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("#### 1. Research Question & Objectives")
        
        # Preset question selector
        preset_questions = [
            "Custom Research Question",
            "How does dense passage retrieval (DPR) compare with traditional sparse BM25 retrieval for open-domain question answering?",
            "What is the impact of multi-stage cross-encoder reranking on retrieval precision and latency?",
            "How do Retrieval-Augmented Generation (RAG) pipelines mitigate hallucinations compared to long-context LLMs?",
            "What are the primary computational limitations and research gaps in deploying production RAG systems?",
        ]
        
        selected_preset = st.selectbox("Select Benchmark Question or write custom:", options=preset_questions)
        
        if selected_preset != "Custom Research Question":
            default_q = selected_preset
        else:
            default_q = config.research_question or "How does dense retrieval compare with BM25 across zero-shot out-of-domain benchmarks?"

        research_q = st.text_area(
            "Primary Research Question",
            value=default_q,
            height=90,
            help="The overarching scientific or technical question you want the report to answer."
        )

        st.markdown("#### Optional Sub-Objectives / Facets")
        sub_objs_text = st.text_area(
            "Sub-Objectives (one per line)",
            value="\n".join(config.sub_objectives) if config.sub_objectives else "Methodology and architectural differences\nQuantitative benchmark results and metrics\nTrade-offs, limitations, and failure modes",
            height=90,
            help="Specific angles to retrieve evidence for."
        )
        sub_objectives = [line.strip() for line in sub_objs_text.split("\n") if line.strip()]

    with col2:
        st.markdown("#### 2. Retrieval & Reranking Settings")
        
        top_k = st.slider("Top-K Evidence Chunks", min_value=3, max_value=20, value=config.top_k_chunks or 8)
        sim_thresh = st.slider("Minimum Similarity Threshold", min_value=0.0, max_value=0.6, value=float(config.similarity_threshold or 0.15), step=0.05)
        enable_rerank = st.checkbox("Enable Cross-Encoder Reranking", value=config.enable_reranking)
        dedup_thresh = st.slider("Chunk Deduplication Threshold", min_value=0.5, max_value=0.95, value=float(config.deduplication_threshold or 0.85), step=0.05)

        st.markdown("#### 3. LLM Provider")
        avail_providers = LLMClient.get_available_providers()
        provider_options = ["gemini", "groq", "openai", "ollama", "local"]
        
        def format_prov(p):
            active = "✅ Active" if avail_providers.get(p, False) else "🔑 Requires Key"
            names = {
                "gemini": "Google Gemini (Gemini-1.5-Flash)",
                "groq": "Groq (Llama-3.1-8B)",
                "openai": "OpenAI (GPT-4o-Mini)",
                "ollama": "Ollama (Local LLM)",
                "local": "Deterministic Offline Synthesizer (Zero API Keys)",
            }
            return f"{names.get(p, p)} — {active}"

        selected_provider = st.selectbox("LLM Provider", options=provider_options, format_func=format_prov, index=provider_options.index(config.llm_provider) if config.llm_provider in provider_options else 4)

    # Save to state
    updated_config = ResearchConfig(
        research_question=research_q,
        sub_objectives=sub_objectives,
        top_k_chunks=top_k,
        similarity_threshold=sim_thresh,
        enable_reranking=enable_rerank,
        deduplication_threshold=dedup_thresh,
        llm_provider=selected_provider,
        temperature=0.2,
    )
    state["config"] = updated_config
    st.success("✅ Research configuration updated.")


def render_generate_report(state: Dict[str, Any], ingestion_mgr, retriever, report_gen, evaluator):
    """Renders the Generate Report and Output view."""
    st.markdown("## 📊 Generate Autonomous Research Report")

    config: ResearchConfig = state.get("config", ResearchConfig())
    docs = ingestion_mgr.get_all_documents()

    if not docs:
        st.warning("⚠️ No documents loaded! Please go to **📄 Documents** and upload PDFs or load the sample papers first.")
        return

    st.markdown(f"**Current Topic**: `{config.research_question}`")
    st.markdown(f"**Target Corpus**: {len(docs)} documents | **Provider**: `{config.llm_provider}`")

    if st.button("🚀 Generate Autonomous Research Report", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Starting RAG pipeline...")
        status_box = st.empty()

        start_time = time.time()

        # Step 1: Retrieval
        status_box.info("Step 1/5: Executing faceted multi-aspect evidence retrieval across FAISS index...")
        progress_bar.progress(20)
        retrieved_chunks = retriever.retrieve_evidence(config)

        if not retrieved_chunks:
            st.error("No relevant chunks found above similarity threshold. Try lowering the similarity threshold in ⚙️ Research Setup.")
            progress_bar.empty()
            return

        # Step 2: Reranking & Analysis
        status_box.info(f"Step 2/5: Retrieved {len(retrieved_chunks)} evidence chunks. Analyzing cross-paper dimensions & contradictions...")
        progress_bar.progress(45)

        # Step 3: Synthesis Generation
        status_box.info(f"Step 3/5: Synthesizing structured 13-section academic report via {config.llm_provider}...")
        progress_bar.progress(70)
        report = report_gen.generate_report(
            config=config,
            retrieved_chunks=retrieved_chunks,
            documents=docs,
            chunks_by_doc=ingestion_mgr.chunks_by_doc,
        )

        # Step 4: Citation Grounding Audit
        status_box.info("Step 4/5: Running citation validation & hallucination verification audit...")
        progress_bar.progress(88)

        # Step 5: Evaluation Metrics
        status_box.info("Step 5/5: Computing retrieval MRR, coverage, and faithfulness scores...")
        elapsed = time.time() - start_time
        metrics = evaluator.evaluate_report(report, docs, execution_time=elapsed)

        progress_bar.progress(100)
        status_box.success(f"✅ Research Report Generated and Verified in {elapsed:.2f}s!")

        state["current_report"] = report
        state["current_metrics"] = metrics

    # Display Report if available
    report: ResearchReport = state.get("current_report")
    if not report:
        st.info("Click 'Generate Autonomous Research Report' above to run the pipeline.")
        return

    st.markdown("---")
    
    # Validation & Metric Header
    metrics = state.get("current_metrics")
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Citation Validity</div>
                <div class="metric-card-value">{metrics.citation_validity_rate * 100:.1f}%</div>
                <div class="metric-card-subtitle">{len([c for c in report.citations if c.is_verified])}/{len(report.citations)} verified</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Semantic Grounding</div>
                <div class="metric-card-value">{metrics.semantic_grounding_score:.3f}</div>
                <div class="metric-card-subtitle">Context alignment score</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Hallucination Rate</div>
                <div class="metric-card-value">{metrics.hallucination_claim_rate * 100:.1f}%</div>
                <div class="metric-card-subtitle">Unsupported claim rate</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Source Coverage</div>
                <div class="metric-card-value">{metrics.source_coverage_ratio * 100:.1f}%</div>
                <div class="metric-card-subtitle">{metrics.retrieval_count} chunks retrieved</div>
            </div>
            """, unsafe_allow_html=True)

    # Export Buttons
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    with col_exp1:
        md_text = ReportExporter.to_markdown(report)
        st.download_button(
            "📥 Download Report (Markdown)",
            data=md_text,
            file_name=f"ResearchReport_{report.report_id}.md",
            mime="text/markdown",
            use_container_width=True,
        )
    with col_exp2:
        try:
            pdf_bytes = ReportExporter.to_pdf_bytes(report)
            st.download_button(
                "📥 Download Report (PDF)",
                data=pdf_bytes,
                file_name=f"ResearchReport_{report.report_id}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"PDF export unavailable: {e}")
    with col_exp3:
        if report.comparison_table:
            csv_data = ReportExporter.comparison_to_csv(report.comparison_table)
            st.download_button(
                "📥 Download Comparison (CSV)",
                data=csv_data,
                file_name="Paper_Comparison_Matrix.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("---")

    # Render Report Content
    st.markdown(report.raw_markdown)

    # Interactive Citation Inspector
    if report.citations:
        st.markdown("### 🔎 Interactive Citation & Provenance Explorer")
        for c in report.citations:
            badge_cls = "badge-verified" if c.is_verified else "badge-unverified"
            badge_text = "Verified Grounded" if c.is_verified else "Unverified / Phantom"
            with st.expander(f"[{c.citation_id}] {c.filename} (Page {c.page_number}) — {badge_text}"):
                st.markdown(f"<span class='{badge_cls}'>{badge_text}</span> &nbsp; Confidence: `{c.verification_confidence:.2f}`", unsafe_allow_html=True)
                st.markdown(f"**Section**: `{c.section}`")
                st.markdown(f"**Underlying Evidence Text**:\n> {c.quoted_evidence}")


def render_evidence_explorer(state: Dict[str, Any]):
    """Renders the Evidence Explorer view."""
    st.markdown("## 🔍 Evidence Explorer")
    st.markdown("Inspect all retrieved evidence chunks, similarity scores, and section provenance.")

    report: ResearchReport = state.get("current_report")
    if not report or not report.retrieved_chunks:
        st.info("No report generated yet. Go to **📊 Generate Report** first.")
        return

    chunks = report.retrieved_chunks

    col1, col2 = st.columns([2, 1])
    with col1:
        search_query = st.text_input("Filter Evidence by Keyword", "")
    with col2:
        doc_filter = st.selectbox(
            "Filter by Document",
            options=["All Documents"] + list(dict.fromkeys([c.chunk.filename for c in chunks]))
        )

    # Filter chunks
    filtered = []
    for c in chunks:
        if doc_filter != "All Documents" and c.chunk.filename != doc_filter:
            continue
        if search_query.strip() and search_query.lower() not in c.chunk.text.lower():
            continue
        filtered.append(c)

    st.markdown(f"Showing **{len(filtered)}** of **{len(chunks)}** retrieved chunks:")

    for i, item in enumerate(filtered, 1):
        c = item.chunk
        rerank_str = f" | Rerank Score: {item.rerank_score:.4f}" if item.rerank_score is not None else ""
        subtopic_str = f" | Subtopic: <i>{item.subtopic_matched}</i>" if item.subtopic_matched else ""
        
        st.markdown(f"""
        <div class="evidence-card">
            <div class="evidence-header">
                <span>Chunk #{i} | <b>{c.filename}</b> (Page {c.page_number}, Sec: {c.section})</span>
                <span>Similarity: <b>{item.similarity_score:.4f}</b>{rerank_str}{subtopic_str}</span>
            </div>
            <div class="evidence-text">{c.text}</div>
        </div>
        """, unsafe_allow_html=True)

    # Download Evidence Ledger
    st.markdown("---")
    csv_ledger = ReportExporter.evidence_to_csv(chunks)
    st.download_button(
        "📥 Download Full Evidence Ledger (CSV)",
        data=csv_ledger,
        file_name="Retrieved_Evidence_Ledger.csv",
        mime="text/csv",
    )


def render_paper_comparison(state: Dict[str, Any], ingestion_mgr):
    """Renders the Paper Comparison view."""
    st.markdown("## 📑 Multi-Paper Comparison Matrix")
    st.markdown("Structured comparison across methodologies, datasets, benchmarks, strengths, and limitations.")

    report: ResearchReport = state.get("current_report")
    docs = ingestion_mgr.get_all_documents()

    if not docs:
        st.info("No documents uploaded yet.")
        return

    from app.analysis.comparator import PaperComparator
    comparator = PaperComparator()
    matrix_rows = report.comparison_table if (report and report.comparison_table) else comparator.build_comparison_matrix(docs, ingestion_mgr.chunks_by_doc)

    if not matrix_rows:
        st.info("No comparison rows available.")
        return

    df = comparator.to_dataframe(matrix_rows)
    st.dataframe(df, use_container_width=True)

    # Contradictions & Disagreements Section
    st.markdown("### ⚡ Detected Disagreements & Contradictions")
    contradictions = report.contradictions if (report and report.contradictions) else []
    if contradictions:
        for c in contradictions:
            st.markdown(f"""
            <div class="contradiction-box">
                <div style="font-weight: 700; color: #991b1b; font-size: 0.95rem; margin-bottom: 4px;">
                    ⚡ {c.topic} (Confidence: {c.confidence_score*100:.0f}%)
                </div>
                <div style="font-size: 0.88rem; color: #334155; margin-bottom: 6px;">
                    <b>{c.source_a} (p. {c.page_a})</b>: "{c.claim_a}"<br>
                    <b>vs.</b><br>
                    <b>{c.source_b} (p. {c.page_b})</b>: "{c.claim_b}"
                </div>
                <div style="font-size: 0.82rem; color: #7f1d1d; font-style: italic;">
                    <b>Synthesis Note</b>: {c.explanation}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No explicit contradictory claims detected between the ingested sources.")

    # Research Gaps Section
    st.markdown("### 🔭 Identified Research Gaps & Limitations")
    gaps = report.research_gaps if (report and report.research_gaps) else []
    if gaps:
        for g in gaps:
            st.markdown(f"""
            <div class="gap-box">
                <div style="font-weight: 700; color: #1e40af; font-size: 0.92rem; margin-bottom: 2px;">
                    🔭 {g.category} &nbsp;<span style="font-size: 0.75rem; color: #475569; font-weight: normal;">[{g.gap_type}]</span>
                </div>
                <div style="font-size: 0.88rem; color: #1e293b; margin-bottom: 4px;">
                    {g.description}
                </div>
                <div style="font-size: 0.82rem; color: #2563eb;">
                    <b>Suggested Direction</b>: {g.suggested_future_work}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Download CSV
    st.markdown("---")
    csv_data = ReportExporter.comparison_to_csv(matrix_rows)
    st.download_button(
        "📥 Download Comparison Matrix (CSV)",
        data=csv_data,
        file_name="Paper_Comparison_Matrix.csv",
        mime="text/csv",
    )


def render_evaluation(state: Dict[str, Any], evaluator):
    """Renders the Evaluation & Quality Dashboard view."""
    st.markdown("## 📈 Quantitative Evaluation & Quality Metrics")
    st.markdown("Real-time statistical evaluation of retrieval precision, citation validity, and faithfulness.")

    metrics = state.get("current_metrics")

    if metrics:
        st.markdown(f"### Current Report Metrics (`{metrics.query}`)")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Retrieval Count", f"{metrics.retrieval_count} chunks")
            st.metric("Mean Similarity", f"{metrics.mean_similarity_score:.4f}")
        with col2:
            st.metric("Citation Validity", f"{metrics.citation_validity_rate * 100:.1f}%")
            st.metric("Page Match Rate", f"{metrics.page_match_rate * 100:.1f}%")
        with col3:
            st.metric("Semantic Grounding", f"{metrics.semantic_grounding_score:.4f}")
            st.metric("Hallucination Rate", f"{metrics.hallucination_claim_rate * 100:.1f}%")
        with col4:
            st.metric("Source Coverage", f"{metrics.source_coverage_ratio * 100:.1f}%")
            st.metric("Pipeline Latency", f"{metrics.execution_time_seconds:.2f}s")

        st.markdown("#### Evaluation Notes")
        for note in metrics.evaluation_notes:
            st.markdown(f"- {note}")

    st.markdown("---")
    st.markdown("### 🏆 Standard Benchmark Evaluation Suite")
    st.markdown("Run automated evaluation across the 4 standard benchmark research questions:")

    if st.button("🚀 Run Full Benchmark Suite (4 Questions)", type="primary"):
        with st.spinner("Running benchmark suite..."):
            from scripts.evaluate import run_benchmark_evaluation
            results = run_benchmark_evaluation(output_json=True)
            state["benchmark_summary"] = results
            st.success("✅ Benchmark Evaluation Complete!")

    bm_summary = state.get("benchmark_summary")
    if bm_summary:
        st.markdown("#### Aggregate Benchmark Summary")
        agg = bm_summary["aggregate_metrics"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Reciprocal Rank (MRR)", f"{agg['mean_reciprocal_rank']:.4f}")
            st.metric("Retrieval Precision@K", f"{agg['retrieval_precision_at_k'] * 100:.1f}%")
        with col2:
            st.metric("Citation Validity Rate", f"{agg['citation_validity_rate'] * 100:.1f}%")
            st.metric("Semantic Grounding Score", f"{agg['semantic_grounding_score']:.4f}")
        with col3:
            st.metric("Hallucination Rate", f"{agg['hallucination_rate'] * 100:.1f}%")
            st.metric("Avg Latency", f"{agg['average_latency_seconds']:.2f}s")

        st.markdown("#### Per-Question Benchmark Breakdown")
        bm_df = pd.DataFrame(bm_summary["individual_benchmarks"])
        st.dataframe(bm_df[[
            "benchmark_id", "retrieval_count", "mean_similarity", "mrr",
            "citation_validity", "grounding_score", "hallucination_rate",
            "contradictions_found", "research_gaps_found", "latency_seconds"
        ]], use_container_width=True)
