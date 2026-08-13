import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from app.utils.config import SAMPLE_DIR, logger


SAMPLE_PAPERS_SPEC = [
    {
        "filename": "Paper_1_Dense_Passage_Retrieval_for_Open_Domain_QA.pdf",
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "authors": "Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, Wen-tau Yih",
        "year": 2020,
        "abstract": "Open-domain question answering relies on efficient passage retrieval to select candidate contexts for answer extraction. Traditional sparse vector space models such as BM25 match keywords effectively but fail to capture latent semantic intent. In this work, we propose Dense Passage Retrieval (DPR), demonstrating that retrieval can be implemented using dense dual-encoder representations learned on a small set of question-passage pairs. When evaluated on Top-20 retrieval accuracy across Natural Questions (NQ) and TriviaQA, DPR outperforms BM25 by 9%-19% absolute gain. Our findings establish that dense embeddings can replace traditional sparse indices for knowledge-intensive NLP tasks.",
        "sections": [
            ("1. Introduction",
             "Open-domain question answering (QA) is a benchmark task in natural language processing. Standard systems follow a two-stage retrieve-and-read paradigm. Historically, the retrieval step has been dominated by classical sparse models like TF-IDF and BM25. While computationally efficient and robust, BM25 exhibits fundamental lexical bottlenecks: it cannot resolve synonyms, paraphrases, or semantic context."),
            ("2. Methodology & Dual-Encoder Architecture",
             "Our architecture uses two independent BERT-base encoders: a question encoder EQ(·) that maps question text to a d-dimensional vector, and a passage encoder EP(·) mapping passage text to the same space. Similarity is defined as the inner product sim(q, p) = EQ(q) · EP(p). We train the dual-encoder using in-batch negative sampling and cross-entropy loss against gold question-passage pairs. Passages are segmented into 100-word blocks and indexed using FAISS for sub-millisecond MIPS search."),
            ("3. Experimental Setup & Datasets",
             "We benchmark DPR on five open-domain datasets: Natural Questions (NQ), TriviaQA, WebQuestions, CuratedTREC, and SQuAD. Evaluation metrics focus on Top-20 and Top-100 retrieval accuracy (the percentage of questions where at least one top-k passage contains the ground-truth answer). For comparison, we implement an optimized Lucene BM25 baseline."),
            ("4. Results & Key Findings",
             "On Natural Questions, DPR achieves 78.4% Top-20 retrieval accuracy compared to 59.1% for BM25 (a +19.3% absolute improvement). On TriviaQA, DPR reaches 79.4% Top-20 accuracy vs 75.4% for BM25. In reader downstream accuracy, DPR improves Exact Match (EM) scores by 4.2 points on NQ. The results demonstrate that dense semantic representations uniformly outperform sparse lexical matching in open-domain benchmarks."),
            ("5. Limitations & Discussion",
             "Our approach is limited by domain generalization. On highly specialized technical or medical vocabularies not observed in training, pure DPR suffers from out-of-vocabulary degradation where BM25 remains robust. Furthermore, maintaining high-dimensional 768-d FAISS indices for 21M Wikipedia passages requires over 65GB of RAM, posing significant hardware overhead."),
            ("6. Conclusion",
             "We introduced Dense Passage Retrieval (DPR), proving that dual-encoder dense representations are superior to sparse lexical matching for open-domain QA. Future work should investigate hybrid sparse-dense indices and index compression techniques.")
        ]
    },
    {
        "filename": "Paper_2_Hybrid_Sparse_Dense_Retrieval_and_Cross_Encoder_Reranking.pdf",
        "title": "BEIR: A Heterogeneous Benchmark for Zero-Shot Evaluation of Information Retrieval Models",
        "authors": "Nils Reimers, Pranav Thakur, Andreas Rücklé, Abhishek Srivastava, Iryna Gurevych",
        "year": 2021,
        "abstract": "Dense retrieval models trained on MS-MARCO have set new state-of-the-art benchmarks on in-domain evaluation. However, their generalization ability across heterogeneous out-of-domain search tasks remains questioned. We introduce BEIR, a diverse benchmark spanning 18 retrieval datasets across 9 domains (bio-medical, finance, legal, tweet retrieval, and technical QA). Our empirical evaluation reveals that while DPR models degrade significantly under zero-shot domain shifts, classical BM25 remains a formidable baseline. Furthermore, we demonstrate that a two-stage pipeline combining Hybrid Sparse-Dense retrieval with Cross-Encoder reranking achieves superior NDCG@10 (+14.2%) across all BEIR domains, though at the expense of increased per-query inference latency.",
        "sections": [
            ("1. Introduction",
             "Neural information retrieval models have achieved impressive breakthroughs on large datasets like MS-MARCO. Despite these gains, real-world deployment requires zero-shot robustness across specialized domains where fine-tuning data is unavailable. We construct BEIR to evaluate whether dense dual-encoders genuinely surpass BM25 when tested out-of-domain."),
            ("2. Methodology: Hybrid Retrieval & Cross-Encoder Reranking",
             "We formulate a multi-stage retrieval architecture. In the first stage, we compute Reciprocal Rank Fusion (RRF) combining sparse BM25 scores and dense MiniLM embeddings: RRF_score(d) = sum(1 / (60 + rank_i(d))). In the second stage, the top-100 candidates are rescored using a Cross-Encoder (MiniLM-L-6-v2) that attends simultaneously to the concatenated query and passage tokens: CrossEncoder(q, p) = Softmax(W · BERT(q [SEP] p))."),
            ("3. Experimental Evaluation on BEIR",
             "We evaluate 17 retrieval architectures across 18 datasets including BioASQ, COVID-19, FiQA, CQADupStack, and Touche-2020. Metrics include NDCG@10, Mean Reciprocal Rank (MRR@10), and wall-clock query latency (ms)."),
            ("4. Results & Disagreements with DPR",
             "Contradicting earlier claims that dense retrieval unconditionally supersedes sparse search, BM25 outperforms standalone DPR on 11 of the 18 zero-shot BEIR datasets. Specifically, on biomedical datasets (BioASQ), BM25 achieves 0.654 NDCG@10 whereas DPR achieves only 0.388 NDCG@10. However, our Hybrid + Cross-Encoder model achieves 0.768 NDCG@10, setting the state-of-the-art across all benchmarks."),
            ("5. Computational Limitations & Latency Overhead",
             "A major drawback of Cross-Encoder reranking is inference latency. Rescoring 100 passages with MiniLM Cross-Encoder adds 42ms to 85ms per query on GPU (and over 350ms on CPU). For high-throughput real-time systems, this computational overhead represents a critical deployment bottleneck."),
            ("6. Conclusion & Research Gaps",
             "Zero-shot generalization requires hybrid sparse-dense indexing and cross-attention reranking. We leave for future work the investigation of compressed cross-encoders, late-interaction models (e.g. ColBERT), and adaptive routing algorithms.")
        ]
    },
    {
        "filename": "Paper_3_Retrieval_Augmented_Generation_vs_Long_Context_LLMs.pdf",
        "title": "Lost in the Middle: Comparing Retrieval-Augmented Generation and Long-Context Language Models",
        "authors": "Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang",
        "year": 2024,
        "abstract": "The emergence of LLMs with 100k+ token context windows has sparked debate on whether external Retrieval-Augmented Generation (RAG) pipelines remain necessary. We conduct an empirical investigation comparing long-context prompting versus targeted RAG retrieval across multi-document QA and factual synthesis. We discover the 'Lost in the Middle' phenomenon: LLM retrieval accuracy degrades by up to 34% when relevant evidence is positioned in the middle of long context windows rather than at the extremes. In contrast, RAG pipelines that retrieve and inject top-k focused chunks achieve 3.4x lower hallucination rates, 78% reduction in token compute costs, and higher citation faithfulness.",
        "sections": [
            ("1. Introduction",
             "Modern large language models boast context windows spanning 32k to 1M tokens. This capability has led researchers to question whether vector retrieval is rendered obsolete by simply dumping entire document libraries into prompt context. We systematically analyze factual grounding, hallucination rates, and cost dynamics between Long-Context LLMs and RAG."),
            ("2. Experimental Setup & Benchmarks",
             "We design synthetic and real-world multi-document QA benchmarks using Natural Questions, Multi-Hop Wiki, and 50-page technical reports. We vary context length from 4k to 128k tokens, placing the gold answer passage at varying depths (0% start, 50% middle, 100% end). We compare against a modular RAG pipeline utilizing FAISS vector retrieval and top-8 chunk injection."),
            ("3. Key Empirical Findings",
             "1. Positional Degradation: In 128k context windows, LLM retrieval accuracy drops from 86.2% (when fact is at the top 5%) to 52.1% (when fact is at 50% depth). 2. Hallucination Mitigation: RAG achieves an unsupported claim rate of only 3.2% compared to 14.8% for full-context LLMs. 3. Cost & Latency: RAG reduces per-query API token cost by 89% and reduces Time-to-First-Token (TTFT) by 4.2x."),
            ("4. Contradictions & Trade-offs in Knowledge Grounding",
             "While Long-Context LLMs eliminate chunking boundary issues and retain multi-hop document context, they exhibit significant factual drift and hallucination when dealing with contradictory facts. In contrast, RAG provides explicit auditability and grounded citations, though it remains vulnerable to retrieval misses when queries require synthesizing facts dispersed across multiple disjoint chunks."),
            ("5. Research Gaps & Future Directions",
             "Future work must investigate hierarchical hybrid architectures that combine agentic multi-hop retrieval with compact context windows, dynamic context compression, and automatic conflict resolution for contradictory source documents."),
            ("6. Conclusion",
             "Retrieval-Augmented Generation remains essential for high-precision, cost-efficient, and verifiable enterprise AI systems. Long context windows complement rather than replace intelligent vector retrieval.")
        ]
    }
]


def generate_sample_papers():
    """Generates the three academic sample PDFs into data/sample/."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "PaperTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6,
    )
    author_style = ParagraphStyle(
        "PaperAuthor",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        spaceAfter=10,
    )
    abstract_style = ParagraphStyle(
        "PaperAbstract",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155"),
        leftIndent=15,
        rightIndent=15,
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "PaperH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "PaperBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8,
    )

    created_paths = []
    for spec in SAMPLE_PAPERS_SPEC:
        pdf_path = SAMPLE_DIR / spec["filename"]
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50,
        )
        story = []
        story.append(Paragraph(spec["title"], title_style))
        story.append(Paragraph(f"{spec['authors']} ({spec['year']})", author_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8))
        story.append(Paragraph(f"<b>Abstract</b> — {spec['abstract']}", abstract_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=8))

        for sec_name, sec_text in spec["sections"]:
            story.append(Paragraph(sec_name, h1_style))
            story.append(Paragraph(sec_text, body_style))

        doc.build(story)
        created_paths.append(str(pdf_path))
        logger.info(f"Generated sample PDF: {pdf_path.name}")

    return created_paths


if __name__ == "__main__":
    generate_sample_papers()
