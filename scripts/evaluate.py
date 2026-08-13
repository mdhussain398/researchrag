"""
Quantitative evaluation benchmark runner for ResearchRAG.
Evaluates retrieval relevance, citation grounding, source coverage, and contradiction detection.
"""

import sys
import json
import time
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from app.utils.config import SAMPLE_DIR, logger
from app.ingestion.manager import IngestionManager
from app.retrieval.vector_store import FaissVectorStore
from app.retrieval.retriever import EvidenceRetriever
from app.generation.report_generator import ResearchReportGenerator
from app.evaluation.evaluator import ResearchEvaluator
from app.evaluation.benchmark_dataset import BENCHMARK_SUITE
from app.models.schemas import ResearchConfig


def run_benchmark_evaluation(output_json: bool = True):
    """Executes the benchmark evaluation suite across sample papers."""
    print("=" * 80)
    print("ResearchRAG: Automated Quantitative Benchmark Evaluation")
    print("=" * 80)

    # 1. Ingest sample papers
    ingestion_mgr = IngestionManager()
    sample_files = list(SAMPLE_DIR.glob("*.pdf"))
    if not sample_files:
        from scripts.generate_sample_papers import generate_sample_papers
        sample_files = [Path(p) for p in generate_sample_papers()]

    print(f"\n[1/4] Ingesting {len(sample_files)} sample papers...")
    ingest_result = ingestion_mgr.process_files([str(f) for f in sample_files])
    docs = ingestion_mgr.get_all_documents()
    chunks = ingestion_mgr.get_all_chunks()
    print(f"Ingested {len(docs)} documents ({len(chunks)} total chunks).")

    # 2. Vector indexing
    print("\n[2/4] Indexing chunks in FAISS Vector Store...")
    vector_store = FaissVectorStore(index_name="benchmark_index")
    vector_store.clear()
    vector_store.add_chunks(chunks)

    retriever = EvidenceRetriever(vector_store=vector_store)
    report_gen = ResearchReportGenerator()
    evaluator = ResearchEvaluator()

    benchmark_results = []

    print("\n[3/4] Running Benchmark Suite across 4 Research Questions...")
    print("-" * 80)

    for i, bm in enumerate(BENCHMARK_SUITE, 1):
        q = bm["question"]
        print(f"\nBenchmark #{i}: {q}")

        config = ResearchConfig(
            research_question=q,
            sub_objectives=bm["expected_subtopics"],
            top_k_chunks=8,
            similarity_threshold=0.15,
            enable_reranking=True,
            llm_provider="local",  # Test using offline deterministic synthesizer for reproducible local scoring
        )

        start_time = time.time()
        # Retrieve
        retrieved = retriever.retrieve_evidence(config)
        # Generate Report
        report = report_gen.generate_report(
            config=config,
            retrieved_chunks=retrieved,
            documents=docs,
            chunks_by_doc=ingestion_mgr.chunks_by_doc,
        )
        elapsed = time.time() - start_time

        # Evaluate
        metrics = evaluator.evaluate_report(report, docs, execution_time=elapsed)

        # Check expected contradiction detection
        detected_topics = [c.topic for c in report.contradictions]
        contradiction_hit = any(bm["expected_contradiction_topic"].lower() in t.lower() for t in detected_topics)

        res = {
            "benchmark_id": bm["benchmark_id"],
            "question": q,
            "retrieval_count": metrics.retrieval_count,
            "mean_similarity": metrics.mean_similarity_score,
            "mrr": metrics.retrieval_mrr,
            "precision_at_k": metrics.retrieval_precision_at_k,
            "recall_at_k": metrics.retrieval_recall_at_k,
            "source_coverage": metrics.source_coverage_ratio,
            "citation_validity": metrics.citation_validity_rate,
            "grounding_score": metrics.semantic_grounding_score,
            "hallucination_rate": metrics.hallucination_claim_rate,
            "contradictions_found": len(report.contradictions),
            "contradiction_target_hit": contradiction_hit,
            "research_gaps_found": len(report.research_gaps),
            "latency_seconds": metrics.execution_time_seconds,
        }
        benchmark_results.append(res)

        print(f"  -> Retr Count: {res['retrieval_count']} | Sim: {res['mean_similarity']:.3f} | MRR: {res['mrr']:.3f} | P@K: {res['precision_at_k']:.3f}")
        print(f"  -> Citation Validity: {res['citation_validity']*100:.1f}% | Grounding: {res['grounding_score']:.3f} | Hallucination: {res['hallucination_rate']*100:.1f}%")
        print(f"  -> Contradictions: {res['contradictions_found']} (Target Match: {contradiction_hit}) | Gaps: {res['research_gaps_found']} | Latency: {res['latency_seconds']:.2f}s")

    # Aggregate Summary
    print("\n" + "=" * 80)
    print("BENCHMARK AGGREGATE SUMMARY")
    print("=" * 80)
    avg_mrr = float(np.mean([r["mrr"] for r in benchmark_results]))
    avg_p_at_k = float(np.mean([r["precision_at_k"] for r in benchmark_results]))
    avg_recall = float(np.mean([r["recall_at_k"] for r in benchmark_results]))
    avg_coverage = float(np.mean([r["source_coverage"] for r in benchmark_results]))
    avg_validity = float(np.mean([r["citation_validity"] for r in benchmark_results]))
    avg_grounding = float(np.mean([r["grounding_score"] for r in benchmark_results]))
    avg_hallucination = float(np.mean([r["hallucination_rate"] for r in benchmark_results]))
    avg_latency = float(np.mean([r["latency_seconds"] for r in benchmark_results]))

    print(f"Mean Reciprocal Rank (MRR):          {avg_mrr:.4f}")
    print(f"Retrieval Precision@K:                {avg_p_at_k * 100:.1f}%")
    print(f"Retrieval Recall@K:                   {avg_recall * 100:.1f}%")
    print(f"Source Document Coverage:             {avg_coverage * 100:.1f}%")
    print(f"Citation Validity Rate:               {avg_validity * 100:.1f}%")
    print(f"Semantic Grounding Score:             {avg_grounding:.4f}")
    print(f"Hallucination / Unsupported Rate:     {avg_hallucination * 100:.1f}%")
    print(f"Average Report Generation Latency:    {avg_latency:.2f}s")
    print("=" * 80)

    summary_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aggregate_metrics": {
            "mean_reciprocal_rank": round(avg_mrr, 4),
            "retrieval_precision_at_k": round(avg_p_at_k, 4),
            "retrieval_recall_at_k": round(avg_recall, 4),
            "source_document_coverage": round(avg_coverage, 4),
            "citation_validity_rate": round(avg_validity, 4),
            "semantic_grounding_score": round(avg_grounding, 4),
            "hallucination_rate": round(avg_hallucination, 4),
            "average_latency_seconds": round(avg_latency, 2),
        },
        "individual_benchmarks": benchmark_results,
    }

    if output_json:
        eval_json_path = ROOT_DIR / "data" / "processed" / "evaluation_results.json"
        with open(eval_json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)
        print(f"\nSaved evaluation results to {eval_json_path}")

    return summary_payload


if __name__ == "__main__":
    run_benchmark_evaluation()
