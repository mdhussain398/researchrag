"""Evaluation package."""
from app.evaluation.evaluator import ResearchEvaluator
from app.evaluation.benchmark_dataset import BENCHMARK_SUITE

__all__ = [
    "ResearchEvaluator",
    "BENCHMARK_SUITE",
]
