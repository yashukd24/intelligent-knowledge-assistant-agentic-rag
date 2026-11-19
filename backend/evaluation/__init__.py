"""
backend/evaluation
─────────────────────────────────────────────────────────────
Semantic Retrieval Evaluation and Quality Assessment Module.
Provides metrics, synthetic datasets, evaluator runners, and
structured evaluation reporting.
─────────────────────────────────────────────────────────────
"""

from backend.evaluation.evaluation_models import (
    AggregateMetrics,
    EvaluationReport,
    EvaluationTestCase,
    QueryEvaluationResult,
)
from backend.evaluation.retrieval_evaluator import RetrievalEvaluator
from backend.evaluation.retrieval_metrics import (
    calculate_hit_at_k,
    calculate_latency_stats,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)

__all__ = [
    "AggregateMetrics",
    "EvaluationReport",
    "EvaluationTestCase",
    "QueryEvaluationResult",
    "RetrievalEvaluator",
    "calculate_hit_at_k",
    "calculate_latency_stats",
    "calculate_precision_at_k",
    "calculate_recall_at_k",
    "calculate_reciprocal_rank",
]
