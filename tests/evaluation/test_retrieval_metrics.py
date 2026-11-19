"""
tests/evaluation/test_retrieval_metrics.py
─────────────────────────────────────────────────────────────
Comprehensive unit tests for mathematical metric calculators:
  - Precision@K
  - Recall@K
  - Reciprocal Rank
  - Hit Rate@K
  - Latency statistics (mean, median, p95)
─────────────────────────────────────────────────────────────
"""

import pytest

from backend.evaluation.retrieval_metrics import (
    calculate_hit_at_k,
    calculate_latency_stats,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)


# ── Precision@K Tests ────────────────────────────────────────────────────────


def test_precision_perfect_retrieval():
    """All retrieved chunks in top K are relevant."""
    retrieved = ["c1", "c2", "c3"]
    relevant = {"c1", "c2", "c3", "c4"}
    assert calculate_precision_at_k(retrieved, relevant, k=3) == 1.0


def test_precision_partial_retrieval():
    """2 out of 5 slots in top 5 are relevant."""
    retrieved = ["c1", "x1", "c2", "x2", "x3"]
    relevant = {"c1", "c2", "c3"}
    assert calculate_precision_at_k(retrieved, relevant, k=5) == 2.0 / 5.0


def test_precision_zero_retrieval():
    """None of the retrieved chunks are relevant."""
    retrieved = ["x1", "x2", "x3"]
    relevant = {"c1", "c2"}
    assert calculate_precision_at_k(retrieved, relevant, k=3) == 0.0


def test_precision_empty_retrieved():
    """Empty retrieved list yields 0.0 precision."""
    assert calculate_precision_at_k([], {"c1"}, k=5) == 0.0


def test_precision_empty_relevant():
    """No relevant chunks in ground truth yields 0.0 precision."""
    assert calculate_precision_at_k(["c1", "c2"], set(), k=5) == 0.0


def test_precision_fewer_than_k_retrieved():
    """Fewer results than budget K penalizes denominator (standard IR convention)."""
    retrieved = ["c1", "c2"]
    relevant = {"c1", "c2"}
    # 2 relevant retrieved out of budget K=5 -> 2/5 = 0.4
    assert calculate_precision_at_k(retrieved, relevant, k=5) == 0.4


def test_precision_duplicate_chunk_ids():
    """Duplicate candidate chunk IDs are deduplicated preserving rank."""
    retrieved = ["c1", "c1", "x1", "c2"]
    relevant = {"c1", "c2"}
    # Deduped to ["c1", "x1", "c2"], top 3 has c1 and c2 -> 2/3
    assert calculate_precision_at_k(retrieved, relevant, k=3) == pytest.approx(2.0 / 3.0)


def test_precision_invalid_k():
    """k <= 0 raises ValueError."""
    with pytest.raises(ValueError):
        calculate_precision_at_k(["c1"], {"c1"}, k=0)


# ── Recall@K Tests ───────────────────────────────────────────────────────────


def test_recall_perfect_retrieval():
    """All expected relevant chunks are retrieved in top K."""
    retrieved = ["c1", "x1", "c2", "x2", "c3"]
    relevant = {"c1", "c2", "c3"}
    assert calculate_recall_at_k(retrieved, relevant, k=5) == 1.0


def test_recall_partial_retrieval():
    """1 out of 2 expected relevant chunks retrieved in top 3."""
    retrieved = ["c1", "x1", "x2"]
    relevant = {"c1", "c2"}
    assert calculate_recall_at_k(retrieved, relevant, k=3) == 0.5


def test_recall_zero_retrieval():
    """No expected relevant chunks retrieved in top K."""
    retrieved = ["x1", "x2", "x3"]
    relevant = {"c1", "c2"}
    assert calculate_recall_at_k(retrieved, relevant, k=3) == 0.0


def test_recall_unanswerable_query_empty_retrieved():
    """Negative/unanswerable query with 0 retrieved items yields 1.0 recall."""
    assert calculate_recall_at_k([], set(), k=5) == 1.0


def test_recall_unanswerable_query_false_positives():
    """Negative/unanswerable query where model retrieved items yields 0.0 recall."""
    assert calculate_recall_at_k(["x1"], set(), k=5) == 0.0


def test_recall_truncation_to_k():
    """Relevant chunk beyond cutoff K does not contribute to Recall@K."""
    retrieved = ["x1", "x2", "x3", "c1"]
    relevant = {"c1"}
    # Cutoff at 3 misses c1 at rank 4
    assert calculate_recall_at_k(retrieved, relevant, k=3) == 0.0
    # Cutoff at 4 catches c1
    assert calculate_recall_at_k(retrieved, relevant, k=4) == 1.0


# ── Reciprocal Rank Tests ───────────────────────────────────────────────────


def test_reciprocal_rank_first_position():
    """First result is relevant -> RR = 1.0."""
    retrieved = ["c1", "x1", "x2"]
    relevant = {"c1"}
    assert calculate_reciprocal_rank(retrieved, relevant, k=3) == 1.0


def test_reciprocal_rank_second_position():
    """Second result is relevant -> RR = 0.5."""
    retrieved = ["x1", "c1", "x2"]
    relevant = {"c1"}
    assert calculate_reciprocal_rank(retrieved, relevant, k=3) == 0.5


def test_reciprocal_rank_fourth_position():
    """Fourth result is relevant -> RR = 0.25."""
    retrieved = ["x1", "x2", "x3", "c1"]
    relevant = {"c1"}
    assert calculate_reciprocal_rank(retrieved, relevant, k=5) == 0.25


def test_reciprocal_rank_multiple_relevant():
    """First relevant result determines RR; later relevant results ignored."""
    retrieved = ["x1", "c1", "c2", "c3"]
    relevant = {"c1", "c2", "c3"}
    assert calculate_reciprocal_rank(retrieved, relevant, k=5) == 0.5


def test_reciprocal_rank_not_found():
    """No relevant chunk in top K -> RR = 0.0."""
    retrieved = ["x1", "x2", "x3"]
    relevant = {"c1"}
    assert calculate_reciprocal_rank(retrieved, relevant, k=3) == 0.0


def test_reciprocal_rank_empty():
    """Empty results -> RR = 0.0."""
    assert calculate_reciprocal_rank([], {"c1"}, k=5) == 0.0


# ── Hit Rate@K Tests ────────────────────────────────────────────────────────


def test_hit_rate_present_in_top_k():
    """Relevant item in top K yields Hit = True."""
    retrieved = ["x1", "c1", "x2"]
    relevant = {"c1"}
    assert calculate_hit_at_k(retrieved, relevant, k=3) is True


def test_hit_rate_not_present():
    """No relevant item in top K yields Hit = False."""
    retrieved = ["x1", "x2", "x3"]
    relevant = {"c1"}
    assert calculate_hit_at_k(retrieved, relevant, k=3) is False


def test_hit_rate_beyond_cutoff_k():
    """Relevant item at rank K+1 yields Hit = False for cutoff K."""
    retrieved = ["x1", "x2", "c1"]
    relevant = {"c1"}
    assert calculate_hit_at_k(retrieved, relevant, k=2) is False
    assert calculate_hit_at_k(retrieved, relevant, k=3) is True


# ── Latency Statistics Tests ────────────────────────────────────────────────


def test_latency_stats_normal():
    """Validates mean, median, and p95 calculations."""
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
    avg_ms, med_ms, p95_ms = calculate_latency_stats(latencies)
    assert avg_ms == 30.0
    assert med_ms == 30.0
    assert p95_ms == 50.0


def test_latency_stats_empty():
    """Empty latency list returns (0.0, 0.0, 0.0)."""
    assert calculate_latency_stats([]) == (0.0, 0.0, 0.0)


def test_latency_stats_single():
    """Single latency value returns that value across all three stats."""
    avg_ms, med_ms, p95_ms = calculate_latency_stats([12.5])
    assert avg_ms == 12.5
    assert med_ms == 12.5
    assert p95_ms == 12.5
