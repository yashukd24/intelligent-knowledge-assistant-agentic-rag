"""
backend/evaluation/retrieval_metrics.py
─────────────────────────────────────────────────────────────
Pure mathematical metric calculators for retrieval evaluation:
  - Precision@K
  - Recall@K
  - Reciprocal Rank & Mean Reciprocal Rank (MRR)
  - Hit Rate@K
  - Latency statistics (mean, median, p95)
─────────────────────────────────────────────────────────────
"""

import math
import statistics
from typing import Optional, Sequence, Set, Tuple


def calculate_precision_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Set[str],
    k: int = 5,
) -> float:
    """
    Calculate Precision@K:
        Precision@K = (Relevant retrieved chunks in top K) / K

    Rules:
      - K must be >= 1.
      - If retrieved_chunk_ids is empty, Precision is 0.0.
      - If fewer than K chunks are returned, the denominator remains K
        (standard Information Retrieval budget penalty for under-retrieval).
      - Duplicate retrieved chunk IDs are deduplicated preserving first occurrence rank.

    Args:
        retrieved_chunk_ids: Sequence of retrieved chunk IDs in ranking order.
        relevant_chunk_ids: Set of ground-truth relevant chunk IDs.
        k: Maximum rank cutoff (default 5).

    Returns:
        Precision score in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")

    if not retrieved_chunk_ids or not relevant_chunk_ids:
        return 0.0

    # Deduplicate while preserving ranking order
    seen: Set[str] = set()
    deduped_retrieved: list[str] = []
    for cid in retrieved_chunk_ids:
        if cid not in seen:
            seen.add(cid)
            deduped_retrieved.append(cid)

    # Slice top K
    top_k_chunks = deduped_retrieved[:k]

    # Count relevant retrieved
    relevant_count = sum(1 for cid in top_k_chunks if cid in relevant_chunk_ids)

    return relevant_count / float(k)


def calculate_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Set[str],
    k: int = 5,
) -> float:
    """
    Calculate Recall@K:
        Recall@K = (Relevant retrieved chunks in top K) / (Total expected relevant chunks)

    Rules:
      - K must be >= 1.
      - If relevant_chunk_ids is empty (e.g. negative/unanswerable query):
          Returns 1.0 if retrieved_chunk_ids is empty (correct rejection),
          else 0.0 (penalty for false positives).
      - If retrieved_chunk_ids is empty and relevant_chunk_ids is non-empty: returns 0.0.
      - Slices retrieved chunks to top K before computing intersection.

    Args:
        retrieved_chunk_ids: Sequence of retrieved chunk IDs in ranking order.
        relevant_chunk_ids: Set of ground-truth relevant chunk IDs.
        k: Maximum rank cutoff (default 5).

    Returns:
        Recall score in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")

    # Handling unanswerable / negative test queries
    if not relevant_chunk_ids:
        return 1.0 if not retrieved_chunk_ids else 0.0

    if not retrieved_chunk_ids:
        return 0.0

    # Deduplicate while preserving ranking order
    seen: Set[str] = set()
    deduped_retrieved: list[str] = []
    for cid in retrieved_chunk_ids:
        if cid not in seen:
            seen.add(cid)
            deduped_retrieved.append(cid)

    top_k_chunks = deduped_retrieved[:k]
    relevant_count = sum(1 for cid in top_k_chunks if cid in relevant_chunk_ids)

    return min(1.0, relevant_count / float(len(relevant_chunk_ids)))


def calculate_reciprocal_rank(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Set[str],
    k: int = 5,
) -> float:
    """
    Calculate Reciprocal Rank (RR):
        RR = 1.0 / (1-based rank of the FIRST relevant chunk in top K)

    Rules:
      - If no relevant chunk appears in top K, RR = 0.0.
      - If top result is relevant (rank 1), RR = 1.0.
      - If 2nd result is relevant, RR = 0.5, etc.

    Args:
        retrieved_chunk_ids: Sequence of retrieved chunk IDs in ranking order.
        relevant_chunk_ids: Set of ground-truth relevant chunk IDs.
        k: Maximum rank cutoff (default 5).

    Returns:
        Reciprocal rank float in [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")

    if not retrieved_chunk_ids or not relevant_chunk_ids:
        return 0.0

    # Deduplicate while preserving ranking order
    seen: Set[str] = set()
    rank = 1
    for cid in retrieved_chunk_ids:
        if rank > k:
            break
        if cid not in seen:
            seen.add(cid)
            if cid in relevant_chunk_ids:
                return 1.0 / float(rank)
            rank += 1

    return 0.0


def calculate_hit_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Set[str],
    k: int = 5,
) -> bool:
    """
    Calculate Hit@K:
        True if at least one relevant chunk is present in top K, else False.

    Args:
        retrieved_chunk_ids: Sequence of retrieved chunk IDs in ranking order.
        relevant_chunk_ids: Set of ground-truth relevant chunk IDs.
        k: Maximum rank cutoff (default 5).

    Returns:
        Boolean indicating hit presence.
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")

    if not retrieved_chunk_ids or not relevant_chunk_ids:
        return False

    seen: Set[str] = set()
    count = 0
    for cid in retrieved_chunk_ids:
        if count >= k:
            break
        if cid not in seen:
            seen.add(cid)
            count += 1
            if cid in relevant_chunk_ids:
                return True

    return False


def calculate_latency_stats(latencies_ms: Sequence[float]) -> Tuple[float, float, float]:
    """
    Calculate average, median, and 95th percentile latency from query durations.

    Args:
        latencies_ms: Sequence of measured latency values in milliseconds.

    Returns:
        Tuple of (avg_ms, median_ms, p95_ms), each rounded to 2 decimal places.
    """
    if not latencies_ms:
        return 0.0, 0.0, 0.0

    cleaned = [float(x) for x in latencies_ms if x >= 0.0]
    if not cleaned:
        return 0.0, 0.0, 0.0

    avg_ms = statistics.mean(cleaned)
    median_ms = statistics.median(cleaned)

    sorted_lats = sorted(cleaned)
    n = len(sorted_lats)
    p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
    p95_ms = sorted_lats[max(0, p95_idx)]

    return round(avg_ms, 2), round(median_ms, 2), round(p95_ms, 2)
