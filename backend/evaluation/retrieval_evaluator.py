"""
backend/evaluation/retrieval_evaluator.py
─────────────────────────────────────────────────────────────
RetrievalEvaluator: Executes sequential quality and latency
evaluations across ground-truth test cases and compiles
comprehensive, structured Pydantic EvaluationReports.
─────────────────────────────────────────────────────────────
"""

import logging
import platform
import sys
import time
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.evaluation.evaluation_dataset import load_evaluation_dataset
from backend.evaluation.evaluation_models import (
    AggregateMetrics,
    EvaluationReport,
    EvaluationTestCase,
    QueryEvaluationResult,
)
from backend.evaluation.retrieval_metrics import (
    calculate_hit_at_k,
    calculate_latency_stats,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
)
from backend.schemas.retrieval import RetrievalQuery, RetrievalResponse
from backend.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    Executes reproducible semantic retrieval evaluations against a validated
    dataset and computes precision, recall, MRR, hit rate, and latency benchmarks.
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        dataset: Optional[List[EvaluationTestCase]] = None,
        session_factory: Optional[Callable[[], AsyncSession]] = None,
        dataset_name: str = "synthetic-university-policies-v1",
    ):
        """
        Initialize the evaluator via dependency injection.

        Args:
            retrieval_service: Existing RetrievalService instance to evaluate.
            dataset: Optional list of EvaluationTestCase objects. Defaults to synthetic dataset.
            session_factory: Optional async sessionmaker for database context.
            dataset_name: Identifier name for the dataset being evaluated.
        """
        self.retrieval_service = retrieval_service
        self.dataset = dataset or load_evaluation_dataset()
        self.session_factory = session_factory
        self.dataset_name = dataset_name

    async def run_evaluation(
        self,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> EvaluationReport:
        """
        Run the complete evaluation suite sequentially and generate an EvaluationReport.

        Args:
            top_k: Cutoff rank for evaluation metrics (default 5).
            similarity_threshold: Optional minimum cosine similarity threshold.

        Returns:
            Structured EvaluationReport Pydantic model.
        """
        if top_k < 1 or top_k > 100:
            raise ValueError(f"top_k must be between 1 and 100, got {top_k}")

        # 1. Warm-up inference check to isolate cold-start model load from query latency
        warmup_latency_ms: Optional[float] = None
        try:
            t_warm_0 = time.perf_counter()
            if hasattr(self.retrieval_service.embedding_service, "embed_text"):
                self.retrieval_service.embedding_service.embed_text("warmup query probe")
            t_warm_1 = time.perf_counter()
            warmup_latency_ms = round((t_warm_1 - t_warm_0) * 1000.0, 2)
        except Exception as warm_exc:
            logger.debug("Warmup probe exception (safe to ignore in mock tests): %s", warm_exc)

        # 2. Sequential execution across all evaluation test cases
        query_results: List[QueryEvaluationResult] = []
        failed_cases: List[Dict[str, Any]] = []
        warnings: List[str] = []

        for tc in self.dataset:
            expected_set = set(tc.expected_chunk_ids)

            # Establish session context if provided
            session: Optional[AsyncSession] = None
            if self.session_factory is not None:
                session = self.session_factory()

            t0 = time.perf_counter()
            error_str: Optional[str] = None
            retrieval_resp: Optional[RetrievalResponse] = None

            try:
                query_obj = RetrievalQuery(
                    query=tc.query,
                    collection_id=UUID(tc.collection_id),
                    top_k=top_k,
                    similarity_threshold=similarity_threshold,
                )
                retrieval_resp = await self.retrieval_service.retrieve(query=query_obj, db=session)
            except Exception as exc:
                error_str = f"{type(exc).__name__}: {str(exc)}"
                logger.warning("Evaluation query failed on %s: %s", tc.case_id, error_str)
                failed_cases.append({"case_id": tc.case_id, "query": tc.query, "error": error_str})
            finally:
                t1 = time.perf_counter()
                if session is not None:
                    await session.close()

            elapsed_ms = round((t1 - t0) * 1000.0, 3)

            # Process outcome
            if retrieval_resp is not None:
                retrieved_ids = [c.chunk_id for c in retrieval_resp.results]
                rel_retrieved = [cid for cid in retrieved_ids[:top_k] if cid in expected_set]

                p_k = calculate_precision_at_k(retrieved_ids, expected_set, k=top_k)
                r_k = calculate_recall_at_k(retrieved_ids, expected_set, k=top_k)
                rr = calculate_reciprocal_rank(retrieved_ids, expected_set, k=top_k)
                hit = calculate_hit_at_k(retrieved_ids, expected_set, k=top_k)

                q_res = QueryEvaluationResult(
                    case_id=tc.case_id,
                    query=tc.query,
                    query_type=tc.query_type,
                    retrieved_chunk_ids=retrieved_ids,
                    expected_chunk_ids=tc.expected_chunk_ids,
                    relevant_retrieved_chunk_ids=rel_retrieved,
                    precision_at_k=round(p_k, 4),
                    recall_at_k=round(r_k, 4),
                    reciprocal_rank=round(rr, 4),
                    hit=hit,
                    latency_ms=elapsed_ms,
                    total_candidates=retrieval_resp.total_candidates,
                    returned_results=retrieval_resp.returned_results,
                    rejected_candidates=retrieval_resp.rejected_candidates,
                    accepted_candidates=retrieval_resp.accepted_candidates,
                    retrieval_metadata=retrieval_resp.retrieval_metadata,
                    error=None,
                )
            else:
                q_res = QueryEvaluationResult(
                    case_id=tc.case_id,
                    query=tc.query,
                    query_type=tc.query_type,
                    retrieved_chunk_ids=[],
                    expected_chunk_ids=tc.expected_chunk_ids,
                    relevant_retrieved_chunk_ids=[],
                    precision_at_k=0.0,
                    recall_at_k=0.0,
                    reciprocal_rank=0.0,
                    hit=False,
                    latency_ms=elapsed_ms,
                    total_candidates=0,
                    returned_results=0,
                    rejected_candidates=0,
                    accepted_candidates=0,
                    retrieval_metadata={},
                    error=error_str,
                )

            query_results.append(q_res)

        # 3. Compute Aggregate Metrics
        total_queries = len(query_results)
        successful_queries = sum(1 for q in query_results if q.error is None)
        failed_count = total_queries - successful_queries

        if total_queries > 0:
            mean_p = sum(q.precision_at_k for q in query_results) / float(total_queries)
            mean_r = sum(q.recall_at_k for q in query_results) / float(total_queries)
            mrr = sum(q.reciprocal_rank for q in query_results) / float(total_queries)
            hit_rate = sum(1 for q in query_results if q.hit) / float(total_queries)
            empty_count = sum(1 for q in query_results if q.returned_results == 0)
            empty_rate = empty_count / float(total_queries)
        else:
            mean_p = mean_r = mrr = hit_rate = empty_rate = 0.0
            empty_count = 0

        latencies = [q.latency_ms for q in query_results]
        avg_lat, med_lat, p95_lat = calculate_latency_stats(latencies)

        total_cands = sum(q.total_candidates for q in query_results)
        rejected_cands = sum(q.rejected_candidates for q in query_results)
        accepted_cands = sum(q.accepted_candidates for q in query_results)

        # Metrics breakdown by query type
        by_type: Dict[str, Dict[str, float]] = {}
        type_groups: Dict[str, List[QueryEvaluationResult]] = {}
        for q in query_results:
            type_groups.setdefault(q.query_type, []).append(q)

        for q_type, q_list in type_groups.items():
            count = len(q_list)
            by_type[q_type] = {
                "count": float(count),
                "precision_at_k": round(sum(item.precision_at_k for item in q_list) / count, 4),
                "recall_at_k": round(sum(item.recall_at_k for item in q_list) / count, 4),
                "mrr": round(sum(item.reciprocal_rank for item in q_list) / count, 4),
                "hit_rate": round(sum(1 for item in q_list if item.hit) / count, 4),
                "avg_latency_ms": round(sum(item.latency_ms for item in q_list) / count, 2),
            }

        aggregate = AggregateMetrics(
            mean_precision_at_k=round(mean_p, 4),
            mean_recall_at_k=round(mean_r, 4),
            mrr=round(mrr, 4),
            hit_rate_at_k=round(hit_rate, 4),
            empty_result_count=empty_count,
            empty_result_rate=round(empty_rate, 4),
            avg_latency_ms=avg_lat,
            median_latency_ms=med_lat,
            p95_latency_ms=p95_lat,
            total_candidates_sum=total_cands,
            rejected_candidates_sum=rejected_cands,
            accepted_candidates_sum=accepted_cands,
            total_queries=total_queries,
            successful_queries=successful_queries,
            failed_queries=failed_count,
            by_query_type=by_type,
        )

        # 4. Hardware Environment Context
        hardware_info = {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }

        if total_queries < 20:
            warnings.append(
                f"Sample size ({total_queries}) is small; 95th percentile latency may represent nearest rank."
            )

        return EvaluationReport(
            dataset_name=self.dataset_name,
            dataset_size=total_queries,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            warmup_latency_ms=warmup_latency_ms,
            query_results=query_results,
            aggregate_metrics=aggregate,
            failed_cases=failed_cases,
            warnings=warnings,
            hardware_info=hardware_info,
        )
