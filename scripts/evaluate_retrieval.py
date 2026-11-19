#!/usr/bin/env python
"""
scripts/evaluate_retrieval.py
─────────────────────────────────────────────────────────────
CLI command runner for reproducible Semantic Retrieval Evaluation.

Features:
  - Runs isolated evaluation against synthetic ground-truth dataset.
  - Calculates Precision@K, Recall@K, MRR, Hit Rate@K, and Latency stats.
  - Outputs a human-readable terminal table summary.
  - Exports a comprehensive, structured JSON report.
  - Zero modification to production database or ChromaDB storage.

Usage:
  python scripts/evaluate_retrieval.py --help
  python scripts/evaluate_retrieval.py --top-k 5
  python scripts/evaluate_retrieval.py --top-k 3 --threshold 0.20 --output data/eval_report.json
─────────────────────────────────────────────────────────────
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import tempfile

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import Settings
from backend.database import setup_sqlite_foreign_keys
from backend.evaluation.evaluation_dataset import load_evaluation_dataset, seed_evaluation_dataset
from backend.evaluation.retrieval_evaluator import RetrievalEvaluator
from backend.models.base import Base
from backend.services.embedding_service import EmbeddingService
from backend.services.retrieval_service import RetrievalService
from backend.services.vector_store import VectorStoreService


async def run_evaluation_cli(
    top_k: int = 5,
    similarity_threshold: float | None = None,
    dataset_name: str = "synthetic-university-policies-v1",
    output_path: Path | None = None,
    isolated: bool = True,
) -> int:
    """Run retrieval evaluation from CLI parameters."""
    print("=" * 70)
    print("  SEMANTIC RETRIEVAL EVALUATION & QUALITY ASSESSMENT (Phase 7.5)")
    print("=" * 70)
    print(f"Dataset      : {dataset_name}")
    print(f"Top K        : {top_k}")
    print(f"Threshold    : {similarity_threshold}")
    print(f"Isolated Env : {isolated}")
    print("-" * 70)

    if not isolated:
        print("[ERROR] Non-isolated evaluation against production data is forbidden to preserve data integrity.")
        return 1

    temp_chroma_dir = tempfile.mkdtemp(prefix="eval_chroma_")
    db_url = "sqlite+aiosqlite:///:memory:"

    try:
        # 1. Setup isolated database and vector store
        print("[1/4] Initializing isolated evaluation environment...")
        eval_settings = Settings(
            database_url=db_url,
            chroma_persist_dir=temp_chroma_dir,
            log_level="WARNING",
        )

        engine = create_async_engine(db_url, connect_args={"check_same_thread": False}, future=True)
        setup_sqlite_foreign_keys(engine)

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

        vector_store = VectorStoreService(persist_dir=temp_chroma_dir, settings=eval_settings)
        embedding_service = EmbeddingService(settings=eval_settings)

        # 2. Seed synthetic dataset
        print("[2/4] Seeding controlled evaluation corpus into isolated storage...")
        async with session_factory() as session:
            collection_id = await seed_evaluation_dataset(
                db=session,
                vector_store=vector_store,
                embedding_service=embedding_service,
            )

        dataset = load_evaluation_dataset(collection_id=collection_id)
        print(f"      Seeded {len(dataset)} evaluation test cases into collection '{collection_id}'.")

        # 3. Execute evaluation
        print("[3/4] Executing sequential evaluation queries...")
        retrieval_service = RetrievalService(
            embedding_service=embedding_service,
            vector_store=vector_store,
            session_factory=session_factory,
            settings=eval_settings,
        )

        evaluator = RetrievalEvaluator(
            retrieval_service=retrieval_service,
            dataset=dataset,
            session_factory=session_factory,
            dataset_name=dataset_name,
        )

        report = await evaluator.run_evaluation(top_k=top_k, similarity_threshold=similarity_threshold)

        # 4. Print Summary
        agg = report.aggregate_metrics
        print("\n" + "=" * 70)
        print("  RETRIEVAL EVALUATION RESULTS SUMMARY")
        print("=" * 70)
        print(f"Total Queries Evaluated : {agg.total_queries}")
        print(f"Successful Queries       : {agg.successful_queries}")
        print(f"Failed Queries           : {agg.failed_queries}")
        print(f"Mean Precision@{top_k:<2}       : {agg.mean_precision_at_k:.4f}")
        print(f"Mean Recall@{top_k:<2}          : {agg.mean_recall_at_k:.4f}")
        print(f"Mean Reciprocal Rank     : {agg.mrr:.4f} (MRR)")
        print(f"Hit Rate@{top_k:<2}             : {agg.hit_rate_at_k:.4f} ({agg.hit_rate_at_k * 100:.1f}%)")
        print(f"Empty Result Rate        : {agg.empty_result_rate:.4f} ({agg.empty_result_count}/{agg.total_queries})")
        print("-" * 70)
        print(f"Latency Avg / Median / P95: {agg.avg_latency_ms:.2f} ms / {agg.median_latency_ms:.2f} ms / {agg.p95_latency_ms:.2f} ms")
        if report.warmup_latency_ms is not None:
            print(f"Model Warmup Latency     : {report.warmup_latency_ms:.2f} ms")
        print("-" * 70)
        print(f"Candidate Totals (Sum)   : {agg.total_candidates_sum} raw, {agg.accepted_candidates_sum} accepted, {agg.rejected_candidates_sum} rejected")
        print("=" * 70)

        # Breakdown by query type
        print("\n  BREAKDOWN BY QUERY TYPE:")
        print(f"  {'Query Type':<16} | {'Count':<5} | {'Prec@K':<8} | {'Recall':<8} | {'MRR':<8} | {'Hit%':<8} | {'Avg Lat':<8}")
        print("  " + "-" * 68)
        for q_type, metrics in agg.by_query_type.items():
            print(
                f"  {q_type:<16} | {int(metrics['count']):<5} | "
                f"{metrics['precision_at_k']:<8.4f} | {metrics['recall_at_k']:<8.4f} | "
                f"{metrics['mrr']:<8.4f} | {metrics['hit_rate']:<8.4f} | {metrics['avg_latency_ms']:<8.2f}"
            )
        print("  " + "-" * 68)

        # Save JSON report
        if output_path is None:
            now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            output_path = PROJECT_ROOT / "data" / "evaluation_reports" / f"retrieval_eval_{now_str}.json"

        saved_file = report.save_json(output_path)
        print(f"\n[4/4] Full JSON evaluation report saved to:\n      {saved_file.resolve()}\n")

        await engine.dispose()
        return 0

    except Exception as exc:
        print(f"\n[FATAL ERROR] Evaluation failed: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 2

    finally:
        # Cleanup temporary ChromaDB directory
        shutil.rmtree(temp_chroma_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run Semantic Retrieval Evaluation and Quality Assessment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum candidate results cutoff for Precision@K and Recall@K.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional cosine similarity threshold in [-1.0, 1.0].",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic-university-policies-v1",
        help="Name identifier of evaluation dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for structured JSON evaluation report.",
    )
    parser.add_argument(
        "--no-isolated",
        action="store_true",
        help="Disable isolated temporary environment (FORBIDDEN for production safety).",
    )

    args = parser.parse_args()
    isolated = not args.no_isolated

    exit_code = asyncio.run(
        run_evaluation_cli(
            top_k=args.top_k,
            similarity_threshold=args.threshold,
            dataset_name=args.dataset,
            output_path=args.output,
            isolated=isolated,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
