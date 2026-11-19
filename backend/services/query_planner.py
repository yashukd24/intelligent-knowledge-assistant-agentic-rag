"""
backend/services/query_planner.py
─────────────────────────────────────────────────────────────
Query Planning Service (Phase 11).

Converts a structured QueryAnalysis into an ordered, executable
ExecutionPlan containing discrete PlanStep objects associated with
specific SubQuery IDs.
─────────────────────────────────────────────────────────────
"""

from typing import List

from backend.schemas.query_plan import (
    ExecutionPlan,
    PlanStep,
    PlanStepType,
    QueryAnalysis,
)


class QueryPlannerService:
    """
    Builds structured, deterministic execution plans for Agentic RAG.
    """

    def create_plan(self, analysis: QueryAnalysis) -> ExecutionPlan:
        """
        Generate an ExecutionPlan based on query analysis.

        - For single-requirement queries: Generates an optimal 4-stage sequential plan.
        - For decomposed multi-requirement queries: Generates dedicated retrieval
          steps for each subquery followed by deduplication, assembly, evaluation,
          and synthesis.
        """
        steps: List[PlanStep] = []
        subqueries = analysis.subqueries
        step_counter = 1
        comparison_obj = analysis.comparison_objective

        if not analysis.is_complex or len(subqueries) <= 1:
            # Single-pass execution plan
            target_sq = subqueries[0] if subqueries else None
            sq_id = target_sq.id if target_sq else None
            sq_text = target_sq.text if target_sq else analysis.normalized_query

            steps.append(
                PlanStep(
                    step_number=step_counter,
                    step_type=PlanStepType.RETRIEVAL,
                    name="Targeted Semantic Retrieval",
                    subquery_id=sq_id,
                    description=f"Execute vector search for: '{sq_text}'",
                )
            )
            step_counter += 1

            steps.append(
                PlanStep(
                    step_number=step_counter,
                    step_type=PlanStepType.CONTEXT_ASSEMBLY,
                    name="Context Assembly & Budgeting",
                    subquery_id=None,
                    description="Format retrieved candidate chunks with deterministic [S1], [S2] source citations within character budget.",
                )
            )
            step_counter += 1

            steps.append(
                PlanStep(
                    step_number=step_counter,
                    step_type=PlanStepType.EVIDENCE_EVALUATION,
                    name="Evidence Sufficiency Gate",
                    subquery_id=None,
                    description="Evaluate candidate similarity against threshold; trigger refusal if zero relevant evidence is found.",
                )
            )
            step_counter += 1

            if comparison_obj:
                synthesis_name = "Comparative Answer Synthesis"
                synthesis_desc = (
                    f"Synthesize a grounded comparative analysis addressing: '{comparison_obj}' "
                    "with deterministic source citations."
                )
                plan_summary = f"Comparative analysis plan for: '{comparison_obj}'."
            else:
                synthesis_name = "Grounded Answer Synthesis"
                synthesis_desc = "Generate grounded answer via LLM and validate source citation integrity."
                plan_summary = f"Single-pass retrieval for factual/direct query ('{sq_text[:40]}...')."

            steps.append(
                PlanStep(
                    step_number=step_counter,
                    step_type=PlanStepType.SYNTHESIS,
                    name=synthesis_name,
                    subquery_id=None,
                    description=synthesis_desc,
                )
            )

            is_decomposed = False
        else:
            # Multi-subquery execution plan
            for idx, sq in enumerate(subqueries, start=1):
                steps.append(
                    PlanStep(
                        step_number=step_counter,
                        step_type=PlanStepType.RETRIEVAL,
                        name=f"Subquery Retrieval {idx}",
                        subquery_id=sq.id,
                        description=f"Execute targeted retrieval for subquery {idx}: '{sq.text}'",
                    )
                )
                step_counter += 1

            steps.append(
                PlanStep(
                    step_number=step_counter,
                    step_type=PlanStepType.CONTEXT_ASSEMBLY,
                    name="Multi-Query Chunk Deduplication & Context Assembly",
                    subquery_id=None,
                    description=(
                        f"Deduplicate candidate chunks across all {len(subqueries)} subqueries by chunk_id, "
                        "retain highest cosine similarity score for duplicates, and format combined context."
                    ),
                )
            )
            step_counter += 1

            steps.append(
                PlanStep(
                    step_number=step_counter,
                    step_type=PlanStepType.EVIDENCE_EVALUATION,
                    name="Cross-Facet Evidence Evaluation",
                    subquery_id=None,
                    description=f"Assess joint evidence sufficiency across all {len(subqueries)} subquery facets.",
                )
            )
            step_counter += 1

            if comparison_obj:
                synthesis_name = "Comparative Answer Synthesis"
                synthesis_desc = (
                    f"Synthesize a grounded comparative analysis addressing: '{comparison_obj}' "
                    f"evaluating trade-offs, differences, and characteristics across both entities with deterministic citations."
                )
                plan_summary = (
                    f"Comparative decomposition plan: Retrieve evidence for each entity to fulfill comparison objective "
                    f"('{comparison_obj}') with unified chunk deduplication and synthesis."
                )
            else:
                synthesis_name = "Comprehensive Multi-Facet Answer Synthesis"
                synthesis_desc = (
                    f"Synthesize unified, grounded answer addressing all {len(subqueries)} information requirements "
                    "with deterministic citations."
                )
                plan_summary = (
                    f"Multi-query decomposition plan: {len(subqueries)} subqueries executed sequentially "
                    "with unified chunk deduplication and synthesis."
                )

            steps.append(
                PlanStep(
                    step_number=step_counter,
                    step_type=PlanStepType.SYNTHESIS,
                    name=synthesis_name,
                    subquery_id=None,
                    description=synthesis_desc,
                )
            )

            is_decomposed = True

        return ExecutionPlan(
            original_query=analysis.original_query,
            steps=steps,
            is_decomposed=is_decomposed,
            subquery_count=len(subqueries),
            plan_summary=plan_summary,
            comparison_objective=comparison_obj,
        )
