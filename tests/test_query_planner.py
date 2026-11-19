"""
tests/test_query_planner.py
─────────────────────────────────────────────────────────────
Deterministic unit tests for Phase 11 Query Planner Service:
  - Single-pass execution plan generation for simple queries.
  - Multi-subquery plan generation with linked subquery IDs.
  - Correct plan step types, order, and descriptions.
  - ExecutionPlan schema compliance and determinism.
─────────────────────────────────────────────────────────────
"""

from backend.schemas.query_plan import (
    PlanStepType,
    QueryAnalysis,
    QueryIntent,
    SubQuery,
)
from backend.services.query_planner import QueryPlannerService
from backend.services.query_understanding import QueryUnderstandingService


def test_planner_simple_query():
    planner = QueryPlannerService()
    sq = SubQuery(
        id="sq-simple-1",
        parent_query="What is the admission policy?",
        text="What is the admission policy?",
        intent=QueryIntent.FACTUAL,
        order=0,
    )
    analysis = QueryAnalysis(
        original_query="What is the admission policy?",
        normalized_query="What is the admission policy?",
        primary_intent=QueryIntent.FACTUAL,
        is_complex=False,
        complexity_factors=[],
        subqueries=[sq],
    )

    plan = planner.create_plan(analysis)

    assert plan.is_decomposed is False
    assert plan.subquery_count == 1
    assert len(plan.steps) == 4

    # Verify step types in order
    assert plan.steps[0].step_type == PlanStepType.RETRIEVAL
    assert plan.steps[0].subquery_id == "sq-simple-1"
    assert plan.steps[1].step_type == PlanStepType.CONTEXT_ASSEMBLY
    assert plan.steps[2].step_type == PlanStepType.EVIDENCE_EVALUATION
    assert plan.steps[3].step_type == PlanStepType.SYNTHESIS

    # Sequential numbering
    for idx, step in enumerate(plan.steps, start=1):
        assert step.step_number == idx


def test_planner_multi_query_decomposition():
    planner = QueryPlannerService()
    sq1 = SubQuery(
        id="sq-part-1",
        parent_query="Compare X and Y",
        text="Overview of X",
        intent=QueryIntent.FACTUAL,
        order=0,
    )
    sq2 = SubQuery(
        id="sq-part-2",
        parent_query="Compare X and Y",
        text="Overview of Y",
        intent=QueryIntent.FACTUAL,
        order=1,
    )
    analysis = QueryAnalysis(
        original_query="Compare X and Y",
        normalized_query="Compare X and Y",
        primary_intent=QueryIntent.COMPARATIVE,
        is_complex=True,
        complexity_factors=["comparative_entity_split"],
        subqueries=[sq1, sq2],
    )

    plan = planner.create_plan(analysis)

    assert plan.is_decomposed is True
    assert plan.subquery_count == 2
    # 2 retrieval steps + 1 deduplication/context + 1 evaluation + 1 synthesis = 5 steps
    assert len(plan.steps) == 5

    # Check that each retrieval step is mapped to its subquery ID
    assert plan.steps[0].step_type == PlanStepType.RETRIEVAL
    assert plan.steps[0].subquery_id == "sq-part-1"
    assert plan.steps[1].step_type == PlanStepType.RETRIEVAL
    assert plan.steps[1].subquery_id == "sq-part-2"

    assert plan.steps[2].step_type == PlanStepType.CONTEXT_ASSEMBLY
    assert plan.steps[3].step_type == PlanStepType.EVIDENCE_EVALUATION
    assert plan.steps[4].step_type == PlanStepType.SYNTHESIS

    for idx, step in enumerate(plan.steps, start=1):
        assert step.step_number == idx


def test_end_to_end_understanding_to_plan():
    understanding_svc = QueryUnderstandingService()
    planner_svc = QueryPlannerService()

    query = "What was the operating revenue in Q3 2025, and what were the major capital expenditures?"
    analysis = understanding_svc.analyze(query)
    plan = planner_svc.create_plan(analysis)

    assert analysis.is_complex is True
    assert plan.is_decomposed is True
    assert len(analysis.subqueries) == 2
    assert plan.subquery_count == 2
    assert len(plan.steps) == 5
    assert plan.steps[0].subquery_id == analysis.subqueries[0].id
    assert plan.steps[1].subquery_id == analysis.subqueries[1].id

    # Verify shared constraint preservation in plan steps
    assert analysis.subqueries[0].text == "What was the operating revenue in Q3 2025?"
    assert analysis.subqueries[1].text == "What were the major capital expenditures in Q3 2025?"
    assert "in Q3 2025" in plan.steps[0].description
    assert "in Q3 2025" in plan.steps[1].description


def test_planner_comparative_objective_in_plan():
    """Verify Issue 2: Comparative objective propagated to plan and synthesis step."""
    understanding_svc = QueryUnderstandingService()
    planner_svc = QueryPlannerService()

    query = "Compare SQLite and ChromaDB."
    analysis = understanding_svc.analyze(query)
    plan = planner_svc.create_plan(analysis)

    assert plan.is_decomposed is True
    assert plan.comparison_objective == "Compare SQLite and ChromaDB"
    assert "Compare SQLite and ChromaDB" in plan.plan_summary
    synthesis_step = plan.steps[-1]
    assert synthesis_step.step_type == PlanStepType.SYNTHESIS
    assert synthesis_step.name == "Comparative Answer Synthesis"
    assert "Compare SQLite and ChromaDB" in synthesis_step.description
