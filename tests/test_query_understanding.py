"""
tests/test_query_understanding.py
─────────────────────────────────────────────────────────────
Comprehensive unit tests for Phase 11 Query Understanding:
  - Query normalization & conversational preamble stripping.
  - Preservation of domain terminology, acronyms, and punctuation.
  - Deterministic intent classification (FACTUAL, COMPARATIVE,
    MULTI_PART, SUMMARIZATION, EXPLORATORY).
  - Complexity detection and multi-part decomposition.
  - Protection of common noun-conjunction idioms ("research and development").
  - Subquery parent_query provenance verification.
─────────────────────────────────────────────────────────────
"""

import pytest

from backend.schemas.query_plan import QueryIntent
from backend.services.query_understanding import (
    IntentClassifier,
    QueryDecomposer,
    QueryNormalizer,
    QueryUnderstandingService,
)


# ── Normalization Tests ──────────────────────────────────────────────────────


def test_normalize_whitespace_and_preambles():
    # 1. Collapse irregular whitespace
    raw = "   What   is    the    Transformer    architecture?   "
    assert QueryNormalizer.normalize(raw) == "What is the Transformer architecture?"

    # 2. Conversational preamble stripping
    p1 = "Can you please tell me what is the quarterly revenue?"
    assert QueryNormalizer.normalize(p1) == "what is the quarterly revenue?"

    p2 = "Could you explain how distributed consensus works?"
    assert QueryNormalizer.normalize(p2) == "how distributed consensus works?"

    p3 = "Please tell me about the admission criteria for 2025"
    assert QueryNormalizer.normalize(p3) == "the admission criteria for 2025"

    p4 = "I would like to know what were the total expenses."
    assert QueryNormalizer.normalize(p4) == "what were the total expenses."

    # 3. Collapse multiple question marks
    p5 = "Is the model quantized????"
    assert QueryNormalizer.normalize(p5) == "Is the model quantized?"


def test_normalize_preserves_technical_terms():
    # Technical IDs, acronyms, and punctuation preserved
    query = "How does BAAI/bge-small-en-v1.5 compare with text-embedding-3-small on MTEB?"
    normalized = QueryNormalizer.normalize(query)
    assert "BAAI/bge-small-en-v1.5" in normalized
    assert "text-embedding-3-small" in normalized
    assert "MTEB" in normalized


# ── Intent Classification Tests ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "query, expected_intent",
    [
        ("What is the capital expenditure for Q3 2025?", QueryIntent.FACTUAL),
        ("Who is the chief financial officer?", QueryIntent.FACTUAL),
        ("When did the acquisition complete?", QueryIntent.FACTUAL),
        ("Compare SQLite with ChromaDB for local vector storage", QueryIntent.COMPARATIVE),
        ("What is the difference between BERT and GPT-3?", QueryIntent.COMPARATIVE),
        ("What are the pros and cons of microservices versus monoliths?", QueryIntent.COMPARATIVE),
        ("Summarize the executive findings in the annual report", QueryIntent.SUMMARIZATION),
        ("Provide a brief overview of the safety guidelines", QueryIntent.SUMMARIZATION),
        ("What are the key takeaways from the meeting transcript?", QueryIntent.SUMMARIZATION),
        ("How does the self-attention mechanism operate?", QueryIntent.EXPLORATORY),
        ("Why does gradient descent require learning rate scheduling?", QueryIntent.EXPLORATORY),
        ("Explain how the SQLite validation gate functions", QueryIntent.EXPLORATORY),
    ],
)
def test_intent_classification(query: str, expected_intent: QueryIntent):
    assert IntentClassifier.classify(query) == expected_intent


# ── Decomposition & Complexity Tests ─────────────────────────────────────────


def test_simple_query_non_decomposition():
    svc = QueryUnderstandingService()
    query = "What is the tuition fee for the engineering program?"
    analysis = svc.analyze(query)

    assert analysis.original_query == query
    assert analysis.primary_intent == QueryIntent.FACTUAL
    assert analysis.is_complex is False
    assert len(analysis.subqueries) == 1
    assert analysis.subqueries[0].parent_query == query
    assert "tuition fee" in analysis.subqueries[0].text


def test_comparative_query_decomposition():
    svc = QueryUnderstandingService()
    query = "Compare SQLite and ChromaDB"
    analysis = svc.analyze(query)

    assert analysis.original_query == query
    assert analysis.primary_intent == QueryIntent.COMPARATIVE
    assert analysis.is_complex is True
    assert len(analysis.subqueries) == 2

    # Check that both entities have generated subqueries
    sub_texts = [sq.text for sq in analysis.subqueries]
    assert any("SQLite" in t for t in sub_texts)
    assert any("ChromaDB" in t for t in sub_texts)

    # Check provenance
    for sq in analysis.subqueries:
        assert sq.parent_query == query


def test_multi_part_query_decomposition():
    svc = QueryUnderstandingService()
    query = "What was the operating revenue in Q3 2025, and what were the major capital expenditures?"
    analysis = svc.analyze(query)

    assert analysis.is_complex is True
    assert len(analysis.subqueries) == 2
    assert "operating revenue" in analysis.subqueries[0].text
    assert "capital expenditures" in analysis.subqueries[1].text
    # Explicit verification of Issue 1: Shared reporting period preservation
    assert analysis.subqueries[0].text == "What was the operating revenue in Q3 2025?"
    assert analysis.subqueries[1].text == "What were the major capital expenditures in Q3 2025?"
    for sq in analysis.subqueries:
        assert sq.parent_query == query


def test_shared_quarter_and_year_preservation():
    """Verify reliable preservation of financial quarters, reporting periods, and years."""
    svc = QueryUnderstandingService()

    # 1. Financial Quarter + Year
    q1 = "What was the operating revenue in Q3 2025, and what were the major capital expenditures?"
    a1 = svc.analyze(q1)
    assert len(a1.subqueries) == 2
    assert a1.subqueries[0].text == "What was the operating revenue in Q3 2025?"
    assert a1.subqueries[1].text == "What were the major capital expenditures in Q3 2025?"

    # 2. Year preservation
    q2 = "What was the net profit in 2024, and what were the total revenues?"
    a2 = svc.analyze(q2)
    assert len(a2.subqueries) == 2
    assert a2.subqueries[0].text == "What was the net profit in 2024?"
    assert a2.subqueries[1].text == "What were the total revenues in 2024?"

    # 3. Fiscal Year preservation
    q3 = "What were the capital investments during fiscal year 2023, and what was the return on equity?"
    a3 = svc.analyze(q3)
    assert len(a3.subqueries) == 2
    assert "fiscal year 2023" in a3.subqueries[0].text
    assert "fiscal year 2023" in a3.subqueries[1].text

    # 4. Backward propagation from second clause to first clause
    q4 = "What was the operating revenue, and what were the major capital expenditures in Q3 2025?"
    a4 = svc.analyze(q4)
    assert len(a4.subqueries) == 2
    assert "in Q3 2025" in a4.subqueries[0].text
    assert "in Q3 2025" in a4.subqueries[1].text


def test_shared_location_preservation():
    """Verify reliable preservation of geographic and spatial constraints."""
    svc = QueryUnderstandingService()

    # 1. Specific city location
    q1 = "What were the hiring numbers in Berlin, and what was the average salary?"
    a1 = svc.analyze(q1)
    assert len(a1.subqueries) == 2
    assert a1.subqueries[0].text == "What were the hiring numbers in Berlin?"
    assert a1.subqueries[1].text == "What was the average salary in Berlin?"

    # 2. Regional location
    q2 = "What is the market penetration across Europe, and what are the primary regulatory hurdles?"
    a2 = svc.analyze(q2)
    assert len(a2.subqueries) == 2
    assert a2.subqueries[0].text == "What is the market penetration across Europe?"
    assert a2.subqueries[1].text == "What are the primary regulatory hurdles across Europe?"


def test_shared_named_entity_context():
    """Verify reliable preservation of named entities and pronoun resolution."""
    svc = QueryUnderstandingService()

    # 1. Named entity with pronoun replacement (its -> Acme Corp's)
    q1 = "What products does Acme Corp manufacture, and who are its primary competitors?"
    a1 = svc.analyze(q1)
    assert len(a1.subqueries) == 2
    assert a1.subqueries[0].text == "What products does Acme Corp manufacture?"
    assert a1.subqueries[1].text == "Who are Acme Corp's primary competitors?"

    # 2. Named entity with prepositional attachment
    q2 = "What is the revenue of Apple, and what was the operating margin?"
    a2 = svc.analyze(q2)
    assert len(a2.subqueries) == 2
    assert a2.subqueries[0].text == "What is the revenue of Apple?"
    assert a2.subqueries[1].text == "What was the operating margin of Apple?"


def test_shared_condition_preservation():
    """Verify preservation of regulatory, compliance, and conditional criteria."""
    svc = QueryUnderstandingService()

    q = "What are the compliance requirements under GDPR, and what are the penalties for violations?"
    a = svc.analyze(q)
    assert len(a.subqueries) == 2
    assert a.subqueries[0].text == "What are the compliance requirements under GDPR?"
    assert a.subqueries[1].text == "What are the penalties for violations under GDPR?"


def test_distinct_constraints_not_overwritten():
    """Ensure subqueries with distinct constraints retain their own independent modifiers."""
    svc = QueryUnderstandingService()

    q = "What was the operating revenue in Q3 2025, and what was the operating revenue in Q4 2025?"
    a = svc.analyze(q)
    assert len(a.subqueries) == 2
    assert a.subqueries[0].text == "What was the operating revenue in Q3 2025?"
    assert a.subqueries[1].text == "What was the operating revenue in Q4 2025?"


def test_comparative_objective_preservation_in_analysis():
    """Verify Issue 2: Comparative objective preserved in QueryAnalysis."""
    svc = QueryUnderstandingService()

    query = "Compare SQLite and ChromaDB."
    analysis = svc.analyze(query)

    assert analysis.primary_intent == QueryIntent.COMPARATIVE
    assert analysis.is_complex is True
    assert analysis.comparison_objective == "Compare SQLite and ChromaDB"
    assert len(analysis.subqueries) == 2
    assert any("SQLite" in sq.text for sq in analysis.subqueries)
    assert any("ChromaDB" in sq.text for sq in analysis.subqueries)
    for sq in analysis.subqueries:
        assert sq.parent_query == query


def test_parent_query_provenance_preserved_across_all_decompositions():
    """Verify that every generated SubQuery maintains strict parent_query provenance."""
    svc = QueryUnderstandingService()
    test_queries = [
        "What was the operating revenue in Q3 2025, and what were the major capital expenditures?",
        "Compare SQLite and ChromaDB",
        "What is the retention rate? How is customer churn calculated?",
        "What is the tuition fee for the engineering program?",
    ]
    for q in test_queries:
        analysis = svc.analyze(q)
        for sq in analysis.subqueries:
            assert sq.parent_query == q


def test_multiple_question_sentences_decomposition():
    svc = QueryUnderstandingService()
    query = "What is the retention rate? How is customer churn calculated?"
    analysis = svc.analyze(query)

    assert analysis.is_complex is True
    assert len(analysis.subqueries) == 2
    assert "retention rate" in analysis.subqueries[0].text
    assert "churn" in analysis.subqueries[1].text


# ── Protected Conjunction Tests (Ambiguous 'and') ────────────────────────────


@pytest.mark.parametrize(
    "query, protected_phrase",
    [
        ("What was the total expenditure on research and development?", "research and development"),
        ("What are the safety regulations in the oil and gas sector?", "oil and gas"),
        ("Explain the merger and acquisition policy.", "merger and acquisition"),
        ("How does the system balance hardware and software latency?", "hardware and software"),
        ("Summarize the quarterly profit and loss statement.", "profit and loss"),
        ("What are the best tips and tricks for prompt engineering?", "tips and tricks"),
    ],
)
def test_protected_and_phrases_not_split(query: str, protected_phrase: str):
    svc = QueryUnderstandingService()
    analysis = svc.analyze(query)

    # Protected phrases must NOT cause spurious decomposition into fragments
    assert len(analysis.subqueries) == 1
    assert analysis.is_complex is False
    assert analysis.subqueries[0].parent_query == query
    assert protected_phrase in analysis.subqueries[0].text.lower()
