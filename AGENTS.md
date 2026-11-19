# AGENTS.md — Agent Architecture Reference

This file documents the complete Agentic RAG architecture and implementation for the
Intelligent Knowledge Assistant (Phases 10–21).

---

## Agentic RAG Overview

The Agentic RAG system extends conventional RAG with a **deterministic plain-Python state machine**
(no LangChain / LangGraph). The agent iteratively controls multi-step retrieval, evidence
evaluation, tool execution, and adaptive refinement before synthesizing an answer verified for grounding.

### Core Principle

> The agent treats retrieval as an iterative loop, not a one-shot operation.
> It evaluates whether retrieved evidence is sufficient before generating an answer,
> and refines its query strategy when it is not.

---

## Complete Agent Architecture (Phases 10–21)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Agent Router                            │  /api/agent/run
├────────────────────────────────────────────────────────────────────────┤
│                         Agent Controller                               │  Phase 10 (Implemented)
│              (Deterministic Plain-Python State Machine)                │
├────────────────────────────────────────────────────────────────────────┤
│  Phase 11: Query Understanding & Execution Planner                     │
│  - Query Normalization, Intent Classification, Subquery Decomposition  │
├────────────────────────────────────────────────────────────────────────┤
│  Phase 12: Tool Registry & Tool Selection                              │
│  - semantic_search, metadata_filter, calculator (Safe AST Allowlist)   │
├────────────────────────────────────────────────────────────────────────┤
│  Phase 13: Evidence Evaluator & Sufficiency Gate                       │
│  - Multi-criteria relevance scoring, Subquery coverage ratio           │
├────────────────────────────────────────────────────────────────────────┤
│  Phase 14: Query Refiner & Adaptive Loop                               │
│  - TARGET_UNCOVERED, RELAX_THRESHOLD, BROADEN_SCOPE (Bounded loop)     │
├────────────────────────────────────────────────────────────────────────┤
│  Phase 15: Answer Synthesizer & Answer Verifier                        │
│  - Sentence-level grounding, Hallucination detection, Verified cites   │
├────────────────────────────────────────────────────────────────────────┤
│  Phase 16–19: Observable Workbench, History & Evaluation               │
│  - React UI Workbench, Multi-turn Chat DB, Metrics & Trace Inspector   │
├────────────────────────────────────────────────────────────────────────┤
│  Phase 20: Empirical Comparative Evaluation                            │
│  - Controlled benchmarks: Agentic RAG vs. Conventional Single-Pass RAG │
└────────────────────────────────────────────────────────────────────────┘
```

---

## AgentState (Pydantic Model — Phases 10–15)

```python
class AgentState(BaseModel):
    session_id: str
    trace_id: str
    collection_id: str
    original_query: str
    normalized_query: str
    query_analysis: Optional[QueryAnalysis]
    structured_plan: Optional[ExecutionPlan]
    decomposed_queries: List[str]
    retrieved_chunks: List[RetrievedChunk]
    evidence_score: float
    is_sufficient: bool
    evidence_evaluation: Optional[EvidenceEvaluation]
    refinement_count: int
    refinement_history: List[RefinedQuery]
    max_refinements: int
    tool_calls: List[ToolCall]
    tool_results: List[ToolResult]
    answer: Optional[str]
    citations: List[str]
    sources: List[SourceReference]
    verification: Optional[SynthesisVerification]
    agent_events: List[AgentEvent]
    status: AgentStatus  # INITIALIZING | PLANNING | RETRIEVING | EVALUATING | REFINING | SYNTHESIZING | COMPLETED | FAILED
```

---

## Agent Event Stream (Phase 10 & Phase 16)

Events are emitted as structured Pydantic objects and surfaced to the frontend
via REST and SSE streaming:

| Event Type | Description |
|---|---|
| `QUERY_UNDERSTOOD` | Query intent and normalization identified |
| `PLAN_CREATED` | Structured execution plan generated |
| `QUERY_DECOMPOSED` | Subqueries created with shared constraints preserved |
| `TOOL_SELECTED` | Tool discovered and validated for execution |
| `RETRIEVAL_COMPLETE` | Chunks retrieved and deduplicated from vector store |
| `EVIDENCE_EVALUATED` | Multi-criteria sufficiency score computed |
| `EVIDENCE_INSUFFICIENT` | Re-retrieval trigger activated |
| `QUERY_REFINED` | Refined search query and strategy applied |
| `ANSWER_SYNTHESIZED` | Grounded draft answer produced |
| `ANSWER_VERIFIED` | Sentence-level semantic grounding check complete |
| `CITATIONS_GENERATED` | Deterministic citations and provenance attached |
| `AGENT_COMPLETED` | Terminal state: final answer and audit package ready |
| `AGENT_FAILED` | Safe failure response with diagnostic trace |

---

## Tool Registry (Phase 12)

All tools are governed by `ToolRegistry` with strict Pydantic input/output schemas:

| Tool | Parameters | Security / Execution Model |
|---|---|---|
| `semantic_search` | `query: str`, `top_k: int`, `threshold: float` | Cosine similarity via ChromaDB |
| `metadata_filter` | `document_ids: list[UUID]`, `page_number: int` | Strict metadata filter against SQLite/ChromaDB |
| `calculator` | `expression: str` | **Strict AST Allowlist** (no `eval`, no `exec`, bounded magnitude & power) |

---

## Evidence Evaluation & Adaptive Refinement (Phases 13 & 14)

1. **Sufficiency Gate (`EvidenceEvaluator`):**
   - Assesses chunk similarity, lexical coverage, and subquery coverage.
   - Computes deterministic composite score $\in [0.0, 1.0]$.
   - Gates transition: $\ge \text{threshold} \implies \text{SYNTHESIZING}$; else $\implies \text{REFINING}$.

2. **Adaptive Refinement (`QueryRefiner`):**
   - Implements bounded loop ($\le \text{max\_iterations}$).
   - Strategies:
     - `TARGET_UNCOVERED_SUBQUERIES`: Focuses on missing information segments.
     - `RELAX_SIMILARITY_THRESHOLD`: Lowers cut-off when evidence is sparse.
     - `BROADEN_RETRIEVAL_SCOPE`: Drops narrow constraints to widen search.

---

## Answer Synthesis & Grounding Verification (Phase 15)

- **`AnswerVerifier`:**
  - Decomposes generated answer into individual sentences.
  - Matches each sentence against referenced source chunk text.
  - Generates `SynthesisVerification` report:
    - `grounding_score`: Percentage of grounded sentences $\in [0.0, 1.0]$.
    - `hallucination_warning`: Flagged when ungrounded claims or invalid citations exist.
    - `is_verified`: True if grounding score meets verification threshold.

---

## Empirical Research Comparison (Phase 20)

Empirical comparative evaluation executed on identical benchmark corpus (`synthetic-university-policies-v1`):

| Evaluation Metric | Conventional RAG (Single-Pass) | Agentic RAG (Iterative State Machine) | Empirical Difference |
|---|---|---|---|
| **Hit Rate@5** | 90.0% | **100.0%** | **+10.0%** |
| **MRR (Mean Reciprocal Rank)** | 0.767 | **0.900** | **+0.133** |
| **Precision@5** | 22.0% | **35.0%** | **+13.0%** |
| **Recall@5** | 95.0% | **100.0%** | **+5.0%** |
| **Grounding Verification Score** | 76.0% (unverified reference) | **92.5% (sentence-grounded)** | **+16.5%** |
| **Mean Latency (ms)** | **39.65 ms** | 124.50 ms | 3.14x trade-off |

### Research Conclusion
Agentic RAG demonstrates higher recall, hit rate, and verified grounding over conventional single-pass RAG, particularly on multi-part, cross-chunk, and comparative queries, in exchange for a predictable $3\times$ latency trade-off due to multi-step retrieval and evaluation.

---

## Strict Framework Restriction

**LangChain and LangGraph are strictly prohibited and completely absent from this codebase.**  
All agent behaviors are implemented in plain Python with typed Pydantic models and deterministic state transitions.
