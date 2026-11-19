# Phase 7.5 — Retrieval Evaluation and Quality Assessment Report

**Project:** Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models  
**Phase:** 7.5 — Retrieval Evaluation and Quality Assessment  
**Date:** 2026-11-09  
**Status:** **PASS WITH CONDITIONS — approved.**  
**Advancement:** Phase 8 has not started and requires explicit user approval.  
**Host Environment:** Windows 11, AMD Ryzen 7 7730U with Radeon Graphics (16 vCPUs, ~2.0 GHz), Python 3.13.3, `.venv`  
**Core Dependencies:** `fastapi==0.115.11`, `pydantic==2.14.3`, `chromadb==1.5.9`, `sentence-transformers==5.3.0`, `torch==2.15.0+cpu`  

---

## 1. Executive Summary & Architectural Overview

Phase 7.5 introduces a reproducible, production-isolated evaluation framework to quantify the retrieval quality, latency characteristics, candidate accounting, and ranking performance of the existing semantic retrieval pipeline (`RetrievalService`, `VectorStoreService`, `EmbeddingService`).

### Architectural Constraints Upheld
- **Strictly Plain Python State Machine & Services:** No LangChain, LangGraph, or high-level orchestration abstractions were introduced (per `AGENTS.md`).
- **Singleton Embedding Instance:** Evaluation leverages the shared embedding model lifecycle via `EmbeddingService._cached_model`, preventing redundant memory allocation.
- **Zero Production Storage Mutation:** All evaluation runs execute against isolated in-memory SQLite instances (`sqlite+aiosqlite:///:memory:`) and temporary ChromaDB persistence directories (`eval_chroma_*`) that are automatically torn down post-run.
- **Hardware-Aware Design:** Tailored for CPU-first execution on AMD Ryzen 7 7730U, utilizing sequential execution, explicit cold-start warmup isolation, and monotonic high-resolution timing (`time.perf_counter`).
- **Baseline Scope:** The synthetic dataset of 10 queries across 4 documents (10 chunks) provides a controlled preliminary baseline to verify algorithmic metric math, rank preservation, and pipeline integration. It is not intended for broad general enterprise performance claims.

```
┌────────────────────────────────────────────────────────────────────────┐
│                      RetrievalEvaluator Architecture                   │
├────────────────────────────────────────────────────────────────────────┤
│  EvaluationTestCase (10 Query Types)                                  │
│    │                                                                   │
│    ▼                                                                   │
│  [Warmup Probe] ──> EmbeddingService (Cold-start separation)           │
│    │                                                                   │
│    ▼                                                                   │
│  [Sequential Query Loop]                                               │
│    │──> Monotonic Timer (time.perf_counter)                            │
│    │──> RetrievalService.retrieve(query, db)                           │
│    │      │──> Cosine Search (ChromaDB candidate_k = min(top_k*3, 100))│
│    │      └──> SQLite Batched Candidate Validation                     │
│    │──> IR Metrics Computation (Precision@K, Recall@K, MRR, Hit@K)     │
│    └──> Candidate Accounting (Raw, Accepted, Rejected, Reasons)       │
│    │                                                                   │
│    ▼                                                                   │
│  AggregateMetrics & Breakdown by Query Type                            │
│    │                                                                   │
│    ▼                                                                   │
│  EvaluationReport (Pydantic v2 Schema ──> Terminal UI + JSON Artifact) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Synthetic Dataset & Ground-Truth Annotation

To eliminate data leakage and ensure 100% reproducible ground-truth labels, Phase 7.5 establishes a deterministic academic policy corpus spanning 4 documents and 10 text chunks:

| Document ID | Title | Chunks | Summary of Domain Content |
|---|---|---|---|
| `eval_doc_01` | Academic Integrity and Code of Conduct | 3 | Definitions of plagiarism, unauthorized collaboration, sanction escalation, and disciplinary committee hearing procedures. |
| `eval_doc_02` | Graduate Degree Requirements & Graduation | 2 | Minimum cumulative GPA (3.0), thesis credit requirements, graduation application deadlines, and committee defense timelines. |
| `eval_doc_03` | Campus Housing and Residence Life Regulations | 3 | Residence hall quiet hours, fire safety and prohibited appliances, housing contract cancellation fee ($500) and refund schedule. |
| `eval_doc_04` | Scholarships and Merit Financial Aid Guidelines | 2 | Satisfactory Academic Progress (SAP) credit pace and GPA (2.0), Presidential Research Fellowship stipend ($28,000) and renewal GPA (3.5). |

### Test Cases & Query Categories (10 Query Archetypes)
The dataset defines 10 specialized evaluation queries designed to benchmark dense vector retrieval across different semantic challenges:

1. **Factual (`tc_01_factual_gpa`):** "What is the minimum cumulative GPA required for graduate degree graduation?"  
   *Expected:* `eval_doc_02_v1_c0000` (1 chunk)
2. **Conceptual (`tc_02_conceptual_integrity`):** "How does the university handle academic dishonesty and integrity violations?"  
   *Expected:* `eval_doc_01_v1_c0000`, `eval_doc_01_v1_c0001` (2 chunks)
3. **Definition (`tc_03_definition_plagiarism`):** "What constitutes plagiarism according to the academic honor code?"  
   *Expected:* `eval_doc_01_v1_c0000` (1 chunk)
4. **Multi-Part (`tc_04_multipart_degree_defense`):** "What are the required semester credit hours and the thesis defense committee composition for master's students?"  
   *Expected:* `eval_doc_02_v1_c0000`, `eval_doc_02_v1_c0001` (2 chunks)
5. **Cross-Chunk (`tc_05_crosschunk_sanctions_appeals`):** "What sanctions are imposed for honor code offenses and how can a student file an appeal?"  
   *Expected:* `eval_doc_01_v1_c0001`, `eval_doc_01_v1_c0002` (2 chunks)
6. **Paraphrase (`tc_06_paraphrase_dorm_noise`):** "When are dormitory residents expected to maintain silence during the week?"  
   *Expected:* `eval_doc_03_v1_c0000` (1 chunk)
7. **Limited-Info (`tc_07_limited_info_stipend`):** "How is the Presidential Research Fellowship stipend disbursed and what are the renewal rules?"  
   *Expected:* `eval_doc_04_v1_c0001` (1 chunk)
8. **Unanswerable / Negative (`tc_08_unanswerable_parking`):** "What are the fees for campus vehicle parking permits and shuttle bus timetables?"  
   *Expected:* None (`[]`) — out-of-corpus query testing negative relevance rejection.
9. **Similar-Terms (`tc_09_similar_terms_quorum`):** "What quorum is required for the Academic Appeals Committee to hold a hearing?"  
   *Expected:* `eval_doc_01_v1_c0002` (1 chunk)
10. **Document-Specific (`tc_10_docspecific_housing_refund`):** "Under campus housing regulations, what is the cancellation penalty and refund schedule for terminating a room contract?"  
    *Expected:* `eval_doc_03_v1_c0002` (1 chunk)

---

## 3. Mathematical Metric Definitions

The evaluation engine computes Information Retrieval (IR) standard metrics in `backend/evaluation/retrieval_metrics.py`:

### 1. Precision@K
Fraction of retrieved chunks within the top-$K$ cutoff that are relevant to the query:
$$\text{Precision@}K = \frac{|\text{Retrieved}_{1..K} \cap \text{Relevant}|}{K}$$
- Uses a fixed denominator $K = 5.0$ (standard Information Retrieval penalty for under-retrieval).
- Deduplicates retrieved chunk IDs while strictly preserving the rank order of their first appearance.
- If $0$ chunks are expected (unanswerable query), Precision@K returns $0.0$.

### 2. Recall@K
Fraction of ground-truth relevant chunks that appear within the top-$K$ retrieved results:
$$\text{Recall@}K = \frac{|\text{Retrieved}_{1..K} \cap \text{Relevant}|}{|\text{Relevant}|}$$
- **Negative / Unanswerable Query Convention:** When $|\text{Relevant}| = 0$:
  $$\text{Recall@}K = \begin{cases} 1.0 & \text{if } |\text{Retrieved}_{1..K}| = 0 \text{ (correct corpus rejection)} \\ 0.0 & \text{if } |\text{Retrieved}_{1..K}| > 0 \text{ (penalized for false positives)} \end{cases}$$

### 3. Mean Reciprocal Rank (MRR)
Measures the rank position of the *first* relevant chunk:
$$\text{RR} = \begin{cases} \frac{1}{\text{rank}_1} & \text{if a relevant chunk appears at rank } \le K \\ 0.0 & \text{otherwise} \end{cases}$$
$$\text{MRR} = \frac{1}{N} \sum_{i=1}^N \text{RR}_i$$

### 4. Hit Rate@K
Binary indicator of whether at least one relevant chunk is present in the top-$K$ results:
$$\text{Hit@}K = \begin{cases} 1 & \text{if } |\text{Retrieved}_{1..K} \cap \text{Relevant}| > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 5. Latency Statistics
Monotonically measured using `time.perf_counter()` per query (isolated from cold-start model weight loading):
- **Average Latency:** Arithmetic mean of query durations in milliseconds.
- **Median Latency (P50):** 50th percentile response time.
- **P95 Latency:** 95th percentile response time.

---

## 4. Empirical Evaluation Results ($K = 5$)

Running the automated evaluation runner (`scripts/evaluate_retrieval.py --top-k 5 --output docs/evaluation_report_top5.json`) against the synthetic university policy benchmark produced the following verified empirical metrics:

### 4.1 Aggregate Benchmark Metrics (Overall vs. Answerable-Only)

| Metric | Overall Benchmark ($N=10$, includes unanswerable) | Answerable-Only ($N=9$, excludes `tc_08`) | Interpretation / Notes |
|---|---|---|---|
| **Total Queries** | 10 | 9 | 100% execution success rate (0 failures). |
| **Mean Precision@5** | **0.2400** | **0.2667** | Exact theoretical maximum achievable for fixed $K=5$ given ground-truth cardinality ($|\text{Rel}| \le 2$). |
| **Mean Recall@5** | **0.9000** (90.0%) | **1.0000** (100.0%) | Overall includes `tc_08` as 0.0. Every answerable query retrieved 100% of target chunks. |
| **Mean Reciprocal Rank (MRR)** | **0.9000** | **1.0000** | Every single answerable query placed its primary target chunk at **Rank 1**. |
| **Hit Rate@5** | **0.9000** (90.0%) | **1.0000** (100.0%) | 9 of 9 answerable queries achieved a hit in top-5. |
| **Empty Result Rate** | **0.0000** (0/10) | **0.0000** (0/9) | All queries returned 5 chunks. |
| **Model Warmup Latency** | **22.74 ms** | **22.74 ms** | First query probe inference after model loading. |
| **Average Query Latency** | **39.65 ms** | **40.47 ms** | End-to-end (embedding + vector search + batched SQLite validation). |
| **Median Query Latency** | **36.09 ms** | **36.50 ms** | 50th percentile. |
| **P95 Query Latency** | **60.42 ms** | **60.42 ms** | Represents sample maximum for $N=10$. |
| **Raw Candidates Sum** | **100** | **90** | 10 candidates per query (capped by corpus size of 10). |
| **Accepted Candidates Sum** | **100** | **90** | All 10 candidates verified in SQLite. |
| **Rejected Candidates Sum** | **0** | **0** | Clean evaluation corpus (no stale vectors present). |

### 4.2 Disaggregated Performance by Query Type

| Query Type | Case ID | Precision@5 | Recall@5 | MRR | Hit Rate | Latency (ms) | Ground Truth Chunks |
|---|---|---|---|---|---|---|---|
| **factual** | `tc_01_factual_gpa` | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 46.12 | `eval_doc_02_v1_c0000` (Rank 1) |
| **conceptual** | `tc_02_conceptual_integrity` | 0.4000 | 1.0000 | 1.0000 | 1.0000 | 35.69 | `c0000` (Rank 1), `c0001` (Rank 2) |
| **definition** | `tc_03_definition_plagiarism` | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 40.35 | `eval_doc_01_v1_c0000` (Rank 1) |
| **multi-part** | `tc_04_multipart_degree_defense` | 0.4000 | 1.0000 | 1.0000 | 1.0000 | 60.42 | `eval_doc_02_v1_c0001` (Rank 1), `c0000` (Rank 2) |
| **cross-chunk** | `tc_05_crosschunk_sanctions_appeals`| 0.4000 | 1.0000 | 1.0000 | 1.0000 | 46.84 | `c0001` (Rank 1), `c0002` (Rank 2) |
| **paraphrase** | `tc_06_paraphrase_dorm_noise` | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 35.37 | `eval_doc_03_v1_c0000` (Rank 1) |
| **limited-info** | `tc_07_limited_info_stipend` | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 32.30 | `eval_doc_04_v1_c0001` (Rank 1) |
| **unanswerable** | `tc_08_unanswerable_parking` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 32.27 | None (`[]` — out of corpus) |
| **similar-terms** | `tc_09_similar_terms_quorum` | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 36.50 | `eval_doc_01_v1_c0002` (Rank 1) |
| **doc-specific** | `tc_10_docspecific_housing_refund` | 0.2000 | 1.0000 | 1.0000 | 1.0000 | 30.68 | `eval_doc_03_v1_c0002` (Rank 1) |

---

## 5. Detailed Metric & Operational Investigations

### 5.1 Precision@5 Mathematical Upper Bound Analysis
The reported `mean_precision_at_k` of `0.2400` is the **exact theoretical upper bound** for fixed-denominator Precision@K ($K=5$) on this dataset:
- In this corpus, queries contain at most 1 or 2 relevant chunks:
  - 6 queries have only **1** relevant chunk in the entire corpus ($\text{Max } P = 1/5 = 0.20$).
  - 3 queries have **2** relevant chunks in the entire corpus ($\text{Max } P = 2/5 = 0.40$).
  - 1 query has **0** relevant chunks in the corpus ($\text{Max } P = 0/5 = 0.00$).
- Theoretical maximum:
  $$\text{Precision@5}_{\text{max}} = \frac{(6 \times 0.20) + (3 \times 0.40) + (1 \times 0.00)}{10} = \frac{1.20 + 1.20 + 0.00}{10} = \mathbf{0.2400}$$
- **Clarification:** The score of 0.2400 confirms that the retrieval engine retrieved 100% of all available relevant chunks at the top ranks. However, because $K=5$, the remaining $5 - |\text{Relevant}|$ slots are populated by off-target distractor chunks from the corpus. These remaining chunks are not relevant, highlighting that downstream synthesis components should not treat all $K$ returned chunks as equally informative.

### 5.2 Candidate Oversampling Mechanics
- **Specification:** Phase 7.3 specifies `candidate_k = min(max(top_k * 3, top_k), 100)`.
- **Actual Request:** When `top_k = 5`, `RetrievalService` requests $\mathbf{15}$ candidates ($3\times$ oversampling).
- **Corpus Limitation:** The synthetic university policy dataset contains exactly **10 chunks** ($3 + 2 + 3 + 2 = 10$). ChromaDB cannot return more vectors than exist in the collection.
- **Finding:** ChromaDB returned all 10 available chunks per query. The total raw candidates sum of `100` reflects $10 \text{ queries} \times 10 \text{ available chunks}$, bounded by the corpus size, not by a $2\times$ oversampling ratio. Phase 7.3 implementation remains fully compliant with the $3\times$ specification.

### 5.3 Unanswerable Queries & Negative Retrieval
- For `tc_08_unanswerable_parking` (campus parking fees), the query has no relevant content in the corpus (`expected_chunk_ids = []`).
- Because dense vector search without a similarity threshold (`similarity_threshold = None`) always returns the nearest $K$ vectors in Euclidean/cosine space, the engine returned 5 chunks with similarity scores ranging from `0.5000` to `0.6276`.
- Because irrelevant chunks were returned, Recall@5 evaluated to `0.0000` and Precision@5 evaluated to `0.0000`.
- **Architectural Prerequisite for Phase 13 (Evidence Evaluator):** This empirical finding demonstrates that unassisted dense retrieval cannot reject unanswerable queries on its own. Phase 13 must implement an explicit evidence sufficiency evaluation gate to inspect candidate similarity scores, filter out low-confidence context, and trigger query refinement (Phase 14) or graceful refusal when retrieved evidence is insufficient.

### 5.4 Benchmark Coverage vs. Adverse Database Testing
- **Current Benchmark Corpus:** The evaluation seeder (`backend/evaluation/evaluation_dataset.py`) populates only clean, active, version-1 documents with status `INDEXED`. Consequently, all 100 candidates were accepted (`accepted_candidates_sum = 100`, `rejected_candidates_sum = 0`).
- **Unit Test Coverage:** Phase 7.3 unit tests ([`tests/test_retrieval_service.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/tests/test_retrieval_service.py), 22 tests) comprehensively verify that stale versions, deleted documents (`DELETION_PENDING`), non-indexed statuses (`PENDING`, `PROCESSING`, `FAILED`), wrong collections, and invalid metadata are properly rejected by SQLite validation.
- **Coverage Improvement:** Future evaluation phases should introduce an adverse benchmark dataset containing deliberately planted stale vectors and desynchronized records to quantitatively test the candidate rejection accounting within the evaluation report itself.

---

## 6. Test Suite Verification & Coverage Evidence

### 6.1 Unit & Integration Test Results (Phase 7.5 Modules)
```powershell
.\.venv\Scripts\python.exe -m pytest tests/evaluation/ -v
```
- `tests/evaluation/test_retrieval_metrics.py`: **26 PASSED** (0.12s)
  - Precision math, edge cases, zero division, duplicate chunk IDs, cutoff truncation.
  - Recall math, zero division, negative unanswerable queries (0 expected).
  - MRR math, first rank, deep rank, missing rank, empty retrieved set.
  - Hit rate math, presence vs cutoff boundary.
  - Latency stats math, empty sets, single values, P95 calculations.
- `tests/evaluation/test_retrieval_evaluator.py`: **6 PASSED** (28.58s)
  - Dataset schema & query type verification (10 distinct types).
  - Pydantic validation on `EvaluationTestCase` and invalid values.
  - Deterministic evaluation using mock `RetrievalService`.
  - Error isolation and partial failure handling (graceful capture in report).
  - Report JSON serialization and disk writing.
  - End-to-end integration test with isolated SQLite & ChromaDB storage.
- **Evaluation Total: 32 tests passed (100% pass rate).**

### 6.2 Full Regression Suite (Phase 7 Pipeline)
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_retrieval_router.py tests/test_retrieval_service.py tests/test_vector_store.py tests/test_document_processor.py -v
```
- `tests/test_retrieval_router.py`: **18 PASSED** (Phase 7.4 REST endpoints)
- `tests/test_retrieval_service.py`: **22 PASSED** (Phase 7.3 SQLite-validated retrieval service)
- `tests/test_vector_store.py`: **42 PASSED** (Phase 7.2 vector search & storage operations)
- `tests/test_document_processor.py`: **20 PASSED** (Phase 6 document ingestion pipeline)
- **Regression Total: 102 tests passed (100% pass rate).**

### 6.3 Dependency Verification
```powershell
.\.venv\Scripts\python.exe -m pip check
```
- **Result:** `No broken requirements found.`

---

## 7. Safety, Production Isolation & Integrity Audit

Production stores were inspected before and after evaluation execution:
- `data/knowledge_assistant.db`: 0 collections, 0 documents, `PRAGMA integrity_check = ok`.
- `data/chromadb/chroma.sqlite3`: 0 collections, 0 embeddings, `PRAGMA integrity_check = ok`.
- `data/uploads/`: 0 files.
- **Verdict:** Zero production data modified. All evaluation artifacts reside strictly in `docs/` or isolated temporary storage.

---

## 8. Phase Acceptance Checklist & Final Status

| Item | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Pydantic v2 Evaluation Schemas | **VERIFIED** | `backend/evaluation/evaluation_models.py` |
| 2 | Metric Mathematical Implementations | **VERIFIED** | `backend/evaluation/retrieval_metrics.py` (26 unit tests passed) |
| 3 | Synthetic Academic Policy Corpus (10 Chunks) | **VERIFIED** | `backend/evaluation/evaluation_dataset.py` |
| 4 | 10 Distinct Query Archetypes | **VERIFIED** | All 10 query types covered and evaluated |
| 5 | Evaluator Class with Warmup Separation | **VERIFIED** | `backend/evaluation/retrieval_evaluator.py` |
| 6 | CLI Runner with Summary & JSON Export | **VERIFIED** | `scripts/evaluate_retrieval.py` (`docs/evaluation_report_top5.json`) |
| 7 | Zero LangChain / LangGraph Dependencies | **VERIFIED** | Pure Python state machine architecture |
| 8 | Production Data Store Isolation | **VERIFIED** | Zero records written to SQLite or ChromaDB production storage |
| 9 | Full Regression Suite Passing | **VERIFIED** | 102/102 regression tests passed |
| 10 | Git Clean State Maintained | **VERIFIED** | No unauthorized commits or pushes |

---

## 9. Final Phase Status

**Phase 7.5: PASS WITH CONDITIONS — approved.**

**Phase 8 has not started and requires explicit user approval.**
