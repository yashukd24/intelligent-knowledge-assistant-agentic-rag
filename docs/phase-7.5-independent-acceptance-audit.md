# Phase 7.5 — Independent Retrieval Evaluation Acceptance Audit

**Project:** Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models  
**Audit Target:** Phase 7.5 — Retrieval Evaluation and Quality Assessment  
**Auditor:** Independent Technical Quality Auditor  
**Date:** 2026-11-09  
**Audit Type:** Strictly READ-ONLY Source & Results Verification  
**Final Audit Verdict:** **PASS WITH CONDITIONS**  

---

## 1. Executive Summary & Audit Scope

This audit performs an independent, read-only technical review of the Phase 7.5 retrieval evaluation implementation, mathematical metric calculations, dataset ground truth, empirical results, and system performance benchmarks.

### Inspected Code and Artifacts
- `backend/evaluation/evaluation_models.py`
- `backend/evaluation/evaluation_dataset.py`
- `backend/evaluation/retrieval_metrics.py`
- `backend/evaluation/retrieval_evaluator.py`
- `scripts/evaluate_retrieval.py`
- `tests/evaluation/test_retrieval_metrics.py`
- `tests/evaluation/test_retrieval_evaluator.py`
- `docs/phase-7.5-retrieval-evaluation-report.md`
- `docs/evaluation_report_top5.json`
- `backend/services/retrieval_service.py`
- `backend/services/vector_store.py`

### Summary of Audit Findings

| ID | Category | Severity | Description | Status |
|---|---|---|---|---|
| **F-01** | Metric Math | **Informational** | Precision@5 (0.2400) reflects the theoretical maximum possible score given ground-truth cardinality ($|\text{Rel}| \le 2$ when $K=5$). | Verified mathematically |
| **F-02** | Metric Aggregation | **Informational** | Overall Recall@5 (0.9000) and MRR (0.9000) include the unanswerable query as 0.0. Answerable-only Recall and MRR are 1.0000. | Disaggregated and verified |
| **F-03** | Documentation | **Low** | Implementation report narrative misattributed 10 candidates to "$2\times$ oversampling". In code, $3\times$ was requested and capped at 10 by total corpus size. | Clarified |
| **F-04** | Out-of-Domain Retrieval | **Informational** | Dense retrieval without threshold returned irrelevant chunks for the negative query, confirming the architectural need for Phase 13 evidence sufficiency gating. | Analyzed |
| **F-05** | Benchmark Coverage | **Medium** | Evaluation dataset contains only clean, valid documents; adverse database rejection accounting is not exercised during the benchmark run. | Documented as coverage gap |
| **F-06** | Latency Measurement | **Informational** | P95 latency on $N=10$ samples represents the sample maximum; appropriate warning emitted in report schema. | Verified |

---

## 2. Section 1 — Metric Correctness Audit

### 2.1 Precision@K
- **Implementation:** [`backend/evaluation/retrieval_metrics.py#L18-L63`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/evaluation/retrieval_metrics.py#L18-L63)
- **Formula:** $\text{Precision@}K = \frac{|\text{Retrieved}_{1..K} \cap \text{Relevant}|}{K}$
- **Denominator:** Fixed $K = 5.0$. Standard Information Retrieval (IR) convention penalizes systems that retrieve fewer relevant items than the requested budget.
- **Duplicate Handling:** Lines 49–55 deduplicate retrieved chunk IDs while strictly preserving the rank order of their first occurrence.
- **Empty / Under-Retrieval:**
  - If `retrieved_chunk_ids` is empty, returns `0.0`.
  - If fewer than $K$ chunks are returned, denominator remains $K$.
  - If `relevant_chunk_ids` is empty (unanswerable query), returns `0.0`.

### 2.2 Recall@K
- **Implementation:** [`backend/evaluation/retrieval_metrics.py#L65-L112`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/evaluation/retrieval_metrics.py#L65-L112)
- **Formula:** $\text{Recall@}K = \frac{|\text{Retrieved}_{1..K} \cap \text{Relevant}|}{|\text{Relevant}|}$
- **Unanswerable / Negative Query Edge Case:**
  - When $|\text{Relevant}| = 0$:
    $$\text{Recall@}K = \begin{cases} 1.0 & \text{if } |\text{Retrieved}_{1..K}| = 0 \text{ (correct corpus rejection)} \\ 0.0 & \text{if } |\text{Retrieved}_{1..K}| > 0 \text{ (penalized for false positives)} \end{cases}$$
  - For `tc_08_unanswerable_parking`, 5 irrelevant chunks were returned, so Recall@5 correctly evaluated to `0.0`.

### 2.3 Reciprocal Rank (MRR)
- **Implementation:** [`backend/evaluation/retrieval_metrics.py#L114-L155`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC%20AND%20LLM%20BOT/backend/evaluation/retrieval_metrics.py#L114-L155)
- **Formula:** $\text{RR} = \frac{1}{\text{rank}_1}$ of the first relevant chunk within top $K$; $0.0$ if no relevant chunk is retrieved.
- **Ranking Performance on Answerable Queries:**
  - For all 9 answerable queries (`tc_01`, `tc_02`, `tc_03`, `tc_04`, `tc_05`, `tc_06`, `tc_07`, `tc_09`, `tc_10`), the primary target chunk was retrieved at **Rank 1**.
  - Reciprocal Rank for all 9 answerable queries = $1.0000$.
  - Reciprocal Rank for `tc_08` = $0.0000$.

### 2.4 Aggregate Metric Analysis: Overall vs Answerable-Only

The audit confirms that the reported overall metrics **include the unanswerable query (`tc_08`) as a 0.0**:

$$\text{Overall Mean Recall@5} = \frac{1.0 \times 9 + 0.0 \times 1}{10} = \frac{9.0}{10} = \mathbf{0.9000}$$
$$\text{Overall MRR} = \frac{1.0 \times 9 + 0.0 \times 1}{10} = \frac{9.0}{10} = \mathbf{0.9000}$$
$$\text{Overall Hit Rate@5} = \frac{1 \times 9 + 0 \times 1}{10} = \frac{9}{10} = \mathbf{0.9000} \text{ (90.0\%)}$$

When disaggregated to evaluate **answerable queries only** ($N=9$):

$$\text{Answerable Recall@5} = \frac{9.0}{9} = \mathbf{1.0000} \text{ (100.0\%)}$$
$$\text{Answerable MRR} = \frac{9.0}{9} = \mathbf{1.0000}$$
$$\text{Answerable Hit Rate@5} = \frac{9}{9} = \mathbf{1.0000} \text{ (100.0\%)}$$

**Audit Conclusion:** The metric math is 100% correct, verified by 26 unit tests in `tests/evaluation/test_retrieval_metrics.py`.

---

## 3. Section 2 — Precision@5 Investigation

The reported aggregate `mean_precision_at_k` is `0.2400`. The audit investigated whether this score indicates poor retrieval or metric mechanics.

### 3.1 Investigation Findings
1. **Returned Results per Query:** Every query returned exactly 5 chunks (`returned_results = 5`).
2. **Scoring Granularity:** Relevance is judged strictly at the **chunk ID** level (`eval_doc_XX_v1_cXXXX`), not at the document level.
3. **Denominator:** Uses $K = 5.0$.
4. **Ground-Truth Expected Chunk Counts:**
   - 6 queries have only **1** expected chunk: `tc_01`, `tc_03`, `tc_06`, `tc_07`, `tc_09`, `tc_10` ($\text{Max P} = 1/5 = 0.20$).
   - 3 queries have **2** expected chunks: `tc_02`, `tc_04`, `tc_05` ($\text{Max P} = 2/5 = 0.40$).
   - 1 query has **0** expected chunks: `tc_08` ($\text{Max P} = 0/5 = 0.00$).

### 3.2 Mathematical Precision Ceiling
The maximum theoretical precision achievable by any retrieval system on this dataset when $K=5$ is:
$$\text{Precision@5}_{\text{max}} = \frac{(6 \times 0.20) + (3 \times 0.40) + (1 \times 0.00)}{10} = \frac{1.20 + 1.20 + 0.00}{10} = \frac{2.40}{10} = \mathbf{0.2400}$$

### 3.3 Worked Example (`tc_01_factual_gpa`)
- **Query:** "What is the minimum cumulative GPA required for graduate degree graduation?"
- **Expected Chunk:** `eval_doc_02_v1_c0000` (1 chunk only).
- **Retrieved Top 5:**
  1. `eval_doc_02_v1_c0000` (**Relevant** — GPA requirement: 3.0)
  2. `eval_doc_04_v1_c0000` (Financial aid GPA requirement: 2.0)
  3. `eval_doc_02_v1_c0001` (Master's thesis defense)
  4. `eval_doc_04_v1_c0001` (Presidential Fellowship renewal GPA: 3.5)
  5. `eval_doc_01_v1_c0002` (Academic appeals committee)
- **Relevant Chunks Retrieved in Top 5:** 1
- **Precision@5:** $1 / 5 = \mathbf{0.2000}$ (100% of theoretical maximum for this query).
- **Recall@5:** $1 / 1 = \mathbf{1.0000}$ (100%).

**Audit Conclusion:** The low Precision@5 score of `0.2400` is **not** caused by poor retrieval quality or irrelevant chunk drift. It is the **exact theoretical upper bound (100%)** achievable for standard fixed-denominator Precision@5 on this ground-truth dataset.

---

## 4. Section 3 — Candidate Oversampling Investigation

Phase 7.3 specifies:
$$\text{candidate\_k} = \min(\max(\text{top\_k} \times 3, \text{top\_k}), 100)$$

The Phase 7.5 implementation report stated:
> "Raw Vector Store Candidates: 100 (10 per query due to oversampling)"  
> "10 per query via 2x candidate oversampling"

### 4.1 Audit Verification
- **Code Check:** [`backend/services/retrieval_service.py#L175`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/services/retrieval_service.py#L175):
  ```python
  candidate_k = min(max(top_k * 3, top_k), 100)
  ```
  For `top_k = 5`, `candidate_k` computed and passed to `VectorStoreService.query_vectors` is **15** ($3\times$ oversampling).
- **Evaluation Runner Check:** [`scripts/evaluate_retrieval.py#L105-L119`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/scripts/evaluate_retrieval.py#L105-L119) and [`backend/evaluation/retrieval_evaluator.py#L113-L119`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/evaluation/retrieval_evaluator.py#L113-L119) directly call `self.retrieval_service.retrieve(query=query_obj, db=session)` without altering or overriding oversampling logic.
- **Why ChromaDB returned 10 candidates:**
  The synthetic university policy corpus has **exactly 10 chunks across 4 documents**:
  - `doc_eval_01`: 3 chunks
  - `doc_eval_02`: 2 chunks
  - `doc_eval_03`: 3 chunks
  - `doc_eval_04`: 2 chunks
  - Total: $3 + 2 + 3 + 2 = 10\text{ chunks}$.
  When `candidate_k = 15` is queried, ChromaDB can only return at most 10 embeddings because only 10 exist in the collection.
- **Finding (F-03):** The retrieval service code strictly complies with the Phase 7.3 specification. The reference in the report narrative to "$2\times$ oversampling" was an empirical misattribution by the report author ($10 / 5 = 2$). The actual requested candidate count was 15, bounded by the corpus size of 10.

---

## 5. Section 4 — Unanswerable Query Investigation

### 5.1 Analysis of `tc_08_unanswerable_parking`
- **Query:** "What are the fees for campus vehicle parking permits and shuttle bus timetables?"
- **Expected Relevance:** None (`expected_chunk_ids: []`).
- **Results Retrieved:** 5 chunks from housing and conduct policies.
- **Empirical Similarity Scores:**
  - `eval_doc_03_v1_c0002`: `0.6276` (Housing cancellation fee)
  - `eval_doc_03_v1_c0000`: `0.5645` (Residence quiet hours)
  - `eval_doc_01_v1_c0002`: `0.5613` (Academic appeals)
  - `eval_doc_03_v1_c0001`: `0.5171` (Fire safety)
  - `eval_doc_02_v1_c0000`: `0.5000` (Master's GPA requirements)
- **Configuration:** No `similarity_threshold` was configured (`similarity_threshold = None`).

### 5.2 Meaning of Empty-Result Rate
Because vector search in Euclidean/cosine space always returns the nearest $K$ vectors when no similarity threshold is applied, the `empty_result_rate` of `0.0000` (0/10) is **uninformative** for negative query rejection unless a similarity threshold is enforced.

### 5.3 Framework Measurement of False Positives
The evaluation framework **does** measure false positives via `calculate_recall_at_k`: because 5 chunks were retrieved for an unanswerable query, Recall@5 evaluated to `0.0000`.

### 5.4 Recommendation for Phase 13 (Evidence Evaluator)
Dense retrieval alone cannot distinguish between weakly relevant and truly relevant chunks without a threshold or agentic verification:
- Phase 13 must implement an **Evidence Evaluator** that checks candidate similarity scores against an empirical confidence threshold (e.g., $\ge 0.70$) before passing context to LLM synthesis.
- If all retrieved candidates fall below the threshold, the agent must trigger query refinement (Phase 14) or report that the corpus does not contain the answer.

---

## 6. Section 5 — SQLite Validation and Stale Vector Testing

### 6.1 Evaluation Coverage Audit
The audit examined whether the Phase 7.5 evaluation benchmark exercises the candidate validation rules implemented in Phase 7.3:
- Stale document versions
- Deleted documents (`DELETION_PENDING`)
- Non-indexed documents (`FAILED`, `PROCESSING`, `PENDING`)
- Wrong collection IDs
- Missing / invalid metadata
- ChromaDB / SQLite desynchronization

### 6.2 Audit Finding (F-05)
- **In Unit Tests:** [`tests/test_retrieval_service.py`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/tests/test_retrieval_service.py) comprehensively covers all of these scenarios in 22 dedicated test cases (Tests 3–13, 16–21). All 22 tests pass.
- **In Benchmark Corpus:** The evaluation seeder [`backend/evaluation/evaluation_dataset.py#L312-L360`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/evaluation/evaluation_dataset.py#L312-L360) inserts only active, indexed, version-1 documents.
- **Result:** In the evaluation report, `total_candidates_sum = 100`, `accepted_candidates_sum = 100`, and `rejected_candidates_sum = 0`.
- **Classification:** **Evaluation Coverage Gap**. While candidate accounting fields exist in schemas and reports, the benchmark dataset does not include deliberately corrupted or stale records to quantify candidate rejection in the evaluation report itself.

---

## 7. Section 6 — Latency and Hardware Measurement

### 7.1 Measurement Verification
- **Model Warmup Latency (`22.74 ms`):** Measures the duration of `embed_text("warmup query probe")` prior to the query loop. Because the model weights are already loaded in memory during dataset seeding, this measures the first inference pass (JIT initialization and thread pool warmup), successfully isolating it from steady-state query latencies.
- **Query Latency (`39.65 ms` average):** Monotonically measured around `RetrievalService.retrieve(query, db)` using `time.perf_counter()`. Includes:
  1. Query text embedding generation via `EmbeddingService`.
  2. Vector search in ChromaDB via `VectorStoreService`.
  3. Single-trip batched SQLite candidate validation via `AsyncSession`.
  4. In-memory candidate deduplication and response construction.
- **Benchmark Execution:** Strictly sequential (`for tc in self.dataset:`). No concurrent load was applied.
- **Percentile Calculation (P95):**
  - Implemented in [`backend/evaluation/retrieval_metrics.py#L214-L217`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/evaluation/retrieval_metrics.py#L214-L217) as `p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)`.
  - For $N = 10$, $\text{ceil}(9.5) - 1 = 9$ (the 10th and maximum element, `60.42 ms`).
  - [`backend/evaluation/retrieval_evaluator.py#L250-L254`](file:///c:/Users/YASHWANTH%20D/OneDrive/Desktop/AGENTIC AND LLM BOT/backend/evaluation/retrieval_evaluator.py#L250-L254) correctly emits a structured warning in `report.warnings` alerting consumers that $N=10$ is a small sample where P95 equals nearest rank.

---

## 8. Section 7 — Evaluation Dataset Quality

### 8.1 Verification of 10 Test Cases

| Case ID | Query Type | Query Summary | Ground-Truth Chunk ID | Verified in Corpus |
|---|---|---|---|---|
| `tc_01_factual_gpa` | factual | Graduate minimum GPA | `eval_doc_02_v1_c0000` | YES (GPA 3.0 in Master's regulations) |
| `tc_02_conceptual_integrity` | conceptual | Honor code & sanctions | `c0000`, `c0001` (Doc 1) | YES (Plagiarism def + disciplinary sanctions) |
| `tc_03_definition_plagiarism` | definition | Plagiarism definition | `eval_doc_01_v1_c0000` | YES (Honor Code section 1) |
| `tc_04_multipart_degree_defense`| multi-part | Credits + defense committee | `c0000`, `c0001` (Doc 2) | YES (30 credit hours + 3 faculty members) |
| `tc_05_crosschunk_sanctions_appeals`| cross-chunk | Sanctions + appeal filing | `c0001`, `c0002` (Doc 1) | YES (Sanctions + 10-day appeal window) |
| `tc_06_paraphrase_dorm_noise` | paraphrase | Residence quiet hours | `eval_doc_03_v1_c0000` | YES (Sunday-Thursday quiet hours) |
| `tc_07_limited_info_stipend` | limited-info | Fellowship stipend & renewal | `eval_doc_04_v1_c0001` | YES ($28,000 stipend + 3.5 GPA renewal) |
| `tc_08_unanswerable_parking` | unanswerable| Parking permit fees | None (`[]`) | YES (Corpus has 0 parking references) |
| `tc_09_similar_terms_quorum` | similar-terms| Appeals committee quorum | `eval_doc_01_v1_c0002` | YES (Quorum of 4 voting members) |
| `tc_10_docspecific_housing_refund`| doc-specific | Housing cancellation fee | `eval_doc_03_v1_c0002` | YES ($500 fee + pro-rated 4-week refund) |

### 8.2 Label Integrity Verification
- **Existence:** All ground-truth chunk IDs match chunks seeded in `SYNTHETIC_DOCUMENTS`.
- **Manual Authoring:** Labels were explicitly hand-authored with targeted semantic rationale; no labels were auto-generated or reverse-engineered from model embeddings.
- **Representativeness:** 10 test cases across 4 documents provide an effective baseline to verify pipeline correctness and metric math, but are insufficient to claim general production performance across diverse enterprise domains.

---

## 9. Section 8 — Audit Findings & Conditions

### Operational Conditions
1. **Condition 1 (Score Reporting Context):** When presenting Phase 7.5 results, `mean_precision_at_k = 0.2400` must be accompanied by the explanation that 0.2400 is the theoretical maximum for fixed-denominator $K=5$ given ground-truth cardinality ($|\text{Rel}| \le 2$).
2. **Condition 2 (Answerable vs Overall Disaggregation):** Performance summaries must clearly state that overall Recall (0.9000) and MRR (0.9000) include the unanswerable query as 0.0, and that Answerable-Only Recall and MRR are 1.0000 (100%).
3. **Condition 3 (Phase 13 Prerequisite):** The observation that unanswerable queries retrieve dense false positives confirms that Agentic Phase 13 (Evidence Sufficiency Evaluator) is strictly required to prevent hallucinations prior to generation.
4. **Condition 4 (Adverse Benchmark Test Dataset):** In a future testing phase, add an optional `--adverse` evaluation dataset mode that includes stale versions and deleted documents to benchmark candidate rejection rates.

---

## 10. Final Audit Verdict

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PHASE 7.5 AUDIT VERDICT                         │
│                                                                        │
│                      >>>  PASS WITH CONDITIONS  <<<                     │
│                                                                        │
│  - Architectural integrity: 100% compliant with AGENTS.md (No LangChain)│
│  - Production data: Untouched (0 SQLite docs, 0 ChromaDB embeddings)   │
│  - Mathematical correctness: Verified across all metrics               │
│  - Phase 7 regression: 102/102 tests passed                            │
│  - Dependencies: Clean (pip check verified)                            │
└────────────────────────────────────────────────────────────────────────┘
```

**Phase 7.5: PASS WITH CONDITIONS — approved.**

**Phase 8 has not started and requires explicit user approval.**

