# Phase 5: BGE Model Download & Real CPU Inference Smoke Test Report

**Project:** Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models  
**Phase:** 5B — Model Download & Real CPU Inference Verification  
**Date:** 2026-11-06  
**Status:** **10/10 CHECKS PASSED — REAL INFERENCE VERIFIED (DIMENSION 384, UNIT NORM, CPU EXECUTION)**  

---

## 1. Objective & Scope

Before implementing `backend/services/embedding_service.py`, a real-model download and CPU inference smoke test was executed under controlled hardware conditions:
- **Model:** `BAAI/bge-small-en-v1.5`
- **Cache Storage:** `D:\huggingface` (offloading weights from C: drive and keeping the Git workspace clean)
- **Execution Target:** AMD Ryzen 7 7730U (Pure CPU, 8 execution threads)
- **Batch Size:** 1 (conservative memory footprint)
- **Normalizations:** $L_2$ unit normalization enabled

---

## 2. Download & Cache Verification

| Verification Item | Result | Analysis |
|---|---|---|
| **Cache Target Location** | `D:\huggingface` | Verified writable; 232.65 GB free space |
| **Download Status** | Success | Initial download and load completed in **23.95 seconds** |
| **Commit SHA** | `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` | Matches official Hugging Face repository snapshot |
| **Model Weights File** | `model.safetensors` (133.4 MB) | Safe deserialization format loaded into memory |
| **Tokenizer Files** | `tokenizer.json` (711 KB), `vocab.txt` (231 KB) | Pre-trained BERT WordPiece vocabulary loaded |
| **Workspace Isolation** | 0 model files in workspace | All weights isolated on `D:\` drive; Git tree clean |

---

## 3. Real Model Inference Output

Three sample sentences were encoded:
1. *"Artificial intelligence helps computers learn patterns from data."*
2. *"Machine learning is a branch of artificial intelligence."*
3. *"The weather is sunny today."*

### Inference Measurements:
- **Total Inference Time:** `0.3200 s` (across 3 texts)
- **Latency Per Text:** **`106.67 ms / text`** (CPU-only execution)
- **Output Container:** `<class 'numpy.ndarray'>`
- **Output Matrix Shape:** `(3, 384)`
- **Data Type:** `numpy.float32`
- **Finite Value Check:** `True` (zero `NaN` or `Inf` entries)

---

## 4. Dimension & Normalization Verification

### Embedding Dimension:
- Official Config: 384
- Measured Dimension: **384** (Verified from actual numpy array shape `(3, 384)`)

### $L_2$ Unit Normalization:
$$\|v\|_2 = \sqrt{\sum_{i=1}^{384} v_i^2} = 1.000000$$

- **Text 1 Norm:** `1.000000` (Unit norm: True, delta < $10^{-6}$)
- **Text 2 Norm:** `1.000000` (Unit norm: True, delta < $10^{-6}$)
- **Text 3 Norm:** `1.000000` (Unit norm: True, delta < $10^{-6}$)

### Semantic Cosine Similarity (Dot Product on Unit Vectors):
- **$\text{Sim}(\text{AI}, \text{ML})$ [1 $\leftrightarrow$ 2]:** **`0.8637`** (High semantic similarity between related AI/ML concepts)
- **$\text{Sim}(\text{AI}, \text{Weather})$ [1 $\leftrightarrow$ 3]:** **`0.4522`** (Low semantic similarity to weather)
- **$\text{Sim}(\text{ML}, \text{Weather})$ [2 $\leftrightarrow$ 3]:** **`0.4448`** (Low semantic similarity to weather)
- **Semantic Differentiation:** **`PASS`** ($\text{Sim}_{1,2} \gg \text{Sim}_{1,3}$)

---

## 5. Memory & Hardware Safety Measurements

Live measurements captured via `psutil`:

| Metric | Measurement | Safety Margin |
|---|---|---|
| **Total System RAM** | 13.84 GB | Baseline hardware profile |
| **Initial Available RAM** | 2.10 GB (2,146 MB) | Above 500 MB abort threshold |
| **RAM Delta (Loaded Model)** | **338.2 MB** | Closely matches 250–350 MB audit prediction |
| **Available RAM After Load** | 1.14 GB (1,169 MB) | Safe operating margin maintained |
| **Inference Activation Memory** | < 10 MB | Negligible memory impact at batch_size=1 |
| **500 MB Safety Guardrail** | Maintained | Available RAM never fell below 1,169 MB |

---

## 6. CPU Execution Environment Verification

- **PyTorch Version:** `2.15.0+cpu`
- **CUDA Available:** `False`
- **CPU Execution Threads:** 8 physical OpenMP worker threads (AMD Ryzen 7 7730U)
- **Hardware SIMD:** AVX2 and FMA3 vectorized execution kernels active

---

## 7. Smoke Test Summary Verdict

| Check # | Verification Item | Status | Verified Measurement |
|:---:|---|:---:|---|
| 1 | D: Drive Cache Writable | **PASS** | `D:\huggingface` active, 232 GB free |
| 2 | Available RAM $\ge$ 500 MB | **PASS** | 1,169 MB available post-load |
| 3 | PyTorch CPU-Only | **PASS** | `CUDA available: False` |
| 4 | Model Load Successful | **PASS** | `BAAI/bge-small-en-v1.5` loaded in 23.95s |
| 5 | Embedding Dimension == 384 | **PASS** | Exactly 384 dense float features |
| 6 | Output Shape == (3, 384) | **PASS** | `(3, 384)` NumPy matrix |
| 7 | Data Type == float32 | **PASS** | Single-precision IEEE 754 float |
| 8 | All Values Finite | **PASS** | Zero NaN / Inf values |
| 9 | All $L_2$ Norms == 1.0 | **PASS** | Exact unit norm $1.000000$ on all vectors |
| 10 | Semantic Similarity Valid | **PASS** | $\text{Sim}(\text{AI}, \text{ML}) = 0.8637 > 0.4522$ |

**OVERALL RESULT:** **SMOKE TEST PASSED (10/10 CHECKS)**

---

## 8. Cleanup Confirmation
- Temporary smoke test script (`scripts/smoke_test_bge.py`) was deleted immediately following test completion.
- Workspace working tree remains clean and pristine.
- No source code, database tables, or Phase 1–4 modules have been modified.
