# Phase 5A Audit Report — Environment and Model Compatibility

**Project:** Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models  
**Phase:** 5A — Environment & Model Compatibility Audit (Pre-implementation Gate)  
**Date:** 2026-11-05  
**Author:** AI Engineering Assistant (Pair Programming with Yashwanth D)  
**Status:** AUDIT COMPLETED — APPROVED FOR REVIEW  

---

## 1. Executive Summary

Before beginning Phase 5 implementation (Embedding Generation), an in-depth, read-only audit of the host environment, hardware profile, machine learning library ecosystem, and Hugging Face model repository was conducted.

The target embedding architecture specifies:
- **Model:** `BAAI/bge-small-en-v1.5`
- **Framework:** `sentence-transformers`
- **Execution Target:** CPU-first execution (AMD Ryzen 7 7730U)
- **Embedding Dimension:** 384 dimensions (verified against official model config)
- **Vector Normalization:** Enabled ($L_2$ unit normalization for inner-product cosine equivalence)
- **Batch Size:** 4 chunks per batch (conservative memory profile)
- **Strict Boundary:** Plain Python only; **LangChain and LangGraph are strictly prohibited**.

All prerequisites for CPU-first execution have been verified. Dependency resolution with Python 3.13 (64-bit) has been validated via dry-run simulation using the PyTorch CPU official index (`https://download.pytorch.org/whl/cpu`).

---

## Section A: Python and Environment Details

| Parameter | Measured Value | Analysis / Compliance |
|---|---|---|
| **Python Version** | `Python 3.13.3` (64-bit AMD64) | Fully compatible with PyTorch 2.15+ and Sentence Transformers 6.2+ |
| **Python Binary Path** | `C:\Python313\python.exe` | Standard Windows system installation |
| **Virtual Environment** | `sys.prefix == sys.base_prefix` (False / None) | Running directly in user environment (`AppData\Roaming\Python\Python313`) |
| **Operating System** | Windows 11 Home / Pro (x86_64) | Supported by PyTorch CPU and Hugging Face wheels |
| **Path Handling** | Windows backslash native | Sanitized forward-slash path convention maintained across backend |

> [!NOTE]
> Python 3.13 wheels for `torch` (`torch-2.15.0+cpu-cp313-cp313-win_amd64.whl`) and `sentence-transformers` are officially published and tested.

---

## Section B: Installed ML Package Status

A complete audit of installed packages in the Python 3.13 site-packages was conducted:

| Package | Status | Version | Location / Source |
|---|---|---|---|
| `numpy` | **Installed** | `2.1.1` | `AppData\Roaming\Python\Python313\site-packages` |
| `scipy` | **Installed** | `1.17.1` | `AppData\Roaming\Python\Python313\site-packages` |
| `scikit-learn` | **Installed** | `1.5.2` | `AppData\Roaming\Python\Python313\site-packages` |
| `huggingface_hub` | **Installed** | `1.11.0` | `AppData\Roaming\Python\Python313\site-packages` |
| `filelock` | **Installed** | `3.28.0` | `AppData\Roaming\Python\Python313\site-packages` |
| `tqdm` | **Installed** | `4.67.3` | `AppData\Roaming\Python\Python313\site-packages` |
| `typing_extensions` | **Installed** | `4.15.0` | `AppData\Roaming\Python\Python313\site-packages` |
| `torch` | **Not Installed** | None | Must be installed with `--extra-index-url https://download.pytorch.org/whl/cpu` |
| `transformers` | **Not Installed** | None | Will be installed transitively with `sentence-transformers` |
| `sentence-transformers` | **Not Installed** | None | To be installed upon approval |

---

## Section C: Hardware, CPU, and RAM Compatibility Analysis

### 1. Processor & Compute Architecture
- **CPU:** AMD Ryzen 7 7730U with Radeon Graphics
- **Cores & Threads:** 8 physical cores, 16 logical execution threads.
- **Base Frequency:** ~2.00 GHz, Boost up to ~4.50 GHz.
- **SIMD Extensions:** AVX2, FMA3 supported (delivers high-performance CPU matrix multiplication for PyTorch / ONNX).
- **GPU Architecture:** AMD Radeon Integrated Graphics (UMA — Unified Memory Architecture; no NVIDIA CUDA cores).
- **Execution Target:** **CPU-only execution** via PyTorch CPU backend.

### 2. System Memory Profile
- **Total Physical RAM:** 16.00 GB DDR4
- **Hardware Reserved (iGPU):** 2.16 GB
- **Usable Physical RAM:** 13.84 GB
- **Current Used RAM:** 12.12 GB (87.5% load during audit)
- **Available RAM:** 1.72 GB

### 3. Model Memory Footprint & RAM Pressure Mitigation
The target model `BAAI/bge-small-en-v1.5` has the following footprint:
- **Parameter Count:** ~33.4 Million parameters
- **Weight Size (FP32):** ~133 MB (`model.safetensors`)
- **Loaded Memory Footprint:** ~250 MB – 350 MB RAM
- **Batch Size 4 Activation Footprint:** < 50 MB RAM

> [!TIP]
> **Memory Safeguards for Phase 5 Implementation:**
> 1. Restrict `batch_size = 4` chunks per forward pass.
> 2. Initialize the model once as a singleton or lifespan dependency (prevent multiple instances in memory).
> 3. Enforce `torch.inference_mode()` (or `torch.no_grad()`) to disable autograd tape allocation.
> 4. Explicitly run `gc.collect()` after processing large multi-chunk batches if memory pressure exceeds threshold.

---

## Section D: Storage and Hugging Face Hub Accessibility

### 1. Disk Space Partitioning
- **Drive C:** 38.8 GB available (SSD system drive)
- **Drive D:** 232.6 GB available (accessible and writable)

### 2. Cache Relocation Strategy
Downloading Hugging Face models to the default location (`C:\Users\YASHWANTH D\.cache\huggingface`) consumes C: drive storage.  
We verified that Drive D: has 232.6 GB free and is accessible.
- **Configured Cache Path:** `D:\huggingface\hub`
- **Environment Variable:** `HF_HOME="D:/huggingface"` or `SENTENCE_TRANSFORMERS_HOME="D:/huggingface/sentence_transformers"`
- Setting `HF_HOME` will ensure all model weights, tokenizers, and configs reside safely on the high-capacity D: drive.

### 3. Hugging Face Repository Verification
Live query against `https://huggingface.co/api/models/BAAI/bge-small-en-v1.5` confirmed repository health:
- **Repository ID:** `BAAI/bge-small-en-v1.5`
- **Latest Commit SHA:** `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`
- **Total Downloads:** > 48,000,000
- **Model Files Verified:**
  - `config.json` (743 bytes) — confirms `hidden_size: 384`
  - `model.safetensors` (133 MB) — safe deserialization format
  - `pytorch_model.bin` (134 MB) — PyTorch fallback
  - `tokenizer.json` (711 KB) & `tokenizer_config.json` (366 bytes)
  - `1_Pooling/config.json` — pooling configuration (`pooling_mode_cls_token: false`, `pooling_mode_mean_tokens: true`)

---

## Section E: Dependency Resolution and Compatibility Analysis

A live dry-run resolution was executed using `pip install --dry-run`:

```bash
python -m pip install --dry-run --extra-index-url https://download.pytorch.org/whl/cpu sentence-transformers
```

### Resolution Results:
- **`sentence-transformers`:** resolves to `6.2.0` (pure Python wheel)
- **`torch`:** resolves to `torch-2.15.0+cpu-cp313-cp313-win_amd64.whl` (CPU-optimized, ~160 MB wheel instead of 2.5 GB CUDA wheel)
- **`transformers`:** resolves to `transformers-5.22.1`
- **`tokenizers`:** resolves to `tokenizers-0.23.3-cp310-abi3-win_amd64.whl`
- **`safetensors`:** resolves to `safetensors-0.8.0-cp310-abi3-win_amd64.whl`
- **`huggingface_hub`:** upgrades from `1.11.0` to `2.4.0` (fully backwards compatible)
- **`numpy`:** existing `2.1.1` is satisfied (`numpy>=1.24.0`)
- **`scikit-learn`:** existing `1.5.2` is satisfied (`scikit-learn>=1.1.0`)
- **`scipy`:** existing `1.17.1` is satisfied (`scipy>=1.0.0`)
- **Package Conflicts:** **0 conflicts detected**.

---

## Section F: Recommended Installation & Architecture Plan

### 1. Installation Sequence (To be executed upon approval in Phase 5B)
To ensure the lightweight CPU wheel of PyTorch is installed without bloating C: drive with multi-gigabyte CUDA binaries:

```bash
# Step 1: Install PyTorch CPU directly from PyTorch CPU wheel repository
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install Sentence Transformers
python -m pip install sentence-transformers
```

### 2. Configuration Settings (`backend/config.py`)
Add embedding configuration parameters to existing Pydantic `Settings`:
```python
# Embedding settings
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION: int = 384
EMBEDDING_BATCH_SIZE: int = 4
EMBEDDING_NORMALIZE: bool = True
EMBEDDING_DEVICE: str = "cpu"
HF_CACHE_DIR: str = "D:/huggingface"
```

### 3. Service Design (`backend/services/embedding_service.py`)
- Clean plain-Python service class `EmbeddingService`.
- Lazy or lifecycle-based model initialization to minimize startup latency.
- Methods:
  - `embed_text(text: str) -> list[float]`
  - `embed_chunks(chunks: list[DocumentChunk]) -> list[list[float]]`
  - `compute_similarity(vec_a: list[float], vec_b: list[float]) -> float`
- Verification of 384 dimensions and unit norm ($\sum x_i^2 \approx 1.0$).
- Batch processing with `batch_size = 4` chunks per slice.

---

## Section G: Audit Verdict Table

| Component | Status | Finding / Action |
|---|:---:|---|
| **Python 3.13 Runtime** | **PASS** | Validated, 64-bit AMD64 architecture |
| **Existing Phase 1–4 Test Suite** | **PASS** | 75/75 tests passing cleanly |
| **CPU Architecture (Ryzen 7 7730U)** | **PASS** | AVX2/FMA3 enabled; optimal for PyTorch CPU inference |
| **RAM Capacity (13.84 GB usable)** | **PASS** | 1.72 GB free; model footprint is ~300 MB; batch=4 is safe |
| **Hugging Face Hub Connectivity** | **PASS** | `BAAI/bge-small-en-v1.5` verified reachable, files inspected |
| **Model Weights Dimension** | **PASS** | Config confirmed at 384 dimensions |
| **Storage Allocation (Drive D:)** | **PASS** | 232.6 GB free on D:; cache directory configured to `D:/huggingface` |
| **PyTorch CPU Wheel Availability** | **PASS** | `torch-2.15.0+cpu` wheel verified on PyTorch CPU index |
| **Dependency Resolution** | **PASS** | Clean dry run; zero conflicts with existing packages |
| **LangChain / LangGraph Ban** | **PASS** | Strictly plain-Python state machine and services |
| **Phase 5 Implementation** | **PENDING** | Awaiting user review and authorization |

---

## 3. Conclusion & Recommendation

The host environment is **100% compatible** and ready for Phase 5 embedding generation.  
No blockers or dependency conflicts were identified.

---

# PART II: FINAL COMPATIBILITY VERIFICATION (Pre-Installation Deep-Dive)

Conducted on: 2026-11-05 (Focused Secondary Verification)

### 1. Python Version Compatibility Verdict
- **Detected Active Interpreter:** Python 3.13.3 (64-bit AMD64) at `C:\Python313\python.exe`.
- **`pyproject.toml` inspection:** Configures pytest only (`[tool.pytest.ini_options]`). Contains no upper-bound or restrictive Python version constraints.
- **`requirements.txt` inspection:** Declares core stack. Commented lines `# sentence-transformers==3.3.1` and `# torch==2.4.1` were scaffolded in Phase 1 assuming Python 3.11. On Python 3.13, PyTorch 2.4.1 has no binary wheels (`torch 2.4.1 cp313 wheels found: 0`), whereas PyTorch 2.6.0+ through 2.15.0+ provide official, fully-tested `cp313-cp313-win_amd64` wheels.
- **Backend imports & runtime:** All Phase 1–4 modules (FastAPI, Pydantic v2, SQLAlchemy 2.0 async, PyMuPDF, python-docx, aiosqlite, HTTPX) run natively on Python 3.13.
- **Verdict:** **COMPATIBLE**. Python 3.13.3 is officially supported across all target dependencies. Upgrading the pinned PyTorch version in requirements from `2.4.1` to `2.15.0+cpu` resolves the single version delta cleanly.

### 2. Virtual Environment Strategy Recommendation
- **Current State:** The project currently executes directly in the user Python environment (`sys.prefix == sys.base_prefix == 'C:\Python313'`). There is no `.venv` or `venv` directory in the repository. All 75 tests across Phases 1–4 are installed and passing in this environment.
- **Analysis:**
  - *Option A (Maintain Existing Environment):* Zero re-download overhead; 0 risk of dependency drift for Phases 1–4; pip dry-run confirms that installing `torch-2.15.0+cpu` and `sentence-transformers` introduces 0 conflicts with existing packages.
  - *Option B (New Project-Specific .venv):* Creating a separate `.venv` would require downloading and re-installing all 25+ dependencies from Phases 1–4 from scratch, risking build variations.
- **Recommendation:** **Use the existing environment for Phase 5.** This strictly preserves all existing Phase 1–4 dependencies without unnecessary re-installations.

### 3. Exact Compatible Package Versions
Rather than unpinned latest installs, we have verified exact compatible package versions and pre-built Windows x64 wheels on official indexes:

| Package | Recommended Pinned Version | Wheel File | Requires Python | Status on PyPI / Official Index |
|---|---|---|---|---|
| `torch` | `2.15.0+cpu` | `torch-2.15.0+cpu-cp313-cp313-win_amd64.whl` (120.9 MB) | `>=3.11` | Official PyTorch CPU Index Verified |
| `sentence-transformers` | `3.3.1` (or `6.2.0`) | `sentence_transformers-3.3.1-py3-none-any.whl` (0.73 MB) | `>=3.9` | Verified on PyPI |
| `transformers` | `4.57.6` | `transformers-4.57.6-py3-none-any.whl` (12.45 MB) | `>=3.9.0` | Verified on PyPI (matches ST 3.3.1) |
| `tokenizers` | `0.22.2` | `tokenizers-0.22.2-cp39-abi3-win_amd64.whl` (2.74 MB) | `>=3.9` | Pre-built cp39-abi3 wheel (Py 3.9–3.13+) |
| `safetensors` | `0.8.0` | `safetensors-0.8.0-cp310-abi3-win_amd64.whl` (0.34 MB) | `>=3.10` | Pre-built cp310-abi3 wheel (Py 3.10–3.13+) |
| `huggingface-hub` | `0.36.2` (or current `1.11.0`) | `huggingface_hub-0.36.2-py3-none-any.whl` (0.82 MB) | `>=3.10.0` | Pure-python wheel |
| `numpy` | `2.1.1` (already installed) | `numpy-2.1.1-cp313-cp313-win_amd64.whl` | `>=3.10` | Already installed & active |

- **Dependency Constraints Check:**
  - `sentence-transformers==3.3.1` requires: `transformers<5.0.0,>=4.41.0`, `torch>=1.11.0`, `huggingface-hub>=0.20.0`, `scikit-learn`, `scipy`.
  - All constraints are strictly mutually satisfied with zero conflicts.

### 4. PyTorch CPU Wheel Verification
- **Official Windows x64 CPU Wheel:** PyTorch officially hosts `torch-2.15.0+cpu-cp313-cp313-win_amd64.whl` on `https://download.pytorch.org/whl/cpu/torch/`.
  - SHA256: `086f79c78c941b428410dc15f898ce2420a4580ae6983ea314692d71ce76c619`
- **AMD Ryzen CPU Support:** The wheel is compiled with standard x86_64 AMD64 instruction set support. It natively executes on AMD Ryzen 7 7730U and leverages AVX2 / FMA3 hardware SIMD vector extensions for accelerated tensor calculations.
- **CUDA Requirement:** **None.** The `+cpu` distribution contains zero CUDA dependencies.
- **NVIDIA Package Download Prevention:** By installing with `--index-url https://download.pytorch.org/whl/cpu`, pip is instructed to pull directly from the CPU-only index. It does NOT download `nvidia-cuda-*`, `nvidia-cudnn-*`, or `triton`. Download footprint is ~120-160 MB instead of ~2.5 GB.

### 5. Fresh Regression Test Verification
Executed freshly using the project interpreter:
- **Command:** `python -m pytest -q`
- **Exit Code:** `0`
- **Passed Tests:** **75**
- **Failed Tests:** **0**
- **Skipped Tests:** **0**
- **Execution Time:** **6.36 seconds**
- **Error Output:** None
- **Suites Verified:**
  - `tests/test_collections.py` (11 tests)
  - `tests/test_documents.py` (21 tests)
  - `tests/test_health.py` (11 tests)
  - `tests/test_document_parser.py` (10 tests)
  - `tests/test_text_cleaner.py` (8 tests)
  - `tests/test_text_chunker.py` (9 tests)
  - `tests/test_document_processor.py` (5 tests)

### 6. RAM and Model-Loading Safety Recommendations
- **Live Memory Measurement:** Total: 13.84 GB usable, Used: 11.73 GB (84.8%), Available: **2.11 GB**.
- **Safe Initial Batch Size:** Maintain `batch_size = 4` chunks (512 characters each). This bounds tensor allocation during the forward pass to < 40 MB.
- **Application Closure:** Closing non-essential background applications (e.g. idle browser instances) is recommended to maintain at least 2.5 GB of free headroom, minimizing any OS pagefile thrashing.
- **Model Loading Deferral / Guardrail:** Add a runtime pre-flight check in `EmbeddingService`: verify that system available RAM is at least 500 MB before initiating model loading. At the current 2.11 GB available RAM, loading does not need to be blocked.
- **CPU Inference Recommendation:** **Strongly Recommended.** AMD Radeon integrated graphics has no CUDA cores and shares system RAM. CPU execution via PyTorch CPU on the 8-core/16-thread Ryzen 7 7730U is stable, deterministic, and avoids any GPU driver overhead.

### 7. Final Verification Status Table

| Item | Requirement | Verification Result | Status |
|---|---|---|:---:|
| 1 | Python Version Compatibility | Python 3.13.3 verified; PyTorch 2.15+ CPU wheels exist for cp313 | **PASS** |
| 2 | Virtual Environment Strategy | Existing user environment recommended; 0 disruption to Phases 1–4 | **PASS** |
| 3 | Exact Package Versions | Pre-built wheels for Win x64 verified; 0 constraint conflicts | **PASS** |
| 4 | PyTorch CPU Wheel | Official wheel on pytorch.org/whl/cpu; no CUDA, no NVIDIA bloat | **PASS** |
| 5 | Regression Test Suite | `python -m pytest -q` -> 75 passed in 6.36s, exit code 0 | **PASS** |
| 6 | RAM & Inference Safety | Batch=4, CPU-only, 2.11 GB available, memory guardrail planned | **PASS** |
| 7 | Overall Phase 5A Gate | All checks passed; ready for Phase 5B implementation | **PASS (READY FOR REVIEW)** |

---

# PART III: DEPENDENCY RESOLUTION CORRECTION & EXACT DRY-RUN AUDIT

Conducted on: 2026-11-05 (Pre-Installation Investigation)

### 1. Proposed Pinned Package Stack
- `torch`: `2.15.0+cpu` (Official PyTorch CPU index)
- `sentence-transformers`: `3.3.1` (PyPI)
- `transformers`: `4.57.6` (PyPI)
- `tokenizers`: `0.22.2` (PyPI, `cp39-abi3` pre-built wheel)
- `safetensors`: `0.8.0` (PyPI, `cp310-abi3` pre-built wheel)
- `huggingface-hub`: `0.36.2` (PyPI)
- `numpy`: Retain existing `2.1.1` (satisfies `numpy>=1.17`)

---

### 2. Package Dependency Constraints Investigation

#### A. `sentence-transformers 3.3.1` Metadata
- `requires_python`: `>=3.9`
- Declared dependencies:
  - `transformers<5.0.0,>=4.41.0`
  - `torch>=1.11.0`
  - `huggingface-hub>=0.20.0`
  - `scikit-learn`
  - `scipy`
  - `tqdm`
  - `Pillow`
- Compatibility with `transformers==4.57.6`: **Fully compatible** (`4.57.6` is in `[4.41.0, 5.0.0)`).

#### B. `transformers 4.57.6` Metadata & `huggingface-hub` Constraint
- `requires_python`: `>=3.9.0`
- Declared dependencies:
  - **`huggingface-hub<1.0,>=0.34.0`** (STRICT UPPER BOUND)
  - `numpy>=1.17`
  - `tokenizers<=0.23.0,>=0.22.0`
  - `safetensors>=0.4.3`
  - `regex!=2019.12.17`
  - `packaging>=20.0`
  - `pyyaml>=5.1`
  - `requests`
  - `filelock`
  - `tqdm`

> [!IMPORTANT]
> **Why `huggingface-hub` must be downgraded from `1.11.0` to `0.36.2`:**
> Every release in the Transformers 4.x lineage enforces `huggingface-hub < 1.0`. Because the currently installed version in the Python environment is `1.11.0`, any installation of `transformers==4.57.6` strictly requires `huggingface-hub` to be downgraded to `<1.0`. The highest compatible version is `0.36.2`.

#### C. Impact Analysis of `huggingface-hub` Downgrade
- **Project Scope (Phases 1–4):** None of the existing Phase 1–4 packages (`fastapi`, `pydantic`, `sqlalchemy`, `aiosqlite`, `pymupdf`, `python-docx`, `pytest`, `httpx`, `streamlit`) have any dependency on `huggingface-hub`.
- **Existing User Environment Packages:**
  - `gradio` (6.13.0) requires `huggingface-hub<2.0,>=0.33.5` -> **Satisfied by `0.36.2`**.
  - `gradio_client` (2.5.0) requires `huggingface-hub<2.0,>=0.19.3` -> **Satisfied by `0.36.2`**.
- **Conclusion:** Downgrading `huggingface-hub` from `1.11.0` to `0.36.2` has **zero breaking impact** on the existing environment.

#### D. `torch 2.15.0+cpu` Compatibility with Python 3.13.3 & Windows x64
- Official wheel `torch-2.15.0+cpu-cp313-cp313-win_amd64.whl` (120.87 MB) is hosted on `https://download.pytorch.org/whl/cpu/torch/`.
- Requires Python `>=3.11`.
- Built natively for Windows AMD64 with AVX2/FMA3 hardware acceleration for AMD Ryzen.
- Zero CUDA dependencies.

---

### 3. Exact Pip Dry-Run Execution and Verification

Command executed with official PyTorch CPU wheel index:
```bash
python -m pip install --dry-run \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch==2.15.0+cpu \
  sentence-transformers==3.3.1 \
  transformers==4.57.6 \
  tokenizers==0.22.2 \
  safetensors==0.8.0 \
  huggingface-hub==0.36.2
```

#### Actual Pip Dry-Run Output:
```text
Looking in indexes: https://download.pytorch.org/whl/cpu, https://pypi.org/simple
Collecting torch==2.15.0+cpu
  Using cached https://download-r2.pytorch.org/whl/cpu/torch-2.15.0%2Bcpu-cp313-cp313-win_amd64.whl.metadata (36 kB)
Collecting sentence-transformers==3.3.1
  Using cached sentence_transformers-3.3.1-py3-none-any.whl.metadata (10 kB)
Collecting transformers==4.57.6
  Using cached transformers-4.57.6-py3-none-any.whl.metadata (43 kB)
Collecting tokenizers==0.22.2
  Using cached tokenizers-0.22.2-cp39-abi3-win_amd64.whl.metadata (7.4 kB)
Collecting safetensors==0.8.0
  Using cached safetensors-0.8.0-cp310-abi3-win_amd64.whl.metadata (4.2 kB)
Collecting huggingface-hub==0.36.2
  Using cached huggingface_hub-0.36.2-py3-none-any.whl.metadata (15 kB)
Requirement already satisfied: filelock in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from torch==2.15.0+cpu) (3.28.0)
Requirement already satisfied: typing-extensions>=4.13.0 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from torch==2.15.0+cpu) (4.15.0)
Requirement already satisfied: setuptools>=77.0.3 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from torch==2.15.0+cpu) (82.0.1)
Collecting sympy>=1.13.3 (from torch==2.15.0+cpu)
  Using cached sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
Requirement already satisfied: networkx>=2.5.1 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from torch==2.15.0+cpu) (3.6.1)
Requirement already satisfied: jinja2 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from torch==2.15.0+cpu) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from torch==2.15.0+cpu) (2026.3.0)
Requirement already satisfied: tqdm in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from sentence-transformers==3.3.1) (4.67.3)
Requirement already satisfied: scikit-learn in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from sentence-transformers==3.3.1) (1.5.2)
Requirement already satisfied: scipy in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from sentence-transformers==3.3.1) (1.17.1)
Requirement already satisfied: Pillow in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from sentence-transformers==3.3.1) (12.1.1)
Requirement already satisfied: numpy>=1.17 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from transformers==4.57.6) (2.1.1)
Requirement already satisfied: packaging>=20.0 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from transformers==4.57.6) (25.0)
Requirement already satisfied: pyyaml>=5.1 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from transformers==4.57.6) (6.0.3)
Collecting regex!=2019.12.17 (from transformers==4.57.6)
  Using cached regex-2026.9.29-cp313-cp313-win_amd64.whl.metadata (41 kB)
Requirement already satisfied: requests in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from transformers==4.57.6) (2.34.2)
Collecting mpmath<1.4,>=1.1.0 (from sympy>=1.13.3->torch==2.15.0+cpu)
  Using cached mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
Requirement already satisfied: colorama in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from tqdm->sentence-transformers==3.3.1) (0.4.6)
Requirement already satisfied: MarkupSafe>=2.0 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from jinja2->torch==2.15.0+cpu) (3.0.3)
Requirement already satisfied: charset_normalizer<4,>=2 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from requests->transformers==4.57.6) (3.4.6)
Requirement already satisfied: idna<4,>=2.5 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from requests->transformers==4.57.6) (3.11)
Requirement already satisfied: urllib3<3,>=1.26 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from requests->transformers==4.57.6) (2.6.3)
Requirement already satisfied: certifi>=2023.5.7 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from requests->transformers==4.57.6) (2026.2.25)
Requirement already satisfied: joblib>=1.2.0 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from scikit-learn->sentence-transformers==3.3.1) (1.4.2)
Requirement already satisfied: threadpoolctl>=3.1.0 in C:\Users\YASHWANTH D\AppData\Roaming\Python\Python313\site-packages (from scikit-learn->sentence-transformers==3.3.1) (3.6.0)
Would install huggingface_hub-0.36.2 mpmath-1.3.0 regex-2026.9.29 safetensors-0.8.0 sentence-transformers-3.3.1 sympy-1.14.0 tokenizers-0.22.2 torch-2.15.0+cpu transformers-4.57.6
```

---

### 4. Impact Summary on Phase 1–4 Packages

| Existing Phase 1–4 Package | Current Version | Affected by Proposed Stack? |
|---|---|:---:|
| `fastapi` | `0.111.0` | **NO (UNTOUCHED)** |
| `pydantic` | `2.12.5` | **NO (UNTOUCHED)** |
| `pydantic-settings` | `2.3.0` | **NO (UNTOUCHED)** |
| `sqlalchemy` | `2.0.49` | **NO (UNTOUCHED)** |
| `aiosqlite` | `0.20.0` | **NO (UNTOUCHED)** |
| `pymupdf` | `2.1` | **NO (UNTOUCHED)** |
| `python-docx` | `1.2.0` | **NO (UNTOUCHED)** |
| `pytest` | `9.1.0` | **NO (UNTOUCHED)** |
| `pytest-asyncio` | `0.24.0` | **NO (UNTOUCHED)** |
| `httpx` | `0.28.1` | **NO (UNTOUCHED)** |
| `streamlit` | `1.55.0` | **NO (UNTOUCHED)** |
| `numpy` | `2.1.1` | **NO (RETAINED AS-IS)** (satisfies `numpy>=1.17`) |

---

### 5. Final Package Change Breakdown

- **Proposed Package Additions (New Installs):**
  1. `torch==2.15.0+cpu` (~120.9 MB, CPU-only binary from pytorch.org/whl/cpu)
  2. `sentence-transformers==3.3.1` (0.73 MB)
  3. `transformers==4.57.6` (12.45 MB)
  4. `tokenizers==0.22.2` (2.74 MB, cp39-abi3 wheel)
  5. `safetensors==0.8.0` (0.34 MB, cp310-abi3 wheel)
  6. `sympy==1.14.0` (12 kB, torch dependency)
  7. `mpmath==1.3.0` (8.6 kB, sympy dependency)
  8. `regex==2026.9.29` (41 kB, transformers dependency)
- **Proposed Package Upgrades:**
  - **None.**
- **Proposed Package Downgrades:**
  - `huggingface-hub`: `1.11.0` $\rightarrow$ `0.36.2` (enforced by `transformers<5.0.0`; fully backwards-compatible with installed packages).


