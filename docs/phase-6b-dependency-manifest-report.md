# Phase 6B — Step 1.2: Dependency Manifest Synchronization Report

**Project:** Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models  
**Institution:** Dayananda Sagar University  
**Scope:** Minimal synchronization of `requirements.txt` to reflect the verified installed environment  
**Date:** 2026-11-07  
**Status:** COMPLETE (Step 1.2.1 correction applied) — Verified & Awaiting Review

---

## 1. Files Modified

| File | Action | Notes |
|---|---|---|
| `requirements.txt` | **Updated** | Step 1.2 + Step 1.2.1 corrections applied |
| `requirements.txt.bak` | **Untouched** | Original Phase 1-4 baseline preserved as-is |
| `pyproject.toml` | **Untouched** | Pytest configuration only; no dependency pins |
| All `backend/` source files | **Untouched** | Zero application code changes |
| All `tests/` files | **Untouched** | Zero test changes |

---

## 2. Changes Applied to `requirements.txt` — Step 1.2 (Initial Synchronization)

### 2.1 Pydantic Version Synchronization

**Before (original):**
```
pydantic==2.12.5
pydantic-settings==2.3.0
pydantic-core==2.41.5
```

**After Step 1.2:**
```
pydantic==2.14.3           # Upgraded Phase 6B: required by kubernetes>=37.0 (chromadb transitive dep)
pydantic-settings==2.3.0
pydantic-core==2.50.4      # Upgraded Phase 6B: locked to pydantic==2.14.3 (Rust core engine)
```

### 2.2 Phase 5 Embedding Packages Activated

**Before:**
```
# sentence-transformers==3.3.1   # Install at Phase 5
# torch==2.4.1                   # Install at Phase 5 (CPU-only)
```

**After:**
```
sentence-transformers==3.3.1
torch==2.15.0+cpu          # CPU-only wheel — install via: --index-url https://download.pytorch.org/whl/cpu
transformers==4.57.6
tokenizers==0.22.2
safetensors==0.8.0
huggingface-hub==0.36.2
```

### 2.3 Phase 6 ChromaDB Activated

**Before:**
```
# chromadb==0.5.23          # Install at Phase 6
```

**After:**
```
chromadb==1.5.9            # Installed Phase 6B (embedded PersistentClient only)
```

---

## 3. Corrections Applied — Step 1.2.1

### 3.1 pydantic-core: Direct Pin Removed

**Reason:** `pydantic-core` is a strict transitive dependency of `pydantic`; its version is fully controlled by pydantic's own metadata (`pydantic 2.14.3` declares `pydantic-core==2.50.4` as a hard requirement). Directly pinning it as a first-class requirement is unnecessary and reduces portability. It is replaced with a documentary comment recording the version active at Phase 6B.

**Before (Step 1.2):**
```
pydantic==2.14.3           # Upgraded Phase 6B: required by kubernetes>=37.0 (chromadb transitive dep)
pydantic-settings==2.3.0
pydantic-core==2.50.4      # Upgraded Phase 6B: locked to pydantic==2.14.3 (Rust core engine)
```

**After (Step 1.2.1):**
```
pydantic==2.14.3           # Upgraded Phase 6B: required by kubernetes>=37.0 (chromadb transitive dep)
pydantic-settings==2.3.0
# pydantic-core is a transitive dependency of pydantic; version resolved automatically (2.50.4 at Phase 6B)
```

### 3.2 PyTorch CPU Installation Documentation Improved

**Reason:** The previous single-line inline comment was insufficient to guide a fresh environment setup. The CPU wheel for `torch` is published exclusively on the PyTorch index, not PyPI. A `--extra-index-url` on the `pip install -r` command line would affect resolution of unrelated packages. The correct pattern is to install `torch` first using `--index-url`, then install the remaining requirements separately. This two-step procedure is now documented as explicit comments directly above the `torch` line.

**Before (Step 1.2):**
```
# --- [Phase 5] Embeddings ---
sentence-transformers==3.3.1
torch==2.15.0+cpu          # CPU-only wheel — install via: --index-url https://download.pytorch.org/whl/cpu
transformers==4.57.6
tokenizers==0.22.2
safetensors==0.8.0
huggingface-hub==0.36.2
```

**After (Step 1.2.1):**
```
# --- [Phase 5] Embeddings ---
# NOTE: torch must be installed separately before pip-installing this file, because its CPU
# wheel is published only on the PyTorch index, not PyPI. Use this exact command in a fresh .venv:
#   .\.venv\Scripts\python.exe -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
# Then install the remaining packages normally:
#   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
sentence-transformers==3.3.1
torch==2.15.0+cpu          # CPU-only — see installation note above; CUDA not required or supported
transformers==4.57.6
tokenizers==0.22.2
safetensors==0.8.0
huggingface-hub==0.36.2
```

---

## 4. Diff: requirements.txt.bak → Final requirements.txt

> [!NOTE]
> The repository has no commits yet (`No commits yet` on `master`). `requirements.txt.bak` reflects the original pre-Phase-5A baseline.

```diff
 # --- Data validation and settings ---
-pydantic==2.12.5
+pydantic==2.14.3           # Upgraded Phase 6B: required by kubernetes>=37.0 (chromadb transitive dep)
 pydantic-settings==2.3.0
-pydantic-core==2.41.5
+# pydantic-core is a transitive dependency of pydantic; version resolved automatically (2.50.4 at Phase 6B)

 # --- Testing ---
-pytest==9.1.0
+pytest==8.4.2

-# --- [Phase 5] Embeddings ---
-# sentence-transformers==3.3.1   # Install at Phase 5
-# torch==2.4.1                   # Install at Phase 5 (CPU-only)
+# --- [Phase 5] Embeddings ---
+# NOTE: torch must be installed separately before pip-installing this file…
+sentence-transformers==3.3.1
+torch==2.15.0+cpu
+transformers==4.57.6
+tokenizers==0.22.2
+safetensors==0.8.0
+huggingface-hub==0.36.2

-# chromadb==0.5.23          # Install at Phase 6
+chromadb==1.5.9            # Installed Phase 6B (embedded PersistentClient only)
```

---

## 5. Step 1.2.1 Verification Results

### Check 1 — Installed Pydantic and pydantic-core versions

```
pydantic:      2.14.3
pydantic_core: 2.50.4
```

Removing the direct `pydantic-core` pin had **zero effect** on the installed environment. The version remained `2.50.4` as resolved by pip through pydantic's metadata.

### Check 2 — `pip check`

```
No broken requirements found.
```

✅ Zero dependency conflicts.

### Check 3 — Dependency Resolution Dry-Run

```bash
.\.venv\Scripts\python.exe -m pip install --dry-run -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Every declared package returned `Requirement already satisfied`. Confirmed packages:

| Package | Declared | Dry-Run | Status |
|---|---|---|---|
| `pydantic` | `2.14.3` | Requirement already satisfied | ✅ |
| `pydantic-settings` | `2.3.0` | Requirement already satisfied | ✅ |
| `chromadb` | `1.5.9` | Requirement already satisfied | ✅ |
| `torch` | `2.15.0+cpu` | Requirement already satisfied | ✅ |
| `transformers` | `4.57.6` | Requirement already satisfied | ✅ |
| `tokenizers` | `0.22.2` | Requirement already satisfied | ✅ |
| `sentence-transformers` | `3.3.1` | Requirement already satisfied | ✅ |
| `fastapi` | `0.111.0` | Requirement already satisfied | ✅ |
| `sqlalchemy` | `2.0.49` | Requirement already satisfied | ✅ |

**Zero packages proposed for upgrade, downgrade, or new installation.**

### Check 4 — Regression Suite

```bash
.\.venv\Scripts\python.exe -m pytest -m "not real_model" -q
```

```
........................................................................ [ 70%]
..............................                                           [100%]
102 passed, 1 deselected in 13.90s
```

✅ **102/102 tests pass.** Zero regressions.

### Check 5 — Installed Stack Versions Unchanged

| Package | Version | Status |
|---|---|---|
| `chromadb` | `1.5.9` | ✅ Unchanged |
| `torch` | `2.15.0+cpu` | ✅ Unchanged (CPU-only) |
| `sentence-transformers` | `3.3.1` | ✅ Unchanged |
| `transformers` | `4.57.6` | ✅ Unchanged |
| `tokenizers` | `0.22.2` | ✅ Unchanged |

### Check 6 — CUDA Unavailable, Zero NVIDIA Packages

```
cuda_available: False
NVIDIA/CUDA packages: NONE
```

✅ CPU-only environment confirmed. No CUDA or NVIDIA packages in `.venv`.

### Check 7 — Final requirements.txt Review

```
# ============================================================
# requirements.txt — Intelligent Knowledge Assistant
# Pinned for reproducibility. Update versions only after testing.
# Phase 1: Core stack declared. Packages marked [PhaseN] are
# installed at the start of the relevant phase, NOT before.
# ============================================================

# --- Backend: FastAPI stack ---
fastapi==0.111.0
uvicorn[standard]==0.30.0
python-multipart==0.0.9
starlette==0.37.2

# --- Data validation and settings ---
pydantic==2.14.3           # Upgraded Phase 6B: required by kubernetes>=37.0 (chromadb transitive dep)
pydantic-settings==2.3.0
# pydantic-core is a transitive dependency of pydantic; version resolved automatically (2.50.4 at Phase 6B)

# --- Application database ---
sqlalchemy==2.0.49
aiosqlite==0.20.0          # Async SQLite driver for SQLAlchemy

# --- HTTP client (testing + SSE) ---
httpx==0.28.1
httpx-sse==0.4.0

# --- Frontend ---
streamlit==1.55.0

# --- Environment ---
python-dotenv==1.2.2

# --- Logging and utilities ---
rich==14.3.3
tenacity==9.1.4

# --- Testing ---
pytest==8.4.2
pytest-asyncio==0.24.0     # [Phase 2+] install at Phase 2
anyio==4.12.1

# --- [Phase 3-4] Document processing ---
pymupdf>=1.24.10          # Installed Phase 4 (2.1)
python-docx>=1.1.2        # Installed Phase 4 (1.2.0)

# --- [Phase 5] Embeddings ---
# NOTE: torch must be installed separately before pip-installing this file, because its CPU
# wheel is published only on the PyTorch index, not PyPI. Use this exact command in a fresh .venv:
#   .\.venv\Scripts\python.exe -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
# Then install the remaining packages normally:
#   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
sentence-transformers==3.3.1
torch==2.15.0+cpu          # CPU-only — see installation note above; CUDA not required or supported
transformers==4.57.6
tokenizers==0.22.2
safetensors==0.8.0
huggingface-hub==0.36.2

# --- [Phase 6] Vector database ---
chromadb==1.5.9            # Installed Phase 6B (embedded PersistentClient only)

# --- [Phase 9] LLM API ---
# anthropic==0.40.0         # Install at Phase 9
```

---

## 6. Remaining Limitations

1. **Two-step fresh install required for torch.** The `torch==2.15.0+cpu` CPU wheel is not on PyPI. The installation procedure documented in the file comments must be followed on a clean environment.

2. **`pymupdf` and `python-docx` use minimum-version specifiers.** Range pins (`>=1.24.10`, `>=1.1.2`) allow minor upgrades on a fresh install. The actual tested versions (`2.1` and `1.2.0`) are recorded in inline comments.

3. **ChromaDB transitive dependencies are not individually pinned.** `kubernetes==37.0.2`, `grpcio==1.85.1`, `opentelemetry-*` etc. are resolved through ChromaDB's wheel metadata. A `pip freeze` snapshot is preserved in the Phase 6B installation report.

4. **`requirements.txt.bak` reflects the original pre-Phase-5A baseline** (`pytest==9.1.0`) and was intentionally left unmodified as an audit trail.

---

## 7. Sign-Off

All Step 1.2.1 acceptance criteria met:

| Criterion | Result |
|---|---|
| `pydantic==2.14.3` retained | ✅ |
| Direct `pydantic-core` pin removed | ✅ |
| `pydantic-core 2.50.4` still active (transitive) | ✅ |
| PyTorch two-step install documented | ✅ |
| No global `--index-url` added to requirements | ✅ |
| `requirements.txt.bak` unmodified | ✅ |
| `pip check`: no conflicts | ✅ |
| Dry-run: zero proposals for change | ✅ |
| CUDA unavailable, zero NVIDIA packages | ✅ |
| Regression suite: 102/102 pass | ✅ |
| No source code or data modified | ✅ |
| No Git commits made | ✅ |

**Awaiting user review and authorization to proceed.**
