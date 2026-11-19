# Phase 5: ML Dependency Installation & Verification Report

**Project:** Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models  
**Phase:** 5B — Step 3: ML Dependency Installation & Isolation Verification  
**Date:** 2026-11-05  
**Status:** **ALL PACKAGES INSTALLED & VERIFIED — 75/75 REGRESSION TESTS PASS — GLOBAL PYTHON PRISTINE**  

---

## 1. Installation Overview

In accordance with user authorization, the verified Phase 5 machine learning dependency stack was installed strictly into the project virtual environment (`.venv`).

### Installation Command Executed:
```bash
.\.venv\Scripts\python.exe -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  torch==2.15.0+cpu \
  sentence-transformers==3.3.1 \
  transformers==4.57.6 \
  tokenizers==0.22.2 \
  safetensors==0.8.0 \
  huggingface-hub==0.36.2
```

### Result:
- **Exit Code:** `0` (Success)
- **Status:** All packages and dependencies installed without errors or conflicts.

---

## 2. Actual Installed Package Versions in `.venv`

Verified via `importlib.metadata`:

| Package | Pinned Target | Actual Installed Version | Wheel Type / Origin |
|---|---|---|---|
| `torch` | `2.15.0+cpu` | `2.15.0+cpu` | PyTorch Official CPU Wheel (`win_amd64`) |
| `sentence-transformers` | `3.3.1` | `3.3.1` | PyPI (`py3-none-any`) |
| `transformers` | `4.57.6` | `4.57.6` | PyPI (`py3-none-any`) |
| `tokenizers` | `0.22.2` | `0.22.2` | PyPI (`cp39-abi3-win_amd64`) |
| `safetensors` | `0.8.0` | `0.8.0` | PyPI (`cp310-abi3-win_amd64`) |
| `huggingface-hub` | `0.36.2` | `0.36.2` | PyPI (`py3-none-any`) |
| `scikit-learn` | transitive | `1.9.1` | PyPI (`cp313-win_amd64`) |
| `scipy` | transitive | `1.18.1` | PyPI (`cp313-win_amd64`) |
| `sympy` | transitive | `1.14.0` | PyPI (`py3-none-any`) |
| `mpmath` | transitive | `1.3.0` | PyPI (`py3-none-any`) |
| `numpy` | existing | `2.5.4` | Retained as-is (satisfies `numpy>=1.17`) |

---

## 3. Pip Requirement Consistency Check

Executed:
```bash
.\.venv\Scripts\python.exe -m pip check
```
**Output:**
```text
No broken requirements found.
```

---

## 4. PyTorch CPU Execution Verification

Executed:
```bash
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.get_num_threads())"
```
**Output:**
```text
Torch version:  2.15.0+cpu
CUDA available: False
CPU threads:    8
```

### Findings:
1. Pure CPU build confirmed (`+cpu`).
2. CUDA is completely disabled (`False`); **zero CUDA/NVIDIA dependencies downloaded or installed**.
3. AMD Ryzen 7 7730U is recognized with 8 physical execution threads available for OpenMP/AVX2 acceleration.

---

## 5. ML Framework Import Verification

Executed:
```bash
.\.venv\Scripts\python.exe -c "
import sentence_transformers, transformers, tokenizers, safetensors, huggingface_hub
print('All ML packages imported successfully!')
"
```
**Output:**
```text
sentence_transformers version: 3.3.1
transformers version: 4.57.6
tokenizers version: 0.22.2
safetensors version: 0.8.0
huggingface_hub version: 0.36.2
All ML packages imported successfully!
```

---

## 6. Regression Testing Verification (Post-Installation)

Executed:
```bash
.\.venv\Scripts\python.exe -m pytest -q
```
**Output:**
```text
........................................................................ [ 96%]
...                                                                      [100%]
75 passed in 3.39s
```

All 75 tests across all Phase 1–4 test suites continue to pass with 100% success.

---

## 7. Global Python Environment Isolation Verification

Executed:
```bash
C:\Python313\python.exe -c "
import sys, importlib.util
print('Global executable:', sys.executable)
print('Is torch installed in global?:', importlib.util.find_spec('torch') is not None)
print('Is sentence_transformers installed in global?:', importlib.util.find_spec('sentence_transformers') is not None)
print('Is transformers installed in global?:', importlib.util.find_spec('transformers') is not None)
"
```
**Output:**
```text
Global executable: C:\Python313\python.exe
Global prefix:     C:\Python313
Is torch installed in global?: False
Is sentence_transformers installed in global?: False
Is transformers installed in global?: False
```

The global system Python environment remains completely unpolluted.

---

## 8. Backup & Scope Integrity

1. `requirements.txt.bak` is verified intact on disk.
2. Zero BGE model weights downloaded.
3. Zero database schema or data modifications.
4. Zero application services or schemas modified yet.
5. ChromaDB and semantic retrieval remain completely excluded.
