# Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models

[![Python 3.13](https://img.shields.io/badge/Python-3.13.3-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-green.svg)](https://fastapi.tiangolo.com/)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2.14.3-orange.svg)](https://docs.pydantic.dev/)
[![React](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-blue.svg)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-8.3-purple.svg)](https://vitejs.dev/)
[![Tests](https://img.shields.io/badge/pytest-509%20passed-brightgreen.svg)]()

> **Architecture Guarantee:** Plain-Python State Machine — **Zero LangChain / Zero LangGraph**.  
> **Hardware Target:** CPU-First local execution; no GPU required.

---

## Table of Contents
1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Project Objectives](#project-objectives)
4. [Key Features](#key-features)
5. [System Architecture](#system-architecture)
6. [Agentic RAG State Machine Workflow](#agentic-rag-state-machine-workflow)
7. [Technology Stack](#technology-stack)
8. [Repository Folder Structure](#repository-folder-structure)
9. [Installation & Setup](#installation--setup)
   - [Prerequisites](#prerequisites)
   - [Backend Environment Setup](#1-backend-environment-setup)
   - [Frontend Environment Setup](#2-frontend-environment-setup)
   - [Environment Configuration](#3-environment-configuration)
10. [Running the Application](#running-the-application)
11. [LLM Provider Configuration](#llm-provider-configuration)
    - [Option A: Local Ollama (Default & Recommended)](#option-a-local-ollama-default--recommended)
    - [Option B: Cloud Anthropic Claude API (Optional)](#option-b-cloud-anthropic-claude-api-optional)
12. [Executing Automated Tests](#executing-automated-tests)
13. [API Documentation](#api-documentation)
14. [Application Screenshots](#application-screenshots)
15. [Empirical Research Benchmark](#empirical-research-benchmark)
16. [Limitations & Future Roadmap](#limitations--future-roadmap)
17. [License & Compliance](#license--compliance)

---

## Overview

The **Intelligent Knowledge Assistant** is an enterprise-grade document intelligence system designed for institutions, colleges, and enterprises. It bridges the gap between static **Conventional Retrieval-Augmented Generation (RAG)** and an autonomous, self-evaluating **Agentic RAG State Machine**.

The platform allows users to upload unstructured institutional documents (PDF, DOCX, TXT), organize them into isolated knowledge collections, parse and vectorize chunks with local embeddings, and perform question answering with auditable citations, sentence-level grounding verification, and dynamic execution tracing.

---

## Problem Statement

Conventional RAG operates as a rigid single-pass pipeline: `Query -> Embed -> Retrieve Top-K -> Prompt -> Generate`. This approach exhibits severe vulnerabilities in institutional and technical environments:
1. **False Confidence & Hallucinations:** When retrieved documents contain irrelevant or incomplete information, standard LLMs hallucinate plausible-sounding answers or pull ungrounded external training data.
2. **Failure on Multi-Part Queries:** Complex comparative or multifaceted questions require decomposed information segments across multiple sections or policies, which single-pass vector searches frequently miss.
3. **Black-Box Opacity:** Users and auditors receive answers without insight into retrieval confidence, subquery coverage, or verification provenance.
4. **Framework Bloat & Fragility:** Heavy orchestration frameworks (LangChain, LangGraph) introduce complex abstractions, vendor lock-in, and unpredictable execution paths.

---

## Project Objectives

1. **Deterministic Control:** Build an Agentic RAG system governed by a plain-Python state machine with explicit Pydantic v2 schemas and strict transition guards.
2. **Self-Evaluating Retrieval:** Implement an `EvidenceEvaluator` that scores chunk relevance, lexical keyword overlap, and subquery coverage before deciding whether evidence is sufficient to synthesize an answer.
3. **Adaptive Query Refinement:** Implement a `QueryRefiner` capable of targeting uncovered subqueries, relaxing similarity thresholds, and broadening retrieval scope across bounded iterations (maximum 3 total rounds).
4. **Sentence-Level Grounding:** Ensure every generated factual claim is automatically verified against cited chunk text, flagging hallucinated citations.
5. **Full System Observability:** Surface an interactive React 19 UI featuring live lifecycle timelines, state inspection, and source verification drawers.

---

## Key Features

- **Dual RAG Engine:**
  - **Conventional RAG:** Rapid single-pass semantic search, context building within strict character budgets, prompt construction, and citation validation.
  - **Agentic RAG:** Multi-step autonomous state machine: Query Understanding -> Decomposition -> Planning -> Tool Selection -> Retrieval -> Sufficiency Gating -> Adaptive Refinement -> Grounded Synthesis -> Verification.
- **Secure Tool Registry:**
  - Registered tools: `semantic_search`, `metadata_filter`, and `calculator`.
  - The arithmetic `calculator` uses a **Safe AST Allowlist** (strictly zero `eval()`, zero `exec()`, and bounded operation magnitude).
- **Institutional Multi-Document Isolation:**
  - Strict collection boundaries prevent cross-domain information leakage.
  - Automatic handling of conflicting policies or superseded document revisions with explicit comparison directives.
- **Rule-Based Safe Refusal:**
  - When evidence is insufficient after adaptive retrieval rounds, the system abstains safely with zero LLM inference calls, saving cost and eliminating hallucinations.
- **Persistent Chat History:**
  - Full multi-turn conversation persistence in SQLite linked to knowledge collections.
- **High-Performance React UI:**
  - Document upload with drag-and-drop, collection management, real-time agent execution timeline, and citation drill-down drawer.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                     React 19 + TypeScript Frontend                     │
│    (Dashboard, Collections, Document Management, Chat, Trace Drawer)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP REST / SSE Events
┌───────────────────────────────────▼────────────────────────────────────┐
│                         FastAPI Backend Server                         │
├───────────────────────────────────┬────────────────────────────────────┤
│         Conventional RAG          │            Agentic RAG             │
│  - Single-Pass Vector Retrieval   │  - Plain-Python State Machine      │
│  - Character Budgeting            │  - Query Decomposition & Planning  │
│  - Injection-Resistant Prompts    │  - Tool Registry (Search, Math)    │
│  - Direct Provenance Validation   │  - Evidence Sufficiency Evaluator  │
│                                   │  - Adaptive Query Refiner          │
│                                   │  - Sentence-Level Answer Verifier  │
├───────────────────────────────────┴────────────────────────────────────┤
│                          Core Infrastructure                           │
│  - Document Parsers: PyMuPDF (PDF), python-docx (DOCX), UTF-8/Latin-1  │
│  - Embeddings: BAAI/bge-small-en-v1.5 (384-dim, CPU execution)         │
│  - Vector Store: ChromaDB (PersistentClient with Cosine Similarity)    │
│  - Relational Database: SQLite with SQLAlchemy 2.0 Async               │
│  - LLM Runtime: Local Ollama (Qwen 2.5) / Anthropic Claude API         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Agentic RAG State Machine Workflow

The Agentic RAG system progresses through deterministic lifecycle states:

```
[INITIALIZING]
      │
      ▼
  [PLANNING] ───────► Query intent analyzed & subqueries decomposed
      │
      ▼
 [RETRIEVING] ──────► Approved tools selected & executed via ToolRegistry
      │
      ▼
 [EVALUATING] ──────► Multi-criteria sufficiency score & subquery coverage ratio
      │
      ├──────────────────┐
      ▼ (Insufficient)   ▼ (Sufficient)
  [REFINING]       [SYNTHESIZING]
      │                  │
      │ (Up to 2x)       ▼
      └────────────► [VERIFYING] ──► Sentence-level grounding & citation check
                         │
                         ▼
                    [COMPLETED]
```

---

## Technology Stack

| Layer | Component | Version | Purpose |
|---|---|---|---|
| **Backend** | Python | 3.13.3 | Core runtime |
| | FastAPI | 0.111.0 | Asynchronous REST API framework |
| | Pydantic | 2.14.3 | Type contracts, input validation, schemas |
| | SQLAlchemy | 2.0.49 | Asynchronous relational ORM with SQLite |
| | aiosqlite | 0.20.0 | Async SQLite driver |
| **Embeddings & Vector** | Sentence Transformers | 3.3.1 | Embedding generation |
| | Model | BAAI/bge-small-en-v1.5 | 384-dimensional dense semantic vectors |
| | ChromaDB | 1.5.9 | Embedded local vector database |
| **Document Processing** | PyMuPDF | 1.24.10+ | High-speed PDF text and layout extraction |
| | python-docx | 1.1.2+ | DOCX document and table parser |
| **Frontend** | React | 19.0 | User interface components |
| | TypeScript | 5.9 | Strongly-typed client codebase |
| | Vite | 8.3 | Ultra-fast frontend bundler |
| | Tailwind CSS | 4.0 | Responsive utility-first styling |
| | Lucide React | 1.16 | Modern UI iconography |
| **LLM Support** | Ollama | Local | Default open-weights execution (Qwen 2.5) |
| | Anthropic API | Optional | Cloud LLM integration (Claude 3.5 Sonnet) |

---

## Repository Folder Structure

```
├── .env.example                     # Environment template
├── .gitignore                        # Strict rules excluding DBs, uploads, secrets
├── AGENTS.md                         # Authoritative Agentic RAG architectural specification
├── README.md                         # Comprehensive project documentation
├── pyproject.toml                    # Pytest configuration and markers
├── requirements.txt                  # Pinned backend dependencies
├── backend/
│   ├── main.py                       # FastAPI application entrypoint & lifespan
│   ├── config.py                     # Centralized Pydantic application settings
│   ├── database.py                   # Async SQLite connection and engine
│   ├── models/                       # SQLAlchemy ORM models (Collection, Document, Chat)
│   ├── routers/                      # REST API routers (Agent, RAG, Collections, Docs, Health)
│   ├── schemas/                      # Pydantic v2 schemas (AgentState, Evidence, Tools, API)
│   └── services/                     # Core business logic services:
│       ├── agent_controller.py       # Deterministic plain-Python state machine
│       ├── answer_generator.py       # Unified LLM provider interface
│       ├── answer_verifier.py        # Sentence-level grounding verifier
│       ├── citation_validator.py     # Deterministic citation parser and provenance check
│       ├── collection_lock.py        # Async concurrency lock manager
│       ├── context_builder.py        # Token/character bounded context assembler
│       ├── document_parser.py        # PDF, DOCX, TXT parsers
│       ├── document_processor.py     # Document indexing and vectorization pipeline
│       ├── document_recovery.py      # Resilient deletion and crash recovery
│       ├── embedding_service.py      # Pre-flight RAM checked local embedding generator
│       ├── evidence_evaluator.py     # Rule-based sufficiency gate & coverage auditor
│       ├── file_storage.py           # Streamed 50MB-capped file storage & path sanitizer
│       ├── prompt_builder.py         # Injection-resistant RAG prompt constructor
│       ├── query_planner.py          # Execution planner
│       ├── query_refiner.py          # Multi-strategy adaptive query refiner
│       ├── query_understanding.py    # Intent classifier and subquery decomposer
│       ├── retrieval_service.py      # Filtered semantic search service
│       ├── tool_registry.py          # Typed ToolRegistry with safe AST arithmetic
│       └── vector_store.py           # ChromaDB wrapper
├── frontend/
│   ├── package.json                  # Frontend dependencies
│   ├── vite.config.ts                # Vite configuration
│   └── src/
│       ├── App.tsx                   # Main router and app shell
│       ├── api/                      # Axios/fetch API clients (Collections, Docs, RAG)
│       ├── components/               # Modular UI components (Chat, Drawers, Timeline)
│       ├── pages/                    # Views (Dashboard, Chat, Collections, Evaluation)
│       └── types/                    # TypeScript interfaces
├── demo_output/
│   └── screenshots/                  # Verified UI screenshots
├── docs/                             # Phase-by-phase architectural audits and research notes
└── tests/                            # 40 comprehensive automated test suites (509 tests)
```

---

## Installation & Setup

### Prerequisites
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** 3.13.x installed
- **Node.js:** 18.x or 20.x installed with `npm`
- **Git**

---

### 1. Backend Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/intelligent-knowledge-assistant-agentic-rag.git
cd intelligent-knowledge-assistant-agentic-rag

# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# Install CPU-only PyTorch (crucial for CPU-first execution without CUDA bloat)
python -m pip install torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install all required backend dependencies
python -m pip install -r requirements.txt
```

---

### 2. Frontend Environment Setup

```bash
cd frontend
npm install
cd ..
```

---

### 3. Environment Configuration

Copy `.env.example` to create your local `.env` file:

```bash
cp .env.example .env
```

The default settings are preconfigured for local offline development.

---

## Running the Application

### 1. Start the FastAPI Backend

From the repository root:
```bash
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- API Health Status: `http://127.0.0.1:8000/health`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`
- ReDoc Documentation: `http://127.0.0.1:8000/redoc`

### 2. Start the React Frontend

In a separate terminal:
```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```
- Open your browser at: `http://127.0.0.1:5173`

---

## LLM Provider Configuration

The Intelligent Knowledge Assistant supports both local offline execution and cloud API execution.

### Option A: Local Ollama (Default & Recommended)
1. Download and install [Ollama](https://ollama.com/).
2. Pull the default lightweight model:
   ```bash
   ollama pull qwen2.5:3b
   ```
3. In `.env`:
   ```env
   LLM_PROVIDER=ollama
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_MODEL=qwen2.5:3b
   ```

### Option B: Cloud Anthropic Claude API (Optional)
1. Acquire an API key from Anthropic Console.
2. In `.env`:
   ```env
   LLM_PROVIDER=anthropic
   ANTHROPIC_API_KEY=sk-ant-your-actual-api-key-here
   CLAUDE_MODEL=claude-3-5-sonnet-20241022
   ```

---

## Executing Automated Tests

The codebase includes an extensive suite of 40 test files covering unit, integration, concurrency, security, and edge-case behaviors:

```bash
# Run the complete test suite (509 passing tests, excluding heavy model weights)
.\.venv\Scripts\python.exe -m pytest -m "not real_model" -q

# Run dedicated Agent Decision-Making & Adaptive Retrieval tests
.\.venv\Scripts\python.exe -m pytest tests/test_agent_decision_making.py -v

# Run Document Processing, Security, & Upload tests
.\.venv\Scripts\python.exe -m pytest tests/test_documents.py tests/test_document_deletion_router.py tests/test_document_processor.py -v

# Run Concurrency, Locking, & Grounding tests
.\.venv\Scripts\python.exe -m pytest tests/test_collection_concurrency.py tests/test_college_grounding_regression.py -v

# Verify Frontend Production Build & TypeScript Types
npm run build --prefix frontend
```

---

## API Documentation

The FastAPI backend automatically generates interactive documentation accessible at:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

### Primary Endpoints:
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check (API status, database connection, uptime) |
| `GET` | `/api/collections` | List all document knowledge collections |
| `POST` | `/api/collections` | Create a new isolated knowledge collection |
| `POST` | `/api/collections/{id}/documents` | Upload and vectorize a PDF, DOCX, or TXT file (50MB cap) |
| `DELETE` | `/api/documents/{id}` | Durably delete document and remove vectors from ChromaDB |
| `POST` | `/api/agent/run` | Execute the Agentic RAG State Machine with full event tracing |
| `POST` | `/api/rag/answer` | Execute Conventional single-pass RAG |
| `GET` | `/api/chat/{collection_id}/history` | Retrieve persistent multi-turn chat history |
| `POST` | `/api/evaluation/run` | Execute comparative benchmark between Conventional and Agentic RAG |

---

## Application Screenshots

Genuine interface screenshots captured from the running application:

### 1. Dashboard Overview
![Dashboard Overview](demo_output/screenshots/01_dashboard.png)
*Central dashboard displaying collections, indexed documents, and system health status.*

### 2. Knowledge Collections Management
![Knowledge Collections](demo_output/screenshots/02_collections_list.png)
*Collection listing showing isolated document boundaries and document counts.*

### 3. Collection Detail & Document Explorer
![Collection Detail](demo_output/screenshots/03_collection_detail_college.png)
*Document management showing indexed college documents, statuses, and chunk counts.*

### 4. Interactive Grounded Chat Interface
![Grounded Chat](demo_output/screenshots/05_chat_cse_hod_grounded.png)
*Factual question answering strictly grounded in uploaded college documents with clickable citations.*

### 5. Verified Source Drawer
![Source Drawer](demo_output/screenshots/06_chat_source_drawer.png)
*Detailed source provenance inspection showing chunk text, similarity score, and metadata.*

### 6. Agent Execution Trace Drawer
![Trace Drawer](demo_output/screenshots/07_chat_trace_drawer.png)
*Step-by-step lifecycle event trace showing query intent, subqueries, and sufficiency gating.*

### 7. Safe Rule-Based Abstention on Unanswerable Queries
![Safe Refusal](demo_output/screenshots/08_chat_unanswerable_refusal.png)
*Safe refusal without hallucinations when evidence is absent from uploaded institutional files.*

---

## Empirical Research Benchmark

Evaluated on the standardized `synthetic-university-policies-v1` evaluation benchmark corpus:

| Evaluation Metric | Conventional RAG (Single-Pass) | Agentic RAG (Iterative State Machine) | Empirical Difference |
|---|---|---|---|
| **Hit Rate@5** | 90.0% | **100.0%** | **+10.0%** |
| **Mean Reciprocal Rank (MRR)** | 0.767 | **0.900** | **+0.133** |
| **Precision@5** | 22.0% | **35.0%** | **+13.0%** |
| **Recall@5** | 95.0% | **100.0%** | **+5.0%** |
| **Sentence Grounding Verification** | 76.0% (unverified reference) | **92.5% (sentence-grounded)** | **+16.5%** |
| **Mean Latency (ms)** | **39.65 ms** | 124.50 ms | $3.14\times$ trade-off |

### Benchmark Conclusion:
Agentic RAG demonstrates higher recall, hit rate, and verified sentence grounding over conventional single-pass RAG—especially on multi-part, cross-chunk, and comparative queries—in exchange for a predictable $3\times$ latency trade-off due to multi-step retrieval and sufficiency auditing.

---

## Limitations & Future Roadmap

- **Subquery Coverage Ratio:** The current `min_coverage_ratio = 0.60` effectively triggers refinement when 1 of 2 subqueries is missing (0.50 < 0.60), but for $N \ge 3$ subqueries (e.g. 2/3 = 66.7%), partial coverage meets the threshold. Future versions can support configurable strict 100% coverage gates.
- **Multimodal Support:** Currently optimized for text in PDF, DOCX, and TXT files. OCR for scanned image PDFs is gracefully rejected via `ScannedPDFError`.
- **Hybrid Search:** Future enhancements may integrate BM25 sparse keyword indices with ChromaDB dense vectors for hybrid retrieval.

---

## License & Compliance

Developed under strict enterprise software engineering standards:
- **No LangChain or LangGraph:** Pure Python state machine architecture.
- **Security-First:** Safe AST allowlists for arithmetic; no arbitrary code execution.
- **Data Privacy:** Local execution on CPU; user documents and database files remain strictly on the host system.
