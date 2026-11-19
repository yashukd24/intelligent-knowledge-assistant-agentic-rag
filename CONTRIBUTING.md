# Contributing to Intelligent Knowledge Assistant

Thank you for your interest in contributing to the **Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models**.

## Architectural Guarantees & Strict Rules

Before submitting pull requests or modifying code, please review the core architectural constraints outlined in `AGENTS.md`:

1. **Strictly Zero LangChain / LangGraph:**
   - All agent behaviors, state transitions, and tool executions MUST be implemented in plain Python with deterministic state machines and Pydantic v2 schemas.
   - Do not introduce LangChain, LangGraph, LlamaIndex, or other monolithic orchestration frameworks.

2. **Security & Safe Tool Execution:**
   - Tools must never execute arbitrary code (`eval`, `exec`, shell subprocesses, or unrestricted filesystem access).
   - Any mathematical evaluation must use the existing Safe AST Allowlist in `backend/services/tool_registry.py`.

3. **CPU-First Execution:**
   - The application is optimized for CPU-only execution without requiring GPU hardware. Keep embeddings and local LLM runtime compatible with consumer laptop/desktop CPUs.

4. **Testing & Code Quality:**
   - All new features, bug fixes, and routers must be covered by unit and integration tests under `tests/`.
   - The test suite must pass 100% of tests (`pytest -m "not real_model"`).
   - Frontend changes must pass TypeScript validation (`npm run build --prefix frontend`).

## Development Workflow

1. Fork or branch from `master`.
2. Activate your virtual environment: `.\.venv\Scripts\Activate.ps1` (or `source .venv/bin/activate`).
3. Implement changes with clean, typed code.
4. Run regression tests: `pytest -m "not real_model" -q`.
5. Verify frontend build: `npm run build --prefix frontend`.
6. Submit a pull request with a detailed description of your changes and test verification results.
