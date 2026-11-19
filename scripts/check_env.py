"""
scripts/check_env.py
Phase 1 — Environment Verification Script

Run this script to verify that all required packages are present and
the Python version is compatible before starting each phase.

Usage:
    python scripts/check_env.py
"""

import sys
import importlib
import subprocess
from typing import NamedTuple


class CheckResult(NamedTuple):
    name: str
    required: bool
    phase: int
    status: str       # OK | MISSING | WRONG_VERSION | WARN
    version: str
    message: str


REQUIRED_PHASE_1 = [
    # (import_name, pip_name, min_version, phase)
    ("fastapi",          "fastapi",          "0.111.0", 1),
    ("uvicorn",          "uvicorn",          "0.30.0",  1),
    ("pydantic",         "pydantic",         "2.0.0",   1),
    ("pydantic_settings","pydantic-settings","2.0.0",   1),
    ("sqlalchemy",       "sqlalchemy",       "2.0.0",   1),
    ("httpx",            "httpx",            "0.27.0",  1),
    ("streamlit",        "streamlit",        "1.30.0",  1),
    ("dotenv",           "python-dotenv",    "1.0.0",   1),
    ("pytest",           "pytest",           "7.0.0",   1),
    ("rich",             "rich",             "13.0.0",  1),
    ("tenacity",         "tenacity",         "8.0.0",   1),
    ("anyio",            "anyio",            "4.0.0",   1),
]

LATER_PHASES = [
    ("pymupdf",              "pymupdf",             "1.24.0", 3),
    ("docx",                 "python-docx",         "1.1.0",  3),
    ("sentence_transformers","sentence-transformers","3.0.0",  5),
    ("chromadb",             "chromadb",            "0.5.0",  6),
    ("anthropic",            "anthropic",           "0.30.0", 9),
    ("pytest_asyncio",       "pytest-asyncio",      "0.23.0", 2),
    ("aiosqlite",            "aiosqlite",           "0.19.0", 2),
]

PYTHON_MIN = (3, 11)
PYTHON_MAX = (3, 14)
PYTHON_IDEAL = (3, 13)


def check_python() -> None:
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v < PYTHON_MIN:
        print(f"  [FAIL] Python {version_str} — minimum required: {PYTHON_MIN[0]}.{PYTHON_MIN[1]}")
        sys.exit(1)
    elif v[:2] == PYTHON_IDEAL:
        print(f"  [OK]   Python {version_str} (ideal: 3.13 — using this environment)")
    elif v[:2] == (3, 11):
        print(f"  [OK]   Python {version_str} — project specifies 3.11; consider a venv if needed")
    else:
        print(f"  [WARN] Python {version_str} — project specifies 3.11; running on 3.13 (compatible)")


def check_package(import_name: str, pip_name: str, min_ver: str, phase: int, required: bool) -> CheckResult:
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        return CheckResult(pip_name, required, phase, "OK", version, "")
    except ImportError:
        status = "MISSING" if required else "INFO"
        return CheckResult(pip_name, required, phase, status, "-",
                           f"Install at Phase {phase}: pip install {pip_name}")


def run_checks() -> None:
    print("=" * 60)
    print("  Intelligent Knowledge Assistant — Environment Check")
    print("=" * 60)

    print("\n[1] Python Version")
    check_python()

    print("\n[2] Phase 1 Required Packages")
    results_p1 = []
    for imp, pip, minv, ph in REQUIRED_PHASE_1:
        r = check_package(imp, pip, minv, ph, required=True)
        results_p1.append(r)
        icon = "OK  " if r.status == "OK" else "FAIL"
        print(f"  [{icon}] {r.name:<30} {r.version:<15} {r.message}")

    print("\n[3] Later-Phase Packages (not required now)")
    results_later = []
    for imp, pip, minv, ph in LATER_PHASES:
        r = check_package(imp, pip, minv, ph, required=False)
        results_later.append(r)
        icon = "READY" if r.status == "OK" else f"Phase {r.phase}"
        print(f"  [{icon:<5}] {r.name:<30} {r.version:<15} {r.message}")

    # Summary
    p1_fails = [r for r in results_p1 if r.status != "OK"]
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    if not p1_fails:
        print("  Phase 1 packages: ALL OK")
    else:
        print(f"  Phase 1 packages: {len(p1_fails)} MISSING")
        for r in p1_fails:
            print(f"    - {r.name}: pip install {r.name}")

    later_ready = [r for r in results_later if r.status == "OK"]
    print(f"  Later-phase packages already installed: {len(later_ready)}/{len(results_later)}")

    # LangChain warning
    print("\n[4] Restricted Packages Check")
    for pkg in ["langchain", "langgraph"]:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  [WARN] {pkg} {ver} is installed — do NOT import in this project.")
        except ImportError:
            print(f"  [OK]   {pkg} not imported (present in env but not used by this project)")

    print("\n  Environment check complete.")


if __name__ == "__main__":
    run_checks()
