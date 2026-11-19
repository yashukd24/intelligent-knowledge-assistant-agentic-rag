# Phase 9: Real Claude API Smoke Test Runbook & Endpoint Verification

**Project:** Intelligent Knowledge Assistant Using Agentic RAG and Large Language Models  
**Document:** Verified Smoke Test Runbook & Endpoint Specifications  
**Date:** November 10, 2026  
**Status:** **VERIFIED & READY — Awaiting Explicit Approval Before Real Claude API Call**

---

## 1. Verified Architecture & Endpoint Registry

All routes have been verified by inspecting `backend/main.py` and the registered router definitions:

| Operation | HTTP Method & Verified Path | Router Module | Request Schema / Form Field | Response Model & Key Fields |
|---|---|---|---|---|
| **Document Upload** | `POST /api/collections/{collection_id}/documents` | `backend/routers/documents.py` | Multipart form field: `file` | `DocumentResponse` (`id`, `status="pending"`, `version=1`) |
| **Document Processing & Indexing** | `POST /api/documents/{document_id}/process` | `backend/routers/documents.py` | Query parameter: `embed=true` (default) | `ProcessedDocument` (`status="indexed"`, `chunk_count`, `page_count`) |
| **Semantic Retrieval** | `POST /api/collections/{collection_id}/query` | `backend/routers/retrieval.py` | `RetrievalQueryRequest` (`query`, `top_k`, `similarity_threshold`) | `RetrievalResponse` (`returned_results`, `total_candidates`, `results`) |
| **Grounded RAG Answer** | `POST /api/collections/{collection_id}/rag/answer` | `backend/routers/rag_answer.py` | `RAGAnswerRequest` (`query`, `top_k`, `similarity_threshold`, `max_context_chars`) | `RAGAnswerResponse` (`status`, `answer`, `valid_citations`, `usage`, `latency`) |

### Key Schema Field Verifications:
- **Retrieval Result Count Field:** The exact field name in `RetrievalResponse` is **`returned_results`** (integer count of candidate chunks passing threshold filtering). `results` is the list of `RetrievedChunk` items. `total_candidates` is the count of raw nearest neighbors before threshold filtering.
- **Authoritative Collection ID:** All collection-scoped endpoints enforce the URL path `collection_id` as authoritative; conflicting IDs in request bodies return HTTP `422 Unprocessable Entity`.
- **Target Collection Validation:** Every endpoint verifies collection existence in SQLite; nonexistent IDs return HTTP `404 Not Found`.

---

## 2. Model Configuration Audit

- **Retired Legacy Model:** `claude-3-5-sonnet-20241022` and `claude-3-5-sonnet-20240620` are retired by Anthropic.
- **Active Proposed Model ID:** **`claude-sonnet-5`** is configured as the active default in `backend/config.py` and `.env.example`.
- **Dynamic Configuration:** Resolved via `ANTHROPIC_MODEL` in `.env`, falling back to `backend/config.py`:
  ```bash
  ANTHROPIC_MODEL=claude-sonnet-5
  ```

---

## 3. Safe API Key Verification (Zero Secrets Printed)

In accordance with strict security requirements, **secrets, key lengths, prefixes, and partial tokens are never printed or logged**.

Run this command in PowerShell:
```powershell
.\.venv\Scripts\python.exe -c "
from backend.config import get_settings
s = get_settings()
key = s.anthropic_api_key
is_configured = 'YES' if (key and key.strip()) else 'NO'
is_placeholder = 'YES' if (key in ('sk-ant-REPLACE_WITH_YOUR_KEY', 'your_anthropic_api_key_here')) else 'NO'
print(f'API Key Configured: {is_configured}')
print(f'Placeholder Detected: {is_placeholder}')
"
```

**Required Output Before Live Testing:**
```
API Key Configured: YES
Placeholder Detected: NO
```

---

## 4. PowerShell 5.1 Compatibility for Multipart Upload

The host operating system runs **PowerShell 5.1** (`5.1.26100.9818`). In PowerShell 5.1, `Invoke-RestMethod` does **not** support the `-Form` parameter (which requires PowerShell Core 6.0+).

Two fully compatible upload methods are provided for Windows PowerShell 5.1:

### Method A (Recommended): Built-in Windows `curl.exe`
Windows 10 and 11 include `C:\Windows\System32\curl.exe` natively:
```powershell
$doc = curl.exe -s -X POST "http://127.0.0.1:8000/api/collections/$colId/documents" -F "file=@smoke_test_knowledge.pdf" | ConvertFrom-Json
$docId = $doc.id
Write-Host "Uploaded Document ID: $docId, Status: $($doc.status)"
```

### Method B: Native .NET `HttpClient` (Pure PowerShell 5.1)
```powershell
Add-Type -AssemblyName System.Net.Http
$httpClient = [System.Net.Http.HttpClient]::new()
$content = [System.Net.Http.MultipartFormDataContent]::new()
$fileStream = [System.IO.File]::OpenRead((Resolve-Path ".\smoke_test_knowledge.pdf"))
$fileContent = [System.Net.Http.StreamContent]::new($fileStream)
$content.Add($fileContent, "file", "smoke_test_knowledge.pdf")
$response = $httpClient.PostAsync("http://127.0.0.1:8000/api/collections/$colId/documents", $content).Result
$doc = $response.Content.ReadAsStringAsync().Result | ConvertFrom-Json
$fileStream.Close()
$httpClient.Dispose()
$docId = $doc.id
Write-Host "Uploaded Document ID: $docId, Status: $($doc.status)"
```

---

## 5. End-to-End Live Smoke Test Procedure

When you are ready to authorize the live Claude API request, follow these sequential steps in PowerShell:

### Step 5.1: Launch Backend Server
In Terminal 1:
```powershell
.\.venv\Scripts\uvicorn.exe backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

---

### Step 5.2: Create Dedicated Smoke Test Collection
In Terminal 2:
```powershell
$col = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/collections" `
  -ContentType "application/json" `
  -Body '{"name": "Phase 9 Smoke Test Collection", "description": "Temporary collection for real API smoke test"}'

$colId = $col.id
Write-Host "Created Smoke Test Collection ID: $colId"
```

---

### Step 5.3: Generate Test Multi-Page PDF (`smoke_test_knowledge.pdf`)
Run this snippet to generate a 2-page PDF with verifiable facts using PyMuPDF:

```powershell
.\.venv\Scripts\python.exe -c "
import fitz

doc = fitz.open()

# Page 1: Architecture Facts
p1 = doc.new_page()
p1_text = (
    'Antigravity Knowledge Assistant Architecture Specifications\n\n'
    'Section 1: Database and Storage\n'
    'The Antigravity system was engineered in 2026 as a dual-engine agentic RAG platform.\n'
    'All document metadata and state transitions are persisted in SQLite using SQLAlchemy 2.0 async ORM.\n'
    'Document lifecycle states strictly follow: pending, processing, indexed, failed, or deletion_pending.\n\n'
    'Section 2: Vector Embeddings\n'
    'Vector representations are generated using the sentence transformer model BAAI/bge-small-en-v1.5.\n'
    'The model produces normalized dense embeddings with an exact dimensionality of 384.\n'
    'Vector storage and similarity queries are hosted by an embedded ChromaDB PersistentClient.'
)
p1.insert_text((50, 72), p1_text, fontsize=10)

# Page 2: RAG Pipeline Specifications
p2 = doc.new_page()
p2_text = (
    'Conventional RAG Pipeline and Grounded Answer Generation\n\n'
    'Section 3: Context Assembly\n'
    'The conventional RAG pipeline formats retrieved chunks into numbered source identifiers [S1], [S2].\n'
    'The default character budget for context assembly is 12,000 characters.\n\n'
    'Section 4: LLM Generation and Citation Verification\n'
    'Source-grounded answers are generated using Anthropic Claude models.\n'
    'The citation validator verifies reference integrity by checking that all cited tags exist in context.\n'
    'If retrieval yields zero evidence, the service immediately bypasses Claude to prevent hallucination.'
)
p2.insert_text((50, 72), p2_text, fontsize=10)

doc.save('smoke_test_knowledge.pdf')
doc.close()
print('Generated smoke_test_knowledge.pdf (2 pages).')
"
```

---

### Step 5.4: Upload and Index the PDF
Upload the PDF using PowerShell 5.1 compatible `curl.exe`:
```powershell
$doc = curl.exe -s -X POST "http://127.0.0.1:8000/api/collections/$colId/documents" -F "file=@smoke_test_knowledge.pdf" | ConvertFrom-Json
$docId = $doc.id
Write-Host "Uploaded Document ID: $docId (Initial Status: $($doc.status))"
```

Trigger processing (text extraction, cleaning, chunking, BGE embedding, and ChromaDB vector indexing):
```powershell
$proc = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/documents/$docId/process"
Write-Host "Processing complete! Status: $($proc.status), Chunks: $($proc.chunk_count), Pages: $($proc.page_count)"
```

Verify that the document record in SQLite reflects the `indexed` status:
```powershell
$check = Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/documents/$docId"
Write-Host "Verified Document Status: $($check.status)" # Must be "indexed"
```

---

### Step 5.5: Semantic Retrieval Pre-Check (Matching Question)
Inspect the semantic retrieval endpoint directly to confirm vector indexing and candidate chunk quality:

```powershell
$retPayload = @{
    query = "What embedding model and vector dimension does the system use?"
    top_k = 3
    similarity_threshold = 0.3
} | ConvertTo-Json

$retResp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/collections/$colId/query" `
  -ContentType "application/json" `
  -Body $retPayload

Write-Host "Returned Candidates: $($retResp.returned_results)" # Expect >= 1
Write-Host "Top Chunk Score: $($retResp.results[0].similarity_score)"
Write-Host "Top Chunk Text: $($retResp.results[0].text.Substring(0, 80))..."
```

---

### Step 5.6: Grounded Answer Generation (Real Claude API Call)
Submit the natural-language question to the RAG answer endpoint:

```powershell
$groundedPayload = @{
    query = "What embedding model and vector dimension does the system use?"
    top_k = 3
    similarity_threshold = 0.3
} | ConvertTo-Json

$answerResp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/collections/$colId/rag/answer" `
  -ContentType "application/json" `
  -Body $groundedPayload

$answerResp | ConvertTo-Json -Depth 5
```

#### Expected Successful Response Structure:
```json
{
  "query": "What embedding model and vector dimension does the system use?",
  "answer": "According to the documentation [S1], the system generates vector embeddings using the sentence transformer model BAAI/bge-small-en-v1.5 with an exact dimensionality of 384.",
  "status": "answered",
  "collection_id": "<collection-uuid>",
  "sources": [
    {
      "source_id": "[S1]",
      "document_id": "<document-uuid>",
      "chunk_id": "<chunk-uuid>",
      "filename": "smoke_test_knowledge.pdf",
      "page_number": 1,
      "chunk_index": 0,
      "similarity_score": 0.7255,
      "char_count": 480
    }
  ],
  "valid_citations": ["[S1]"],
  "invalid_citations": [],
  "uncited_sources": [],
  "provider": "anthropic",
  "model": "claude-sonnet-5",
  "usage": {
    "prompt_tokens": 210,
    "completion_tokens": 38,
    "total_tokens": 248
  },
  "generation_latency_ms": 820.4,
  "total_latency_ms": 890.1,
  "retrieval_metadata": {
    "candidate_count": 1,
    "top_k": 3
  },
  "error_message": null,
  "limitations_note": "Citation validation verifies reference existence, not semantic entailment."
}
```

---

### Step 5.7: Insufficient Evidence Test (Zero Tokens / Zero Claude Calls)

To guarantee that Claude is **not** called on unanswerable questions, we first inspect the retrieval endpoint directly to verify that zero candidate chunks meet the threshold.

#### Step 5.7.1: Inspect Retrieval Endpoint Directly
```powershell
$checkRetrievalPayload = @{
    query = "What is the biological migratory pattern of the Arctic Tern in Greenland?"
    top_k = 3
    similarity_threshold = 0.85
} | ConvertTo-Json

$retrievalResp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/collections/$colId/query" `
  -ContentType "application/json" `
  -Body $checkRetrievalPayload

Write-Host "Candidate Chunks Returned: $($retrievalResp.returned_results)"
# Verified: returned_results must be 0
```

#### Step 5.7.2: Execute Answer Endpoint with Empty Evidence
```powershell
$refusalResp = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/api/collections/$colId/rag/answer" `
  -ContentType "application/json" `
  -Body $checkRetrievalPayload

$refusalResp | ConvertTo-Json -Depth 5
```

#### Expected Insufficient Evidence Response:
```json
{
  "query": "What is the biological migratory pattern of the Arctic Tern in Greenland?",
  "answer": "The provided documents do not contain sufficient information to answer this question.",
  "status": "insufficient_evidence",
  "collection_id": "<collection-uuid>",
  "sources": [],
  "valid_citations": [],
  "invalid_citations": [],
  "uncited_sources": [],
  "provider": "rule_based",
  "model": "none",
  "usage": null,
  "generation_latency_ms": 0.0,
  "total_latency_ms": 15.3,
  "retrieval_metadata": {
    "candidate_count": 0,
    "top_k": 3
  },
  "error_message": null,
  "limitations_note": "Citation validation verifies reference existence, not semantic entailment."
}
```
*(Notice: `usage: null` and `generation_latency_ms: 0.0`, proving that zero Claude API calls were made and zero tokens were consumed).*

---

## 6. Targeted Cleanup

Delete **only** the temporary smoke-test collection. Cascading database and filesystem cleanup ensures that its documents and ChromaDB embeddings are completely expunged without affecting existing user data:

```powershell
# 1. Delete the temporary smoke test collection
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8000/api/collections/$colId"

# 2. Verify collection is deleted (Expect HTTP 404)
try {
    Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/api/collections/$colId"
    Write-Host "WARNING: Collection still exists!"
} catch {
    Write-Host "VERIFIED: Collection deleted successfully (HTTP 404)."
}

# 3. Remove local test PDF
Remove-Item -Force .\smoke_test_knowledge.pdf
Write-Host "VERIFIED: Local test PDF removed."
```

---

## 7. Verification Evidence & Summary

1. **Local Pipeline Pre-Flight Test:** Executed full local pipeline test with temporary collection, 2-page PDF generation, multipart upload, processing, vector indexing, grounded retrieval (`returned_results: 3`), insufficient evidence retrieval (`returned_results: 0`), and empty-evidence bypass (`usage: null`). **PASSED.**
2. **Phase 9 Test Suite:** 52 / 52 unit and router tests passing.
3. **Production Data State:** `data/knowledge_assistant.db` (0 collections, 0 documents), `data/chromadb/chroma.sqlite3` (0 collections), `data/uploads/` (0 files).
4. **Paid API Calls Made:** **ZERO (0).**
5. **Phase 10 Status:** **NOT STARTED.**

The runbook is fully verified. The system is paused awaiting your explicit authorization before running the first paid Claude API request.
