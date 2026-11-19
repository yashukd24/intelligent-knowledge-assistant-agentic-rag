"""
backend/evaluation/evaluation_dataset.py
─────────────────────────────────────────────────────────────
Controlled synthetic evaluation dataset for retrieval quality assessment.

Features:
  - 10 distinct test cases spanning all required query types.
  - Deterministic synthetic university policy corpus (4 documents, 10 chunks).
  - Explicit, human-authored ground-truth relevance labels.
  - Seeding utility for isolated in-memory or integration test environments.
─────────────────────────────────────────────────────────────
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from backend.evaluation.evaluation_models import EvaluationTestCase
from backend.models.collection import Collection
from backend.models.document import Document
from backend.schemas.chunk import ChunkMetadata, DocumentChunk
from backend.schemas.document import DocumentStatus
from backend.schemas.embedding import EmbeddedChunk
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStoreService

EVAL_COLLECTION_ID = "00000000-0000-4000-a000-000000000001"


# ── Synthetic Evaluation Corpus ──────────────────────────────────────────────

SYNTHETIC_DOCUMENTS: List[Dict[str, Any]] = [
    {
        "id": "doc_eval_01",
        "filename": "academic_integrity_code.pdf",
        "file_type": "pdf",
        "title": "Academic Integrity and Honor Code",
        "chunks": [
            {
                "chunk_id": "eval_doc_01_v1_c0000",
                "chunk_index": 0,
                "text": (
                    "Section 1: Academic Integrity and Plagiarism Definition. "
                    "Plagiarism is defined as the uncredited use of another person's words, "
                    "ideas, data, or creative expressions as one's own. Students must provide "
                    "explicit citations for direct quotations, paraphrased text, and statistical tables. "
                    "Unauthorized collaboration on individual assignments and purchasing essays "
                    "are severe violations of the University Honor Code."
                ),
                "page_number": 1,
                "page_range": "1",
            },
            {
                "chunk_id": "eval_doc_01_v1_c0001",
                "chunk_index": 1,
                "text": (
                    "Section 2: Sanctions and Disciplinary Proceedings. "
                    "First-time honor code infractions may result in an academic penalty of an F grade "
                    "in the assignment or course. Egregious or repeated offenses are referred to the "
                    "Dean of Students and the Academic Conduct Board, carrying sanctions including "
                    "disciplinary probation, suspension for up to two academic semesters, or permanent expulsion."
                ),
                "page_number": 2,
                "page_range": "2",
            },
            {
                "chunk_id": "eval_doc_01_v1_c0002",
                "chunk_index": 2,
                "text": (
                    "Section 3: Appeals Procedure. "
                    "A student found responsible for an academic violation may file a formal appeal "
                    "within ten business days of the written decision. The Appeals Committee requires "
                    "a quorum of four voting members (two faculty members and two student representatives). "
                    "Appeals must demonstrate procedural error, bias, or newly discovered evidence."
                ),
                "page_number": 3,
                "page_range": "3",
            },
        ],
    },
    {
        "id": "doc_eval_02",
        "filename": "graduate_degree_regulations.pdf",
        "file_type": "pdf",
        "title": "Graduate Degree Regulations and Thesis Guidelines",
        "chunks": [
            {
                "chunk_id": "eval_doc_02_v1_c0000",
                "chunk_index": 0,
                "text": (
                    "Master's Degree Requirements. "
                    "Candidates must complete a minimum of 30 semester credit hours with a cumulative GPA "
                    "of 3.0 or higher. At least 24 credit hours must be earned in residence. "
                    "Grades below B- cannot count toward graduate degree requirements, and students "
                    "falling below a 3.0 GPA face academic dismissal if not remediated within one semester."
                ),
                "page_number": 1,
                "page_range": "1",
            },
            {
                "chunk_id": "eval_doc_02_v1_c0001",
                "chunk_index": 1,
                "text": (
                    "Thesis Defense Protocol. "
                    "The master's thesis defense committee consists of a minimum of three faculty members, "
                    "including the primary research advisor and one external reader. The oral defense is public "
                    "and must be scheduled at least three weeks prior to commencement. A unanimous or majority "
                    "favorable vote is required for recommendation of the degree."
                ),
                "page_number": 2,
                "page_range": "2",
            },
        ],
    },
    {
        "id": "doc_eval_03",
        "filename": "housing_residence_handbook.txt",
        "file_type": "txt",
        "title": "Campus Housing and Residence Life Regulations",
        "chunks": [
            {
                "chunk_id": "eval_doc_03_v1_c0000",
                "chunk_index": 0,
                "text": (
                    "Residence Hall Quiet Hours and Community Standards. "
                    "Quiet hours are strictly enforced Sunday through Thursday from 10:00 PM to 8:00 AM, "
                    "and Friday through Saturday from midnight to 9:00 AM. Courtesy hours are in effect 24/7. "
                    "Excessive noise, loud audio equipment, and hallway disruptions subject the resident to warnings "
                    "and potential relocation to another hall."
                ),
                "page_number": None,
                "page_range": None,
            },
            {
                "chunk_id": "eval_doc_03_v1_c0001",
                "chunk_index": 1,
                "text": (
                    "Fire Safety and Prohibited Items. "
                    "Open flames, candles, incense, space heaters, and hot plates are strictly prohibited "
                    "in dorm rooms. Extension cords must be UL-certified surge protectors. Residence Life staff "
                    "conduct unannounced safety inspections twice per semester with 24 hours advance notification."
                ),
                "page_number": None,
                "page_range": None,
            },
            {
                "chunk_id": "eval_doc_03_v1_c0002",
                "chunk_index": 2,
                "text": (
                    "Housing Contract Termination and Refund Schedule. "
                    "Students terminating their housing contract prior to the end of the academic year "
                    "incur a $500 cancellation fee. Pro-rated refunds are granted up to the fourth week of classes; "
                    "no refunds are issued for cancellations submitted after the fourth week of the semester."
                ),
                "page_number": None,
                "page_range": None,
            },
        ],
    },
    {
        "id": "doc_eval_04",
        "filename": "scholarships_financial_aid.pdf",
        "file_type": "pdf",
        "title": "Scholarships and Merit Financial Aid Guidelines",
        "chunks": [
            {
                "chunk_id": "eval_doc_04_v1_c0000",
                "chunk_index": 0,
                "text": (
                    "Satisfactory Academic Progress (SAP) for Financial Aid. "
                    "To maintain federal and institutional financial aid eligibility, students must achieve "
                    "a minimum 2.0 cumulative GPA and maintain a 67% pace of credit completion. Failure to meet "
                    "SAP standards places the student on Financial Aid Warning for one semester."
                ),
                "page_number": 1,
                "page_range": "1",
            },
            {
                "chunk_id": "eval_doc_04_v1_c0001",
                "chunk_index": 1,
                "text": (
                    "Presidential Fellowship Stipend Disbursement. "
                    "Presidential Research Fellows receive an annual living stipend of $28,000 disbursed in "
                    "bi-weekly installments across the 9-month academic term. Renewal requires maintaining a 3.5 GPA, "
                    "submitting an annual faculty research progress report, and completing 10 hours of weekly research."
                ),
                "page_number": 2,
                "page_range": "2",
            },
        ],
    },
]


# ── 10 Controlled Evaluation Test Cases ──────────────────────────────────────

SYNTHETIC_TEST_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "tc_01_factual_gpa",
        "query": "What is the minimum cumulative GPA required for graduate degree graduation?",
        "collection_id": EVAL_COLLECTION_ID,
        "expected_chunk_ids": ["eval_doc_02_v1_c0000"],
        "expected_document_ids": ["doc_eval_02"],
        "query_type": "factual",
        "description": "1. Direct factual question targeting minimum GPA in degree requirements.",
    },
    {
        "case_id": "tc_02_conceptual_integrity",
        "query": "How does the university handle academic dishonesty and integrity violations?",
        "collection_id": EVAL_COLLECTION_ID,
        "expected_chunk_ids": ["eval_doc_01_v1_c0000", "eval_doc_01_v1_c0001"],
        "expected_document_ids": ["doc_eval_01"],
        "query_type": "conceptual",
        "description": "2. Conceptual question spanning honor code definition and disciplinary sanctions.",
    },
    {
        "case_id": "tc_03_definition_plagiarism",
        "query": "What constitutes plagiarism according to the academic honor code?",
        "collection_id": EVAL_COLLECTION_ID,
        "expected_chunk_ids": ["eval_doc_01_v1_c0000"],
        "expected_document_ids": ["doc_eval_01"],
        "query_type": "definition",
        "description": "3. Definition question asking for specific definition of plagiarism.",
    },
    {
        "case_id": "tc_04_multipart_degree_defense",
        "query": "What are the required semester credit hours and the thesis defense committee composition for master's students?",
        "collection_id": EVAL_COLLECTION_ID,
        "expected_chunk_ids": ["eval_doc_02_v1_c0000", "eval_doc_02_v1_c0001"],
        "expected_document_ids": ["doc_eval_02"],
        "query_type": "multi-part",
        "description": "4. Multi-part question asking for credit hours (chunk 0) and defense committee (chunk 1).",
    },
    {
        "case_id": "tc_05_crosschunk_sanctions_appeals",
        "query": "What sanctions are imposed for honor code offenses and how can a student file an appeal?",
        "collection_id": EVAL_COLLECTION_ID,
        "expected_chunk_ids": ["eval_doc_01_v1_c0001", "eval_doc_01_v1_c0002"],
        "expected_document_ids": ["doc_eval_01"],
        "query_type": "cross-chunk",
        "description": "5. Cross-chunk question requiring sanctions details from chunk 1 and appeals procedure from chunk 2.",
    },
    {
        "case_id": "tc_06_paraphrase_dorm_noise",
        "query": "When are dormitory residents expected to maintain silence during the week?",
        "collection_id": EVAL_COLLECTION_ID,
        "expected_chunk_ids": ["eval_doc_03_v1_c0000"],
        "expected_document_ids": ["doc_eval_03"],
        "query_type": "paraphrase",
        "description": "6. Paraphrased question matching residence hall quiet hours standards.",
    },
    {
        "case_id": "tc_07_limited_info_stipend",
        "query": "How is the Presidential Research Fellowship stipend disbursed and what are the renewal rules?",
        "collection_id": EVAL_COLLECTION_ID,
        "expected_chunk_ids": ["eval_doc_04_v1_c0001"],
        "expected_document_ids": ["doc_eval_04"],
        "query_type": "limited-info",
        "description": "7. Query targeting narrow specialized topic covered in single specific chunk.",
    },
    {
        "case_id": "tc_08_unanswerable_parking",
        "query": "What are the fees for campus vehicle parking permits and shuttle bus timetables?",
        "collection_id": EVAL_COLLECTION_ID,
        "expected_chunk_ids": [],
        "expected_document_ids": [],
        "query_type": "unanswerable",
        "description": "8. Out-of-corpus query testing negative relevance rejection.",
    },
    {
        "case_id": "tc_09_similar_terms_quorum",
        "query": "What quorum is required for the Academic Appeals Committee to hold a hearing?",
        "collection_id": EVAL_COLLECTION_ID,
        "expected_chunk_ids": ["eval_doc_01_v1_c0002"],
        "expected_document_ids": ["doc_eval_01"],
        "query_type": "similar-terms",
        "description": "9. Query with similar terminology ('quorum') distinguishing appeals quorum from thesis defense committee quorum.",
    },
    {
        "case_id": "tc_10_docspecific_housing_refund",
        "query": "Under campus housing regulations, what is the cancellation penalty and refund schedule for terminating a room contract?",
        "collection_id": EVAL_COLLECTION_ID,
        "expected_chunk_ids": ["eval_doc_03_v1_c0002"],
        "expected_document_ids": ["doc_eval_03"],
        "query_type": "doc-specific",
        "description": "10. Document-specific query on housing contract termination fee and pro-rated schedule.",
    },
]


def load_evaluation_dataset(collection_id: Optional[str] = None) -> List[EvaluationTestCase]:
    """
    Load validated EvaluationTestCase Pydantic instances.

    Args:
        collection_id: Optional override collection ID. If provided, sets
                       collection_id on all returned test cases.

    Returns:
        List of validated EvaluationTestCase models.
    """
    cases: List[EvaluationTestCase] = []
    target_col = collection_id or EVAL_COLLECTION_ID
    for raw in SYNTHETIC_TEST_CASES:
        data = dict(raw)
        data["collection_id"] = target_col
        cases.append(EvaluationTestCase.model_validate(data))
    return cases


async def seed_evaluation_dataset(
    db: AsyncSession,
    vector_store: VectorStoreService,
    embedding_service: EmbeddingService,
    collection_id: str = EVAL_COLLECTION_ID,
) -> str:
    """
    Seed the synthetic evaluation dataset into the target SQLite database and
    ChromaDB collection using the provided services.

    Args:
        db: Active async database session.
        vector_store: VectorStoreService pointing to temporary/isolated storage.
        embedding_service: EmbeddingService instance.
        collection_id: Target collection UUID string.

    Returns:
        The collection_id string where dataset was seeded.
    """
    # 1. Insert Collection in SQLite
    col_str = str(collection_id)
    existing_col = await db.get(Collection, col_str)
    if not existing_col:
        col = Collection(
            id=col_str,
            name="Retrieval Evaluation Collection",
            description="Synthetic university policy documents for retrieval evaluation.",
        )
        db.add(col)
        await db.commit()

    # 2. Insert Documents in SQLite and Chunks in ChromaDB
    all_chunks_to_embed: List[DocumentChunk] = []

    for doc_data in SYNTHETIC_DOCUMENTS:
        doc_id = doc_data["id"]
        filename = doc_data["filename"]
        file_type = doc_data["file_type"]

        existing_doc = await db.get(Document, doc_id)
        if not existing_doc:
            doc = Document(
                id=doc_id,
                collection_id=col_str,
                filename=filename,
                stored_filename=f"stored_{doc_id}_{filename}",
                file_type=file_type,
                sha256=f"sha256_{doc_id}_eval_data",
                size_bytes=4096,
                status=DocumentStatus.INDEXED.value,
                version=1,
                page_count=len(doc_data["chunks"]),
                chunk_count=len(doc_data["chunks"]),
            )
            db.add(doc)

        for chunk_raw in doc_data["chunks"]:
            cid = chunk_raw["chunk_id"]
            cidx = chunk_raw["chunk_index"]
            text = chunk_raw["text"]
            pnum = chunk_raw.get("page_number")
            prange = chunk_raw.get("page_range")

            meta = ChunkMetadata(
                document_id=doc_id,
                collection_id=col_str,
                filename=filename,
                file_type=file_type,
                chunk_index=cidx,
                char_count=len(text),
                page_number=pnum,
                page_range=prange,
                start_char=cidx * 200,
                end_char=cidx * 200 + len(text),
                version=1,
            )

            dchunk = DocumentChunk(
                chunk_id=cid,
                document_id=doc_id,
                collection_id=col_str,
                filename=filename,
                file_type=file_type,
                chunk_index=cidx,
                chunk_text=text,
                char_count=len(text),
                page_number=pnum,
                page_range=prange,
                metadata=meta,
                version=1,
            )
            all_chunks_to_embed.append(dchunk)

    await db.commit()

    # 3. Generate embeddings and store in ChromaDB
    embedded_chunks: List[EmbeddedChunk] = []
    texts = [c.chunk_text for c in all_chunks_to_embed]
    vectors = embedding_service.embed_batch(texts)

    for chunk, vec in zip(all_chunks_to_embed, vectors):
        emb_chunk = EmbeddedChunk.from_document_chunk(
            chunk=chunk,
            embedding=vec,
            embedding_dimension=len(vec),
            model_name="evaluation-embedder",
        )
        embedded_chunks.append(emb_chunk)

    vector_store.store_chunks(embedded_chunks, collection_id=col_str)

    return col_str
