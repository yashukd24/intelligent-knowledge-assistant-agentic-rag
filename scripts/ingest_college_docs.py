"""
scripts/ingest_college_docs.py
Upload and process all sample college documents into collection c27fac1e-c28d-48cd-96ee-78a918fb8e40.
"""

import asyncio
from pathlib import Path
from backend.database import AsyncSessionLocal
from backend.services.file_storage import FileStorageService
from backend.services.document_processor import DocumentProcessor
from backend.services.embedding_service import EmbeddingService
from backend.services.vector_store import VectorStoreService
from backend.models.collection import Collection
from backend.models.document import Document
from sqlalchemy import select

DOCS_DIR = Path("data/college_sample_docs")
COLLECTION_ID = "c27fac1e-c28d-48cd-96ee-78a918fb8e40"

DOC_METADATA = {
    "01_campus_directory.txt": {
        "category": "campus_directory",
        "department": "campus_administration",
        "academic_year": "2025-2026",
        "section_heading": "Campus Directory & Locations",
    },
    "02_placement_policy.txt": {
        "category": "placement_policy",
        "department": "placement_cell",
        "academic_year": "2025-2026",
        "section_heading": "Placement Eligibility Criteria v1.0",
    },
    "03_internship_guidelines.txt": {
        "category": "internship_guidelines",
        "department": "placement_cell",
        "academic_year": "2025-2026",
        "section_heading": "Internship Guidelines & Approval",
    },
    "04_faculty_hr_policy.txt": {
        "category": "faculty_hr_policy",
        "department": "campus_administration",
        "academic_year": "2025-2026",
        "section_heading": "Faculty HR Leave Policy",
    },
    "05_admission_brochure.txt": {
        "category": "admissions",
        "department": "admissions_office",
        "academic_year": "2025-2026",
        "section_heading": "B.Tech Admissions & Required Documents",
    },
    "07_conflicting_policy_v2.txt": {
        "category": "placement_policy",
        "department": "placement_cell",
        "academic_year": "2026-2027",
        "section_heading": "Placement Eligibility Criteria v2.0 (Draft)",
    },
}

import io

class FakeUploadFile:
    def __init__(self, filename: str, content: bytes):
        self.filename = filename
        self._io = io.BytesIO(content)
    async def read(self, n: int = -1) -> bytes:
        return self._io.read(n)

async def ingest_all():
    storage = FileStorageService()
    embedding_service = EmbeddingService()
    vector_store = VectorStoreService()
    processor = DocumentProcessor(
        storage_service=storage,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    async with AsyncSessionLocal() as session:
        # Check collection exists
        res = await session.execute(select(Collection).where(Collection.id == COLLECTION_ID))
        col = res.scalar_one_or_none()
        if not col:
            print(f"Collection {COLLECTION_ID} not found!")
            return

        print(f"Target Collection: {col.name} ({col.id})")

        for filename, meta in DOC_METADATA.items():
            file_path = DOCS_DIR / filename
            if not file_path.exists():
                print(f"File not found: {file_path}")
                continue

            content = file_path.read_bytes()
            upload = FakeUploadFile(filename, content)

            # Check if document with same filename already exists in DB
            q = select(Document).where(
                Document.collection_id == COLLECTION_ID,
                Document.filename == filename
            )
            existing = (await session.execute(q)).scalar_one_or_none()

            stored_info = await storage.save_uploaded_file(COLLECTION_ID, upload)

            if existing:
                print(f"Updating existing doc: {filename} ({existing.id})")
                existing.stored_filename = stored_info.stored_filename
                existing.file_type = stored_info.file_type
                existing.sha256 = stored_info.sha256
                existing.size_bytes = stored_info.size_bytes
                existing.status = "uploaded"
                existing.category = meta["category"]
                existing.department = meta["department"]
                existing.academic_year = meta["academic_year"]
                existing.section_heading = meta["section_heading"]
                await session.commit()
                doc_id = existing.id
            else:
                new_doc = Document(
                    collection_id=COLLECTION_ID,
                    filename=filename,
                    stored_filename=stored_info.stored_filename,
                    file_type=stored_info.file_type,
                    sha256=stored_info.sha256,
                    size_bytes=stored_info.size_bytes,
                    status="uploaded",
                    category=meta["category"],
                    department=meta["department"],
                    academic_year=meta["academic_year"],
                    section_heading=meta["section_heading"],
                )
                session.add(new_doc)
                await session.commit()
                await session.refresh(new_doc)
                doc_id = new_doc.id
                print(f"Created new doc: {filename} ({doc_id})")

            # Process & embed
            print(f"Processing and embedding {filename} ({doc_id})...")
            processed = await processor.process_document(doc_id, session, embed=True)
            print(f"  Indexed: {processed.chunk_count} chunks created and stored in ChromaDB.")

        # Update collection document_count
        count_q = select(Document).where(Document.collection_id == COLLECTION_ID)
        docs = (await session.execute(count_q)).scalars().all()
        col.document_count = len(docs)
        await session.commit()
        print(f"Done! Collection {COLLECTION_ID} now has {len(docs)} documents.")

if __name__ == "__main__":
    asyncio.run(ingest_all())
