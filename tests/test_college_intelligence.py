"""
tests/test_college_intelligence.py
─────────────────────────────────────────────────────────────
Unit and API integration tests for Phase 22 & Phase 23:
  - College Knowledge Categories & Departments Catalog.
  - Role definitions (Student, Faculty, New Admission, Administrator).
  - Document metadata assignment (Category, Department, Academic Year).
  - Category and Department document filtering.
  - Structured College Intent Classification & Constraint Extraction.
─────────────────────────────────────────────────────────────
"""

import io
from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app
from backend.schemas.query_plan import QueryIntent
from backend.services.query_understanding import IntentClassifier, QueryUnderstandingService


@pytest.mark.asyncio
async def test_categories_catalog_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        res = await client.get("/api/categories")
        assert res.status_code == 200
        data = res.json()
        assert data["total"] >= 20
        cat_names = [c["name"] for c in data["categories"]]
        assert "Placement Policies" in cat_names
        assert "Internship Guidelines" in cat_names
        assert "Campus Directory" in cat_names
        assert "Faculty HR Policies" in cat_names
        assert "Admissions" in cat_names


@pytest.mark.asyncio
async def test_create_custom_category():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        custom_name = f"Custom Lab Guidelines {uuid4().hex[:6]}"
        res = await client.post("/api/categories", json={"name": custom_name})
        assert res.status_code == 201
        data = res.json()
        assert data["name"] == custom_name
        assert data["is_custom"] is True


@pytest.mark.asyncio
async def test_departments_and_roles_endpoints():
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        dept_res = await client.get("/api/departments")
        assert dept_res.status_code == 200
        depts = dept_res.json()
        assert any("Computer Science and Engineering (CSE)" in d for d in depts)
        assert any("Placement Cell" in d for d in depts)

        roles_res = await client.get("/api/roles")
        assert roles_res.status_code == 200
        roles = roles_res.json()
        role_ids = [r["role_id"] for r in roles]
        assert "student" in role_ids
        assert "faculty" in role_ids
        assert "new_admission" in role_ids
        assert "administrator" in role_ids


@pytest.mark.asyncio
async def test_document_metadata_assignment_and_update(client: AsyncClient):
    # 0. Create collection
    col_res = await client.post("/api/collections", json={"name": f"College_Col_{uuid4().hex[:8]}"})
    assert col_res.status_code == 201
    sample_collection_id = col_res.json()["id"]

    # 1. Upload document with initial metadata
    files = {"file": ("placement_rules.txt", io.BytesIO(b"Placement eligibility criteria."), "text/plain")}
    params = {
        "category": "Placement Policies",
        "department": "Placement Cell",
        "academic_year": "2025-2026",
        "section_heading": "Eligibility Criteria",
    }
    create_res = await client.post(
        f"/api/collections/{sample_collection_id}/documents",
        files=files,
        params=params,
    )
    assert create_res.status_code == 201
    doc_data = create_res.json()
    assert doc_data["category"] == "Placement Policies"
    assert doc_data["department"] == "Placement Cell"
    assert doc_data["academic_year"] == "2025-2026"
    doc_id = doc_data["id"]

    # 2. Update metadata via PATCH
    patch_res = await client.patch(
        f"/api/documents/{doc_id}/metadata",
        json={"effective_date": "2025-08-01", "section_heading": "Updated Eligibility Criteria"},
    )
    assert patch_res.status_code == 200
    updated = patch_res.json()
    assert updated["effective_date"] == "2025-08-01"
    assert updated["section_heading"] == "Updated Eligibility Criteria"
    assert updated["category"] == "Placement Policies"

    # 3. Test filtering by category
    list_res = await client.get(
        f"/api/collections/{sample_collection_id}/documents",
        params={"category": "Placement Policies"},
    )
    assert list_res.status_code == 200
    filtered = list_res.json()
    assert filtered["total"] >= 1
    assert any(d["id"] == doc_id for d in filtered["documents"])


def test_college_intent_classification():
    # Test queries from Section 1 and Section 9
    assert IntentClassifier.classify_college_intent("Where is the CSE HOD cabin?") == QueryIntent.CAMPUS_LOCATION
    assert IntentClassifier.classify_college_intent("What are the placement eligibility conditions?") == QueryIntent.PLACEMENT_POLICY
    assert IntentClassifier.classify_college_intent("What are the internship rules?") == QueryIntent.INTERNSHIP_POLICY
    assert IntentClassifier.classify_college_intent("What is the faculty leave procedure?") == QueryIntent.FACULTY_POLICY
    assert IntentClassifier.classify_college_intent("What are the admission eligibility requirements?") == QueryIntent.ADMISSION_QUERY
    assert IntentClassifier.classify_college_intent("What is the minimum attendance requirement?") == QueryIntent.ATTENDANCE_POLICY
    assert IntentClassifier.classify_college_intent("What courses are available in engineering?") == QueryIntent.COURSE_QUERY
    assert IntentClassifier.classify_college_intent("How do I apply for a scholarship?") == QueryIntent.SCHOLARSHIP_QUERY


def test_target_category_and_department_extraction():
    svc = QueryUnderstandingService()
    analysis = svc.analyze("Where is the CSE HOD cabin?")
    assert analysis.college_intent == QueryIntent.CAMPUS_LOCATION
    assert "Campus Directory" in analysis.target_categories
    assert "Computer Science and Engineering (CSE)" in analysis.target_departments

    analysis_placement = svc.analyze("What are the placement eligibility conditions for CSE students?")
    assert analysis_placement.college_intent == QueryIntent.PLACEMENT_POLICY
    assert "Placement Policies" in analysis_placement.target_categories
    assert "Computer Science and Engineering (CSE)" in analysis_placement.target_departments
