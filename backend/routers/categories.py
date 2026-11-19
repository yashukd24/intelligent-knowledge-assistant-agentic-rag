"""
backend/routers/categories.py
─────────────────────────────────────────────────────────────
Phase 22: College Institutional Categories, Departments, and Roles.
Provides catalog and management endpoints for college document classification.
─────────────────────────────────────────────────────────────
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database import get_db
from backend.models.document import Document

router = APIRouter(prefix="/api", tags=["College Knowledge Management"])

# 20 Official Institutional Categories (Phase 22.1)
DEFAULT_CATEGORIES: List[str] = [
    "General College Information",
    "Admissions",
    "Courses and Programs",
    "Departments",
    "Faculty and Staff",
    "Campus Directory",
    "Placement Policies",
    "Internship Guidelines",
    "Attendance Regulations",
    "Examination Regulations",
    "Academic Calendar",
    "Timetables",
    "Fee Structures",
    "Scholarships",
    "Hostel Information",
    "Library Information",
    "Laboratories and Facilities",
    "Student Services",
    "Faculty HR Policies",
    "Administrative Procedures",
]

DEFAULT_DEPARTMENTS: List[str] = [
    "Computer Science and Engineering (CSE)",
    "Information Technology (IT)",
    "Electronics and Communication (ECE)",
    "Electrical and Electronics (EEE)",
    "Mechanical Engineering (ME)",
    "Civil Engineering (CE)",
    "Artificial Intelligence and Data Science (AI&DS)",
    "Management / MBA",
    "Basic Sciences and Humanities",
    "Admissions Office",
    "Placement Cell",
    "Examination Cell",
    "Campus Administration",
    "Library Services",
]

DEFAULT_ROLES = [
    {
        "role_id": "student",
        "name": "Student",
        "description": "Enrolled students accessing academic rules, placement criteria, internships, and attendance.",
        "allowed_categories": [
            "General College Information",
            "Courses and Programs",
            "Departments",
            "Faculty and Staff",
            "Campus Directory",
            "Placement Policies",
            "Internship Guidelines",
            "Attendance Regulations",
            "Examination Regulations",
            "Academic Calendar",
            "Timetables",
            "Fee Structures",
            "Scholarships",
            "Hostel Information",
            "Library Information",
            "Laboratories and Facilities",
            "Student Services",
        ],
    },
    {
        "role_id": "new_admission",
        "name": "New Admission / Aspirant",
        "description": "Prospective students exploring admission requirements, eligibility, fee structures, courses, and hostels.",
        "allowed_categories": [
            "General College Information",
            "Admissions",
            "Courses and Programs",
            "Campus Directory",
            "Fee Structures",
            "Scholarships",
            "Hostel Information",
            "Library Information",
            "Laboratories and Facilities",
            "Student Services",
        ],
    },
    {
        "role_id": "faculty",
        "name": "Faculty / Staff",
        "description": "Teaching and non-teaching staff accessing leave procedures, academic regulations, examination duties, and HR policies.",
        "allowed_categories": [
            "General College Information",
            "Departments",
            "Faculty and Staff",
            "Campus Directory",
            "Academic Calendar",
            "Examination Regulations",
            "Laboratories and Facilities",
            "Faculty HR Policies",
            "Administrative Procedures",
        ],
    },
    {
        "role_id": "administrator",
        "name": "Administrator",
        "description": "Authorized institutional administrators managing policies, versions, audit records, and governance.",
        "allowed_categories": DEFAULT_CATEGORIES,
    },
]

# In-memory custom categories store (persists for runtime lifecycle)
_custom_categories: List[str] = []


class CategoryCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100, description="Category name")


class CategoryItem(BaseModel):
    name: str
    is_custom: bool = False
    document_count: Optional[int] = None


class CategoryListResponse(BaseModel):
    categories: List[CategoryItem]
    total: int


class RoleItem(BaseModel):
    role_id: str
    name: str
    description: str
    allowed_categories: List[str]


@router.get("/categories", response_model=CategoryListResponse, summary="List college knowledge categories")
async def list_categories(
    collection_id: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
) -> CategoryListResponse:
    """Return all 20 default categories plus any administrator-added categories."""
    all_names = list(DEFAULT_CATEGORIES) + _custom_categories
    all_names = sorted(list(set(all_names)))

    counts: dict[str, int] = {}
    if collection_id:
        stmt = (
            select(Document.category, func.count(Document.id))
            .where(Document.collection_id == collection_id)
            .group_by(Document.category)
        )
        res = await db.execute(stmt)
        for cat, cnt in res.fetchall():
            if cat:
                counts[cat] = cnt

    items = [
        CategoryItem(
            name=name,
            is_custom=name in _custom_categories,
            document_count=counts.get(name, 0) if collection_id else None,
        )
        for name in all_names
    ]
    return CategoryListResponse(categories=items, total=len(items))


@router.post(
    "/categories",
    response_model=CategoryItem,
    status_code=status.HTTP_201_CREATED,
    summary="Add custom college knowledge category",
)
async def create_category(payload: CategoryCreate) -> CategoryItem:
    cat_name = payload.name.strip()
    if cat_name not in DEFAULT_CATEGORIES and cat_name not in _custom_categories:
        _custom_categories.append(cat_name)
    return CategoryItem(name=cat_name, is_custom=True)


@router.get("/departments", response_model=List[str], summary="List college institutional departments")
async def list_departments() -> List[str]:
    """Return standard college engineering and administrative departments."""
    return DEFAULT_DEPARTMENTS


@router.get("/roles", response_model=List[RoleItem], summary="List supported college user roles")
async def list_roles() -> List[RoleItem]:
    """Return configured user roles (Student, New Admission, Faculty, Administrator)."""
    return [RoleItem(**r) for r in DEFAULT_ROLES]
