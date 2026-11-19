"""backend/models/base.py — SQLAlchemy declarative Base."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy ORM models.
    All models in backend/models/ must inherit from this class.
    Table definitions are added in Phases 3, 7, and 17.
    """
    pass
