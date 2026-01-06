"""
Database models and session management for Community Notes API.

Uses SQLAlchemy with SQLite for the standalone service.
Can be configured for PostgreSQL in production.
"""

from datetime import datetime, UTC
from typing import Optional, List

from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, Boolean,
    ForeignKey, Enum, Index, UniqueConstraint, create_engine, JSON
)
from sqlalchemy.orm import (
    DeclarativeBase, relationship, sessionmaker, Session, Mapped, mapped_column
)

from api.config import get_settings
from api.models import NoteStatus, NoteClassification, HelpfulnessLevel


# =============================================================================
# Database Setup
# =============================================================================


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


def get_engine():
    """Create database engine."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
        echo=settings.debug
    )


engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


# =============================================================================
# Database Models
# =============================================================================


class Note(Base):
    """
    A fact-check note on a promise.
    
    This is the equivalent of a Community Notes "note" but specific
    to political promise verification.
    """
    
    __tablename__ = "notes"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    # Links to accountability backend
    promise_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    author_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    
    # Note content
    summary: Mapped[str] = mapped_column(String(280), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    sources: Mapped[str] = mapped_column(Text, default="[]")  # JSON array of URLs
    
    # Classification
    classification: Mapped[str] = mapped_column(
        String(50),
        default=NoteClassification.MISSING_CONTEXT.value,
        nullable=False
    )
    
    # Status determined by scoring algorithm
    status: Mapped[str] = mapped_column(
        String(50),
        default=NoteStatus.NEEDS_MORE_RATINGS.value,
        index=True,
        nullable=False
    )
    
    # Scoring results
    helpfulness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    note_intercept: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    note_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Vote counts (denormalized for performance)
    helpful_count: Mapped[int] = mapped_column(Integer, default=0)
    somewhat_helpful_count: Mapped[int] = mapped_column(Integer, default=0)
    not_helpful_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )
    scored_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Algorithm version used for last scoring
    algorithm_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Relationships
    ratings: Mapped[List["Rating"]] = relationship("Rating", back_populates="note")
    
    __table_args__ = (
        Index('ix_notes_promise_status', 'promise_id', 'status'),
    )


class Rating(Base):
    """
    A user's rating on a fact-check note.
    
    Includes helpfulness level and explanation tags that are
    crucial for the bridging algorithm.
    """
    
    __tablename__ = "ratings"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    note_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("notes.id", ondelete="CASCADE"), index=True
    )
    rater_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    
    # The rating
    helpfulness: Mapped[str] = mapped_column(String(30), nullable=False)
    
    # Explanation tags (stored as JSON arrays)
    helpful_tags: Mapped[str] = mapped_column(Text, default="[]")
    not_helpful_tags: Mapped[str] = mapped_column(Text, default="[]")
    
    # Computed user factor (updated during scoring)
    rater_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )
    
    # Relationships
    note: Mapped["Note"] = relationship("Note", back_populates="ratings")
    
    __table_args__ = (
        # Each user can only rate a note once
        UniqueConstraint('rater_id', 'note_id', name='uq_rater_note'),
    )


class UserFactor(Base):
    """
    Cached user factors computed by the scoring algorithm.
    
    The factor represents where a user falls on the latent
    viewpoint spectrum, inferred from their rating patterns.
    """
    
    __tablename__ = "user_factors"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, unique=True, index=True, nullable=False)
    
    # Computed factor from matrix factorization
    factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Confidence based on number of ratings (0-1)
    factor_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Rating statistics
    total_ratings: Mapped[int] = mapped_column(Integer, default=0)
    helpful_ratings_given: Mapped[int] = mapped_column(Integer, default=0)
    not_helpful_ratings_given: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC), nullable=False
    )
    last_computed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Algorithm version
    algorithm_version: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)


class ScoringRun(Base):
    """
    Log of scoring algorithm runs.
    
    Tracks when scoring was performed and results for monitoring.
    """
    
    __tablename__ = "scoring_runs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Results
    notes_scored: Mapped[int] = mapped_column(Integer, default=0)
    users_updated: Mapped[int] = mapped_column(Integer, default=0)
    status_changes: Mapped[int] = mapped_column(Integer, default=0)
    
    # Performance
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Errors
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Configuration used
    algorithm_version: Mapped[str] = mapped_column(String(20), nullable=False)
    config_snapshot: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON
