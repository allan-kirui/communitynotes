"""
Pydantic models for Community Notes API requests and responses.

These models are adapted for promise verification context.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# Enums
# =============================================================================


class NoteClassification(str, Enum):
    """Classification of what the note asserts about the promise."""
    ACCURATE = "accurate"                    # Promise claim is accurate
    MISLEADING = "misleading"                # Promise framing is misleading
    MISSING_CONTEXT = "missing_context"      # Important context omitted
    FACTUALLY_INCORRECT = "factually_incorrect"  # Contains false claims
    OUTDATED = "outdated"                    # Information is no longer current
    
    # Promise-specific classifications
    PROMISE_KEPT = "promise_kept"            # Evidence promise was fulfilled
    PROMISE_BROKEN = "promise_broken"        # Evidence promise was broken
    PROMISE_IN_PROGRESS = "promise_in_progress"  # Partial progress shown
    PROMISE_STALLED = "promise_stalled"      # No progress for extended period


class NoteStatus(str, Enum):
    """Status of a note based on community ratings."""
    NEEDS_MORE_RATINGS = "needs_more_ratings"
    CURRENTLY_RATED_HELPFUL = "currently_rated_helpful"
    CURRENTLY_RATED_NOT_HELPFUL = "currently_rated_not_helpful"


class HelpfulnessLevel(str, Enum):
    """Rating levels for note helpfulness."""
    HELPFUL = "helpful"
    SOMEWHAT_HELPFUL = "somewhat_helpful"
    NOT_HELPFUL = "not_helpful"


class HelpfulTag(str, Enum):
    """Tags explaining why a note IS helpful."""
    CLEAR_AND_INFORMATIVE = "clear_and_informative"
    GOOD_SOURCES = "good_sources"
    ADDRESSES_CLAIM_DIRECTLY = "addresses_claim_directly"
    IMPORTANT_CONTEXT = "important_context"
    UNBIASED_LANGUAGE = "unbiased_language"
    PROVIDES_EVIDENCE = "provides_evidence"
    CITES_OFFICIAL_RECORDS = "cites_official_records"  # Promise-specific


class NotHelpfulTag(str, Enum):
    """Tags explaining why a note is NOT helpful."""
    INCORRECT_INFORMATION = "incorrect_information"
    SOURCES_MISSING = "sources_missing"
    SOURCES_UNRELIABLE = "sources_unreliable"
    MISSING_KEY_POINTS = "missing_key_points"
    OUTDATED_INFORMATION = "outdated_information"
    OPINION_NOT_FACT = "opinion_not_fact"
    OFF_TOPIC = "off_topic"
    BIASED_LANGUAGE = "biased_language"
    SPAM_OR_HARASSMENT = "spam_or_harassment"
    ARGUMENTATIVE = "argumentative"


# =============================================================================
# Request Models
# =============================================================================


class CreateNoteRequest(BaseModel):
    """Request to create a new fact-check note on a promise."""
    
    promise_id: int = Field(..., description="ID of the promise being fact-checked")
    author_id: int = Field(..., description="User ID of the note author")
    
    summary: str = Field(
        ...,
        min_length=10,
        max_length=280,
        description="Brief summary of what this note adds (tweet-length)"
    )
    content: str = Field(
        ...,
        min_length=20,
        max_length=4000,
        description="Full explanation with context"
    )
    
    sources: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="URLs supporting this note"
    )
    
    classification: NoteClassification = Field(
        ...,
        description="What does this note assert about the promise?"
    )
    
    @field_validator('sources')
    @classmethod
    def validate_sources(cls, v):
        """Ensure sources are valid URLs."""
        for url in v:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f'Invalid URL: {url}')
        return v


class CreateRatingRequest(BaseModel):
    """Request to rate a fact-check note."""
    
    note_id: int = Field(..., description="ID of the note being rated")
    rater_id: int = Field(..., description="User ID of the rater")
    
    helpfulness: HelpfulnessLevel = Field(
        ...,
        description="How helpful is this note?"
    )
    
    helpful_tags: List[HelpfulTag] = Field(
        default_factory=list,
        description="Tags explaining why note IS helpful (if helpful/somewhat)"
    )
    
    not_helpful_tags: List[NotHelpfulTag] = Field(
        default_factory=list,
        description="Tags explaining why note is NOT helpful (if not helpful)"
    )
    
    @field_validator('helpful_tags', 'not_helpful_tags')
    @classmethod
    def validate_tags(cls, v):
        """Ensure no more than 5 tags selected."""
        if len(v) > 5:
            raise ValueError('Maximum 5 tags allowed')
        return v


class TriggerScoringRequest(BaseModel):
    """Request to manually trigger scoring (admin only)."""
    
    promise_ids: Optional[List[int]] = Field(
        default=None,
        description="Specific promise IDs to score (None = all)"
    )
    force_full_rescore: bool = Field(
        default=False,
        description="Force complete recalculation ignoring cache"
    )


# =============================================================================
# Response Models
# =============================================================================


class NoteResponse(BaseModel):
    """Response containing a single note."""
    
    id: int
    promise_id: int
    author_id: int
    
    summary: str
    content: str
    sources: List[str]
    classification: NoteClassification
    
    status: NoteStatus
    
    # Scoring results (populated after scoring)
    helpfulness_score: Optional[float] = None
    note_intercept: Optional[float] = None
    note_factor: Optional[float] = None
    
    # Vote counts
    helpful_count: int = 0
    somewhat_helpful_count: int = 0
    not_helpful_count: int = 0
    
    created_at: datetime
    updated_at: datetime
    scored_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class RatingResponse(BaseModel):
    """Response containing a single rating."""
    
    id: int
    note_id: int
    rater_id: int
    helpfulness: HelpfulnessLevel
    helpful_tags: List[str]
    not_helpful_tags: List[str]
    
    # User factor (computed by algorithm)
    rater_factor: Optional[float] = None
    
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ScoredNotesResponse(BaseModel):
    """Response containing scored notes for a promise."""
    
    promise_id: int
    notes: List[NoteResponse]
    
    # Summary stats
    total_notes: int
    helpful_notes_count: int
    needs_more_ratings_count: int
    
    # Scoring metadata
    last_scored_at: Optional[datetime] = None
    algorithm_version: str = "1.0.0"


class UserFactorResponse(BaseModel):
    """Response containing a user's computed factor."""
    
    user_id: int
    factor: Optional[float] = None
    factor_confidence: float = 0.0  # 0-1, based on number of ratings
    total_ratings: int = 0
    last_computed_at: Optional[datetime] = None


class ScoringResultResponse(BaseModel):
    """Response from a scoring run."""
    
    success: bool
    notes_scored: int
    users_updated: int
    duration_seconds: float
    errors: List[str] = Field(default_factory=list)
    scored_at: datetime


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = "healthy"
    version: str
    database_connected: bool
    last_scoring_run: Optional[datetime] = None
    notes_count: int = 0
    ratings_count: int = 0
