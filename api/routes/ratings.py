"""
API routes for managing ratings on notes.
"""

import json
from datetime import datetime, UTC
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from api.database import get_db, Note, Rating, UserFactor
from api.models import (
    CreateRatingRequest, RatingResponse, HelpfulnessLevel, UserFactorResponse
)


router = APIRouter(prefix="/ratings", tags=["Ratings"])


# =============================================================================
# Create/Update Rating
# =============================================================================


@router.post("/", response_model=RatingResponse, status_code=201)
def create_or_update_rating(
    request: CreateRatingRequest,
    db: Session = Depends(get_db)
) -> RatingResponse:
    """
    Rate a fact-check note.
    
    Users rate notes as helpful, somewhat helpful, or not helpful,
    along with tags explaining their reasoning. These ratings feed
    into the bridging algorithm to identify genuinely helpful notes.
    
    If the user has already rated this note, their rating is updated.
    """
    # Verify the note exists
    note = db.query(Note).filter(Note.id == request.note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    # Check if user is trying to rate their own note
    if note.author_id == request.rater_id:
        raise HTTPException(
            status_code=400,
            detail="Users cannot rate their own notes"
        )
    
    # Check for existing rating
    existing_rating = db.query(Rating).filter(
        Rating.note_id == request.note_id,
        Rating.rater_id == request.rater_id
    ).first()
    
    if existing_rating:
        # Update existing rating
        old_helpfulness = existing_rating.helpfulness
        existing_rating.helpfulness = request.helpfulness.value
        existing_rating.helpful_tags = json.dumps([t.value for t in request.helpful_tags])
        existing_rating.not_helpful_tags = json.dumps([t.value for t in request.not_helpful_tags])
        existing_rating.updated_at = datetime.now(UTC)
        
        # Update note vote counts
        _update_vote_counts(note, old_helpfulness, request.helpfulness.value)
        
        db.commit()
        db.refresh(existing_rating)
        
        return _rating_to_response(existing_rating)
    else:
        # Create new rating
        rating = Rating(
            note_id=request.note_id,
            rater_id=request.rater_id,
            helpfulness=request.helpfulness.value,
            helpful_tags=json.dumps([t.value for t in request.helpful_tags]),
            not_helpful_tags=json.dumps([t.value for t in request.not_helpful_tags]),
        )
        
        db.add(rating)
        
        # Update note vote counts
        _increment_vote_count(note, request.helpfulness.value)
        
        # Update user's rating count
        _update_user_rating_stats(db, request.rater_id, request.helpfulness)
        
        db.commit()
        db.refresh(rating)
        
        return _rating_to_response(rating)


# =============================================================================
# Get Ratings
# =============================================================================


@router.get("/note/{note_id}", response_model=List[RatingResponse])
def get_ratings_for_note(
    note_id: int,
    limit: int = Query(default=100, le=500),
    offset: int = 0,
    db: Session = Depends(get_db)
) -> List[RatingResponse]:
    """Get all ratings for a specific note."""
    note = db.query(Note).filter(Note.id == note_id).first()
    if not note:
        raise HTTPException(status_code=404, detail="Note not found")
    
    ratings = db.query(Rating).filter(
        Rating.note_id == note_id
    ).offset(offset).limit(limit).all()
    
    return [_rating_to_response(r) for r in ratings]


@router.get("/user/{user_id}", response_model=List[RatingResponse])
def get_ratings_by_user(
    user_id: int,
    limit: int = Query(default=100, le=500),
    offset: int = 0,
    db: Session = Depends(get_db)
) -> List[RatingResponse]:
    """Get all ratings submitted by a specific user."""
    ratings = db.query(Rating).filter(
        Rating.rater_id == user_id
    ).order_by(Rating.created_at.desc()).offset(offset).limit(limit).all()
    
    return [_rating_to_response(r) for r in ratings]


@router.get("/user/{user_id}/note/{note_id}", response_model=RatingResponse)
def get_user_rating_for_note(
    user_id: int,
    note_id: int,
    db: Session = Depends(get_db)
) -> RatingResponse:
    """Get a specific user's rating for a specific note."""
    rating = db.query(Rating).filter(
        Rating.rater_id == user_id,
        Rating.note_id == note_id
    ).first()
    
    if not rating:
        raise HTTPException(status_code=404, detail="Rating not found")
    
    return _rating_to_response(rating)


# =============================================================================
# User Factors
# =============================================================================


@router.get("/user/{user_id}/factor", response_model=UserFactorResponse)
def get_user_factor(
    user_id: int,
    db: Session = Depends(get_db)
) -> UserFactorResponse:
    """
    Get a user's computed factor from the scoring algorithm.
    
    The factor represents where the user falls on the latent viewpoint
    spectrum, inferred from their rating patterns. Users don't explicitly
    set this - it's computed by the matrix factorization algorithm.
    """
    user_factor = db.query(UserFactor).filter(
        UserFactor.user_id == user_id
    ).first()
    
    if not user_factor:
        # Return default response for users without computed factors
        return UserFactorResponse(
            user_id=user_id,
            factor=None,
            factor_confidence=0.0,
            total_ratings=0,
            last_computed_at=None
        )
    
    return UserFactorResponse(
        user_id=user_factor.user_id,
        factor=user_factor.factor,
        factor_confidence=user_factor.factor_confidence,
        total_ratings=user_factor.total_ratings,
        last_computed_at=user_factor.last_computed_at
    )


# =============================================================================
# Delete Rating
# =============================================================================


@router.delete("/{rating_id}", status_code=204)
def delete_rating(
    rating_id: int,
    rater_id: int,  # Required to verify ownership
    db: Session = Depends(get_db)
):
    """
    Delete a rating.
    
    Only the rater can delete their own rating.
    """
    rating = db.query(Rating).filter(Rating.id == rating_id).first()
    if not rating:
        raise HTTPException(status_code=404, detail="Rating not found")
    
    if rating.rater_id != rater_id:
        raise HTTPException(
            status_code=403,
            detail="Only the rater can delete this rating"
        )
    
    # Update note vote counts
    note = db.query(Note).filter(Note.id == rating.note_id).first()
    if note:
        _decrement_vote_count(note, rating.helpfulness)
    
    db.delete(rating)
    db.commit()


# =============================================================================
# Helper Functions
# =============================================================================


def _rating_to_response(rating: Rating) -> RatingResponse:
    """Convert database Rating to response model."""
    return RatingResponse(
        id=rating.id,
        note_id=rating.note_id,
        rater_id=rating.rater_id,
        helpfulness=rating.helpfulness,
        helpful_tags=json.loads(rating.helpful_tags) if rating.helpful_tags else [],
        not_helpful_tags=json.loads(rating.not_helpful_tags) if rating.not_helpful_tags else [],
        rater_factor=rating.rater_factor,
        created_at=rating.created_at,
        updated_at=rating.updated_at
    )


def _increment_vote_count(note: Note, helpfulness: str):
    """Increment the appropriate vote count on a note."""
    if helpfulness == HelpfulnessLevel.HELPFUL.value:
        note.helpful_count += 1
    elif helpfulness == HelpfulnessLevel.SOMEWHAT_HELPFUL.value:
        note.somewhat_helpful_count += 1
    elif helpfulness == HelpfulnessLevel.NOT_HELPFUL.value:
        note.not_helpful_count += 1


def _decrement_vote_count(note: Note, helpfulness: str):
    """Decrement the appropriate vote count on a note."""
    if helpfulness == HelpfulnessLevel.HELPFUL.value:
        note.helpful_count = max(0, note.helpful_count - 1)
    elif helpfulness == HelpfulnessLevel.SOMEWHAT_HELPFUL.value:
        note.somewhat_helpful_count = max(0, note.somewhat_helpful_count - 1)
    elif helpfulness == HelpfulnessLevel.NOT_HELPFUL.value:
        note.not_helpful_count = max(0, note.not_helpful_count - 1)


def _update_vote_counts(note: Note, old_helpfulness: str, new_helpfulness: str):
    """Update vote counts when a rating changes."""
    if old_helpfulness != new_helpfulness:
        _decrement_vote_count(note, old_helpfulness)
        _increment_vote_count(note, new_helpfulness)


def _update_user_rating_stats(db: Session, user_id: int, helpfulness: HelpfulnessLevel):
    """Update user's rating statistics."""
    user_factor = db.query(UserFactor).filter(UserFactor.user_id == user_id).first()
    
    if not user_factor:
        user_factor = UserFactor(user_id=user_id)
        db.add(user_factor)
    
    user_factor.total_ratings += 1
    if helpfulness == HelpfulnessLevel.HELPFUL:
        user_factor.helpful_ratings_given += 1
    elif helpfulness == HelpfulnessLevel.NOT_HELPFUL:
        user_factor.not_helpful_ratings_given += 1
