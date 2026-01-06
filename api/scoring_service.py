"""
Scoring service wrapper around the Community Notes matrix factorization algorithm.

This module adapts the original CN scoring algorithm for promise verification,
converting our database format to the expected input format and processing
the results back into our database.
"""

import json
import logging
import time
from datetime import datetime, UTC
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from api.config import get_settings
from api.database import Note, Rating, UserFactor, ScoringRun
from api.models import NoteStatus, HelpfulnessLevel

# Import the actual CN scoring modules
import sys
import os

# Add the scoring src directory to path
SCORING_SRC_PATH = os.path.join(os.path.dirname(__file__), '..', 'scoring', 'src')
if SCORING_SRC_PATH not in sys.path:
    sys.path.insert(0, SCORING_SRC_PATH)

try:
    from scoring.matrix_factorization.matrix_factorization import MatrixFactorization
    from scoring import constants as c
    CN_ALGORITHM_AVAILABLE = True
except ImportError as e:
    CN_ALGORITHM_AVAILABLE = False
    logging.warning(f"CN scoring algorithm not available: {e}. Using simplified scoring.")


logger = logging.getLogger(__name__)


class ScoringService:
    """
    Service for running the Community Notes scoring algorithm.
    
    Adapts the CN matrix factorization algorithm for promise verification:
    - Converts database notes/ratings to CN format
    - Runs the bridging algorithm
    - Updates note statuses and user factors
    """
    
    ALGORITHM_VERSION = "1.0.0"
    
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()
    
    def run_scoring(
        self,
        promise_ids: Optional[List[int]] = None,
        force_full_rescore: bool = False
    ) -> Dict:
        """
        Run the scoring algorithm on notes.
        
        Args:
            promise_ids: Specific promise IDs to score (None = all)
            force_full_rescore: If True, ignore previous scores
            
        Returns:
            Dictionary with scoring results
        """
        start_time = time.time()
        
        # Create scoring run log
        scoring_run = ScoringRun(
            started_at=datetime.now(UTC),
            algorithm_version=self.ALGORITHM_VERSION,
            config_snapshot=json.dumps({
                "min_ratings_for_status": self.settings.min_ratings_for_status,
                "helpful_intercept_threshold": self.settings.helpful_intercept_threshold,
                "not_helpful_intercept_threshold": self.settings.not_helpful_intercept_threshold,
            })
        )
        self.db.add(scoring_run)
        self.db.commit()
        
        try:
            # Get notes to score
            notes_query = self.db.query(Note)
            if promise_ids:
                notes_query = notes_query.filter(Note.promise_id.in_(promise_ids))
            notes = notes_query.all()
            
            if not notes:
                logger.info("No notes to score")
                return self._finalize_scoring_run(scoring_run, 0, 0, start_time, [])
            
            # Get all ratings for these notes
            note_ids = [n.id for n in notes]
            ratings = self.db.query(Rating).filter(Rating.note_id.in_(note_ids)).all()
            
            if not ratings:
                logger.info("No ratings found, cannot score notes")
                return self._finalize_scoring_run(scoring_run, 0, 0, start_time, [])
            
            # Convert to DataFrames
            notes_df = self._notes_to_dataframe(notes)
            ratings_df = self._ratings_to_dataframe(ratings)
            
            # Run the algorithm
            if CN_ALGORITHM_AVAILABLE and len(ratings) >= self.settings.min_ratings_for_status:
                scored_notes_df, user_factors_df = self._run_matrix_factorization(
                    notes_df, ratings_df
                )
            else:
                # Simplified scoring for small datasets or when CN not available
                scored_notes_df, user_factors_df = self._run_simplified_scoring(
                    notes_df, ratings_df
                )
            
            # Update database with results
            notes_updated = self._update_note_scores(scored_notes_df)
            users_updated = self._update_user_factors(user_factors_df)
            
            return self._finalize_scoring_run(
                scoring_run, notes_updated, users_updated, start_time, []
            )
            
        except Exception as e:
            logger.exception("Error during scoring")
            scoring_run.success = False
            scoring_run.error_message = str(e)
            scoring_run.completed_at = datetime.now(UTC)
            self.db.commit()
            raise
    
    def _notes_to_dataframe(self, notes: List[Note]) -> pd.DataFrame:
        """Convert notes to DataFrame format expected by CN algorithm."""
        return pd.DataFrame([
            {
                'noteId': n.id,
                'promiseId': n.promise_id,
                'authorId': n.author_id,
                'classification': n.classification,
                'createdAtMillis': int(n.created_at.timestamp() * 1000),
            }
            for n in notes
        ])
    
    def _ratings_to_dataframe(self, ratings: List[Rating]) -> pd.DataFrame:
        """Convert ratings to DataFrame format expected by CN algorithm."""
        return pd.DataFrame([
            {
                'noteId': r.note_id,
                'raterParticipantId': r.rater_id,
                'helpfulNum': self._helpfulness_to_num(r.helpfulness),
                'helpfulnessLevel': r.helpfulness,
                'createdAtMillis': int(r.created_at.timestamp() * 1000),
            }
            for r in ratings
        ])
    
    def _helpfulness_to_num(self, helpfulness: str) -> float:
        """Convert helpfulness level to numeric value for algorithm."""
        mapping = {
            HelpfulnessLevel.HELPFUL.value: 1.0,
            HelpfulnessLevel.SOMEWHAT_HELPFUL.value: 0.5,
            HelpfulnessLevel.NOT_HELPFUL.value: 0.0,
        }
        return mapping.get(helpfulness, 0.5)
    
    def _run_matrix_factorization(
        self,
        notes_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the actual Community Notes matrix factorization algorithm.
        
        This is the core bridging algorithm that identifies notes helpful
        to people across different viewpoints.
        """
        logger.info(f"Running matrix factorization on {len(notes_df)} notes, {len(ratings_df)} ratings")
        
        # Initialize matrix factorization
        mf = MatrixFactorization(
            numFactors=self.settings.mf_num_factors,
            initLearningRate=self.settings.mf_learning_rate,
            userFactorLambda=self.settings.mf_regularization,
            noteFactorLambda=self.settings.mf_regularization,
            userInterceptLambda=self.settings.mf_regularization * 5,
            noteInterceptLambda=self.settings.mf_regularization * 5,
            globalInterceptLambda=self.settings.mf_regularization * 5,
            useGlobalIntercept=True,
            log=True,
        )
        
        # Prepare rating features
        ratings_for_mf = ratings_df[['noteId', 'raterParticipantId', 'helpfulNum']].copy()
        ratings_for_mf.columns = [c.noteIdKey, c.raterParticipantIdKey, c.helpfulNumKey]
        
        # Run fitting
        mf.fit(ratings_for_mf)
        
        # Extract note scores
        note_params = mf.get_note_params()
        user_params = mf.get_user_params()
        
        # Build scored notes DataFrame
        scored_notes = notes_df.copy()
        scored_notes = scored_notes.merge(
            note_params[['noteId', 'noteIntercept', 'noteFactor']],
            left_on='noteId',
            right_on='noteId',
            how='left'
        )
        
        # Calculate final helpfulness score
        scored_notes['helpfulnessScore'] = scored_notes['noteIntercept']
        
        # Determine status based on thresholds
        scored_notes['status'] = scored_notes.apply(
            lambda row: self._determine_status(
                row['noteIntercept'],
                row.get('noteFactor', 0),
                self._get_rating_count(ratings_df, row['noteId'])
            ),
            axis=1
        )
        
        # Build user factors DataFrame
        user_factors = user_params[['raterParticipantId', 'raterIntercept', 'raterFactor']].copy()
        user_factors.columns = ['userId', 'userIntercept', 'userFactor']
        
        return scored_notes, user_factors
    
    def _run_simplified_scoring(
        self,
        notes_df: pd.DataFrame,
        ratings_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Simplified scoring when CN algorithm isn't available or data is limited.
        
        Uses basic vote aggregation with some bridging heuristics.
        """
        logger.info("Running simplified scoring (insufficient data for full MF)")
        
        scored_notes = notes_df.copy()
        
        # Calculate aggregate scores per note
        note_scores = ratings_df.groupby('noteId').agg({
            'helpfulNum': ['mean', 'count', 'std'],
            'raterParticipantId': 'nunique'
        }).reset_index()
        note_scores.columns = ['noteId', 'meanScore', 'ratingCount', 'scoreStd', 'uniqueRaters']
        
        # Simple intercept approximation: mean score adjusted for variance
        # Notes with high agreement from diverse raters score higher
        note_scores['noteIntercept'] = note_scores.apply(
            lambda row: self._calculate_simple_intercept(
                row['meanScore'],
                row['ratingCount'],
                row.get('scoreStd', 0) or 0,
                row['uniqueRaters']
            ),
            axis=1
        )
        note_scores['noteFactor'] = 0.0  # No factor inference in simplified mode
        
        scored_notes = scored_notes.merge(note_scores, on='noteId', how='left')
        scored_notes['helpfulnessScore'] = scored_notes['noteIntercept'].fillna(0)
        
        # Determine status
        scored_notes['status'] = scored_notes.apply(
            lambda row: self._determine_status(
                row.get('noteIntercept', 0) or 0,
                0,
                row.get('ratingCount', 0) or 0
            ),
            axis=1
        )
        
        # Simple user factors based on rating patterns
        user_factors = self._calculate_simple_user_factors(ratings_df)
        
        return scored_notes, user_factors
    
    def _calculate_simple_intercept(
        self,
        mean_score: float,
        rating_count: int,
        score_std: float,
        unique_raters: int
    ) -> float:
        """Calculate simplified note intercept."""
        if rating_count < self.settings.min_ratings_for_status:
            return 0.0
        
        # Base score from mean
        intercept = (mean_score - 0.5) * 2  # Scale to roughly -1 to 1
        
        # Bonus for low variance (agreement)
        if score_std < 0.3:
            intercept += 0.1
        
        # Bonus for more raters
        rater_bonus = min(0.2, unique_raters * 0.02)
        intercept += rater_bonus
        
        return intercept
    
    def _calculate_simple_user_factors(self, ratings_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate simplified user factors based on rating patterns."""
        user_stats = ratings_df.groupby('raterParticipantId').agg({
            'helpfulNum': ['mean', 'count', 'std'],
            'noteId': 'nunique'
        }).reset_index()
        user_stats.columns = ['userId', 'avgRating', 'totalRatings', 'ratingStd', 'notesRated']
        
        # Simple factor: how much user deviates from average
        # Users who rate more extremely get higher absolute factor
        global_mean = ratings_df['helpfulNum'].mean()
        user_stats['userFactor'] = (user_stats['avgRating'] - global_mean) * 2
        user_stats['userIntercept'] = user_stats['avgRating']
        
        return user_stats[['userId', 'userIntercept', 'userFactor', 'totalRatings']]
    
    def _get_rating_count(self, ratings_df: pd.DataFrame, note_id: int) -> int:
        """Get the number of ratings for a note."""
        return len(ratings_df[ratings_df['noteId'] == note_id])
    
    def _determine_status(
        self,
        note_intercept: Optional[float],
        note_factor: Optional[float],
        rating_count: int
    ) -> str:
        """Determine note status based on scores and thresholds."""
        if rating_count < self.settings.min_ratings_for_status:
            return NoteStatus.NEEDS_MORE_RATINGS.value
        
        if note_intercept is None:
            return NoteStatus.NEEDS_MORE_RATINGS.value
        
        # High intercept + low factor magnitude = bridging note
        if note_intercept >= self.settings.helpful_intercept_threshold:
            # Check factor isn't too extreme (would indicate partisan appeal)
            if note_factor is None or abs(note_factor) < self.settings.helpful_factor_threshold:
                return NoteStatus.CURRENTLY_RATED_HELPFUL.value
        
        if note_intercept < self.settings.not_helpful_intercept_threshold:
            return NoteStatus.CURRENTLY_RATED_NOT_HELPFUL.value
        
        return NoteStatus.NEEDS_MORE_RATINGS.value
    
    def _update_note_scores(self, scored_notes_df: pd.DataFrame) -> int:
        """Update note scores in database."""
        updated = 0
        now = datetime.now(UTC)
        
        for _, row in scored_notes_df.iterrows():
            note = self.db.query(Note).filter(Note.id == row['noteId']).first()
            if note:
                note.helpfulness_score = row.get('helpfulnessScore')
                note.note_intercept = row.get('noteIntercept')
                note.note_factor = row.get('noteFactor')
                note.status = row.get('status', NoteStatus.NEEDS_MORE_RATINGS.value)
                note.scored_at = now
                note.algorithm_version = self.ALGORITHM_VERSION
                updated += 1
        
        self.db.commit()
        return updated
    
    def _update_user_factors(self, user_factors_df: pd.DataFrame) -> int:
        """Update user factors in database."""
        updated = 0
        now = datetime.now(UTC)
        
        for _, row in user_factors_df.iterrows():
            user_id = int(row['userId'])
            
            user_factor = self.db.query(UserFactor).filter(
                UserFactor.user_id == user_id
            ).first()
            
            if not user_factor:
                user_factor = UserFactor(user_id=user_id)
                self.db.add(user_factor)
            
            user_factor.factor = row.get('userFactor')
            
            # Calculate confidence based on number of ratings
            total_ratings = int(row.get('totalRatings', 0))
            user_factor.total_ratings = total_ratings
            user_factor.factor_confidence = min(
                1.0,
                total_ratings / (self.settings.min_ratings_for_user_factor * 10)
            )
            
            user_factor.last_computed_at = now
            user_factor.algorithm_version = self.ALGORITHM_VERSION
            updated += 1
        
        self.db.commit()
        return updated
    
    def _finalize_scoring_run(
        self,
        scoring_run: ScoringRun,
        notes_updated: int,
        users_updated: int,
        start_time: float,
        errors: List[str]
    ) -> Dict:
        """Finalize and log the scoring run."""
        duration = time.time() - start_time
        now = datetime.now(UTC)
        
        scoring_run.completed_at = now
        scoring_run.notes_scored = notes_updated
        scoring_run.users_updated = users_updated
        scoring_run.duration_seconds = duration
        scoring_run.success = len(errors) == 0
        if errors:
            scoring_run.error_message = "; ".join(errors)
        
        self.db.commit()
        
        logger.info(
            f"Scoring complete: {notes_updated} notes, {users_updated} users in {duration:.2f}s"
        )
        
        return {
            "success": scoring_run.success,
            "notes_scored": notes_updated,
            "users_updated": users_updated,
            "duration_seconds": duration,
            "errors": errors,
            "scored_at": now.isoformat()
        }
