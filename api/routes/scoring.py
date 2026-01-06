"""
API routes for triggering and monitoring scoring.
"""

from datetime import datetime, UTC
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Header
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from api.config import get_settings
from api.database import get_db, Note, Rating, ScoringRun
from api.models import (
    TriggerScoringRequest, ScoringResultResponse, HealthResponse
)
from api.scoring_service import ScoringService


router = APIRouter(prefix="/scoring", tags=["Scoring"])

settings = get_settings()


# =============================================================================
# Trigger Scoring
# =============================================================================


@router.post("/run", response_model=ScoringResultResponse)
def trigger_scoring(
    request: TriggerScoringRequest,
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(default=None)
):
    """
    Manually trigger the scoring algorithm.
    
    This runs the Community Notes matrix factorization algorithm to
    update note statuses and user factors. In production, this is
    typically called by a scheduled job every 15-30 minutes.
    
    Requires API key if configured.
    """
    # Check API key if configured
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    scoring_service = ScoringService(db)
    
    result = scoring_service.run_scoring(
        promise_ids=request.promise_ids,
        force_full_rescore=request.force_full_rescore
    )
    
    return ScoringResultResponse(
        success=result["success"],
        notes_scored=result["notes_scored"],
        users_updated=result["users_updated"],
        duration_seconds=result["duration_seconds"],
        errors=result.get("errors", []),
        scored_at=datetime.fromisoformat(result["scored_at"])
    )


@router.post("/run-async")
async def trigger_scoring_async(
    request: TriggerScoringRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    x_api_key: Optional[str] = Header(default=None)
):
    """
    Trigger scoring in the background.
    
    Returns immediately and runs scoring asynchronously.
    Use GET /scoring/status to check progress.
    """
    # Check API key if configured
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    def run_scoring_background():
        from api.database import SessionLocal
        with SessionLocal() as session:
            service = ScoringService(session)
            service.run_scoring(
                promise_ids=request.promise_ids,
                force_full_rescore=request.force_full_rescore
            )
    
    background_tasks.add_task(run_scoring_background)
    
    return {"message": "Scoring started in background", "status": "running"}


# =============================================================================
# Scoring Status
# =============================================================================


@router.get("/status")
def get_scoring_status(db: Session = Depends(get_db)):
    """Get the status of recent scoring runs."""
    # Get last 10 scoring runs
    recent_runs = db.query(ScoringRun).order_by(
        desc(ScoringRun.started_at)
    ).limit(10).all()
    
    return {
        "recent_runs": [
            {
                "id": run.id,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                "success": run.success,
                "notes_scored": run.notes_scored,
                "users_updated": run.users_updated,
                "duration_seconds": run.duration_seconds,
                "error": run.error_message,
            }
            for run in recent_runs
        ]
    }


@router.get("/last-run")
def get_last_scoring_run(db: Session = Depends(get_db)):
    """Get details of the last scoring run."""
    last_run = db.query(ScoringRun).order_by(
        desc(ScoringRun.completed_at)
    ).first()
    
    if not last_run:
        return {"message": "No scoring runs yet"}
    
    return {
        "id": last_run.id,
        "started_at": last_run.started_at.isoformat() if last_run.started_at else None,
        "completed_at": last_run.completed_at.isoformat() if last_run.completed_at else None,
        "success": last_run.success,
        "notes_scored": last_run.notes_scored,
        "users_updated": last_run.users_updated,
        "duration_seconds": last_run.duration_seconds,
        "error": last_run.error_message,
        "algorithm_version": last_run.algorithm_version,
    }


# =============================================================================
# Health Check
# =============================================================================


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """
    Health check endpoint.
    
    Returns service health status and basic statistics.
    """
    try:
        # Test database connection
        db.execute("SELECT 1")
        db_connected = True
    except Exception:
        db_connected = False
    
    # Get counts
    notes_count = db.query(func.count(Note.id)).scalar() or 0
    ratings_count = db.query(func.count(Rating.id)).scalar() or 0
    
    # Get last scoring run
    last_run = db.query(ScoringRun).order_by(
        desc(ScoringRun.completed_at)
    ).first()
    
    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        version=settings.app_version,
        database_connected=db_connected,
        last_scoring_run=last_run.completed_at if last_run else None,
        notes_count=notes_count,
        ratings_count=ratings_count
    )
