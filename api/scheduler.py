"""
Scheduler for periodic Community Notes scoring.

This module provides a background scheduler that periodically triggers
the Community Notes scoring algorithm to update note statuses and user factors.
"""

import asyncio
import logging
from datetime import datetime, UTC
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from api.config import get_settings
from api.database import SessionLocal
from api.scoring_service import ScoringService


logger = logging.getLogger(__name__)

settings = get_settings()


class ScoringScheduler:
    """
    Scheduler for periodic Community Notes scoring runs.
    
    Runs the scoring algorithm at configured intervals to update
    note statuses and user factors based on community ratings.
    """
    
    def __init__(self, interval_minutes: Optional[int] = None):
        """
        Initialize the scheduler.
        
        Args:
            interval_minutes: Scoring interval (default from settings)
        """
        self.interval_minutes = interval_minutes or settings.scoring_interval_minutes
        self.scheduler = AsyncIOScheduler()
        self._is_running = False
        self._last_run: Optional[datetime] = None
        self._last_result: Optional[dict] = None
    
    async def run_scoring(self):
        """
        Run a scoring iteration.
        
        This is called by the scheduler at each interval.
        """
        if self._is_running:
            logger.warning("Scoring already in progress, skipping this iteration")
            return
        
        self._is_running = True
        start_time = datetime.now(UTC)
        
        try:
            logger.info(f"Starting scheduled scoring run at {start_time.isoformat()}")
            
            with SessionLocal() as db:
                service = ScoringService(db)
                result = service.run_scoring()
                
                self._last_run = datetime.now(UTC)
                self._last_result = result
                
                logger.info(
                    f"Scheduled scoring complete: "
                    f"{result['notes_scored']} notes, "
                    f"{result['users_updated']} users in "
                    f"{result['duration_seconds']:.2f}s"
                )
                
        except Exception as e:
            logger.exception(f"Error in scheduled scoring: {e}")
            self._last_result = {"success": False, "error": str(e)}
        finally:
            self._is_running = False
    
    def start(self):
        """Start the scheduler."""
        if self.scheduler.running:
            logger.warning("Scheduler already running")
            return
        
        # Add the scoring job
        self.scheduler.add_job(
            self.run_scoring,
            trigger=IntervalTrigger(minutes=self.interval_minutes),
            id="community_notes_scoring",
            name="Community Notes Scoring",
            replace_existing=True,
            max_instances=1,  # Prevent overlapping runs
        )
        
        self.scheduler.start()
        logger.info(
            f"Scoring scheduler started - running every {self.interval_minutes} minutes"
        )
    
    def stop(self):
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
            logger.info("Scoring scheduler stopped")
    
    def get_status(self) -> dict:
        """Get scheduler status."""
        return {
            "running": self.scheduler.running,
            "interval_minutes": self.interval_minutes,
            "last_run": self._last_run.isoformat() if self._last_run else None,
            "last_result": self._last_result,
            "is_scoring": self._is_running,
            "next_run": self._get_next_run_time(),
        }
    
    def _get_next_run_time(self) -> Optional[str]:
        """Get the next scheduled run time."""
        if not self.scheduler.running:
            return None
        
        job = self.scheduler.get_job("community_notes_scoring")
        if job and job.next_run_time:
            return job.next_run_time.isoformat()
        return None


# Global scheduler instance
_scheduler: Optional[ScoringScheduler] = None


def get_scheduler() -> ScoringScheduler:
    """Get or create the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = ScoringScheduler()
    return _scheduler


def start_scheduler():
    """Start the global scheduler."""
    scheduler = get_scheduler()
    scheduler.start()


def stop_scheduler():
    """Stop the global scheduler."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()
        _scheduler = None


# CLI entry point for running scheduler standalone
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Community Notes Scoring Scheduler")
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Scoring interval in minutes (default from settings)"
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run scoring once and exit"
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    if args.run_once:
        # Run once
        with SessionLocal() as db:
            service = ScoringService(db)
            result = service.run_scoring()
            print(f"Scoring complete: {result}")
    else:
        # Start scheduler
        scheduler = ScoringScheduler(interval_minutes=args.interval)
        scheduler.start()
        
        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            scheduler.stop()
            print("\nScheduler stopped")
