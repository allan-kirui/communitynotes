"""
Routes package for Community Notes API.
"""

from api.routes.notes import router as notes_router
from api.routes.ratings import router as ratings_router
from api.routes.scoring import router as scoring_router

__all__ = ["notes_router", "ratings_router", "scoring_router"]
