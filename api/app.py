"""
Community Notes API Service - FastAPI Application.

REST API wrapper around the Community Notes scoring algorithm
for promise verification in the Accountability platform.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.database import init_db
from api.routes import notes_router, ratings_router, scoring_router


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Community Notes API Service...")
    init_db()
    logger.info("Database initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Community Notes API Service...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="""
## Community Notes API for Promise Verification

This API provides endpoints for the Accountability platform's promise 
verification system, powered by the Community Notes bridging algorithm.

### Key Concepts

- **Notes**: Fact-check submissions on political promises
- **Ratings**: Community votes on note helpfulness
- **Scoring**: Matrix factorization algorithm that identifies notes 
  helpful across different viewpoints (bridging)

### How Bridging Works

Notes are marked "helpful" not just when they receive many positive votes,
but when they receive positive votes from users with DIFFERENT viewpoints.
This prevents partisan content from dominating and surfaces genuinely
informative notes that appeal across the political spectrum.

### API Flow

1. User creates a note on a promise (POST /notes)
2. Other users rate the note (POST /ratings)
3. Scoring algorithm runs periodically (POST /scoring/run)
4. Helpful notes are surfaced (GET /notes/promise/{id}/scored)
    """,
    version=settings.app_version,
    lifespan=lifespan,
)

# Configure CORS
origins = settings.allowed_origins.split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(notes_router, prefix="/api")
app.include_router(ratings_router, prefix="/api")
app.include_router(scoring_router, prefix="/api")


# Root endpoint
@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Community Notes API for Promise Verification",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "endpoints": {
            "notes": "/api/notes",
            "ratings": "/api/ratings",
            "scoring": "/api/scoring",
            "health": "/api/scoring/health",
        }
    }


# Health check at root level too
@app.get("/health")
def root_health():
    """Quick health check."""
    return {"status": "ok", "version": settings.app_version}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.debug
    )
