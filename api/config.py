"""
Configuration settings for Community Notes API Service.
"""

import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Settings
    app_name: str = "Community Notes API"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Database
    database_url: str = "sqlite:///./community_notes.db"
    
    # Scoring Configuration
    scoring_interval_minutes: int = 30  # How often to run batch scoring
    min_ratings_for_status: int = 5     # Minimum ratings before note can get status
    min_ratings_for_user_factor: int = 5  # Min ratings before user factor is reliable
    
    # Thresholds for note status (can be tuned for political content)
    helpful_intercept_threshold: float = 0.40  # Note intercept must exceed this
    not_helpful_intercept_threshold: float = -0.05  # Below this = not helpful
    helpful_factor_threshold: float = 0.5  # Factor magnitude threshold
    
    # Matrix Factorization Parameters
    mf_num_factors: int = 1  # Single factor for left-right spectrum
    mf_learning_rate: float = 0.2
    mf_regularization: float = 0.03
    
    # API Security
    api_key: Optional[str] = None  # Optional API key for internal service auth
    allowed_origins: str = "http://localhost:8000,http://localhost:3000"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_prefix = "CN_"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
