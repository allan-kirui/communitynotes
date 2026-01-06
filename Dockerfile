# Community Notes API Service for Promise Verification
# Dockerfile for the API wrapper around CN scoring algorithm

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY api/requirements.txt /app/api/requirements.txt
COPY scoring/requirements.txt /app/scoring/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/scoring/requirements.txt

# Copy application code
COPY api/ /app/api/
COPY scoring/ /app/scoring/

# Create data directory
RUN mkdir -p /app/data

# Set environment variables
ENV PYTHONPATH=/app:/app/scoring/src
ENV CN_DATABASE_URL=sqlite:///./data/community_notes.db
ENV CN_DEBUG=false
ENV CN_LOG_LEVEL=INFO

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8001"]
