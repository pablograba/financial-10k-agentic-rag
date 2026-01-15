# =============================================================================
# Agentic 10-K RAG - Multi-stage Dockerfile
# =============================================================================
# Uses uv for fast, reliable Python package management
# Produces minimal production images

FROM python:3.11-slim as base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# =============================================================================
# Builder stage - install dependencies
# =============================================================================
FROM base as builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into a virtual environment
RUN uv sync --frozen --no-dev --no-install-project

# =============================================================================
# Production stage
# =============================================================================
FROM base as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Add venv to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Create directories for data (will be mounted as volumes)
RUN mkdir -p /app/data/raw_10k

# Default command (overridden by docker-compose)
CMD ["python", "-m", "src.queue.worker"]

# =============================================================================
# Development stage (optional, for local dev with hot reload)
# =============================================================================
FROM base as development

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install all dependencies including dev
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy application code
COPY . .

# Default command for development
CMD ["python", "-m", "src.queue.worker", "--verbose"]
