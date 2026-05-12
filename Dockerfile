# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (markitdown needs libmagic; lxml needs gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install wheels from local cache first (avoids pypi for air-gapped envs)
COPY wheels/ /tmp/wheels/
COPY pyproject.toml ./

# Install local wheels then project dependencies
RUN pip install /tmp/wheels/*.whl 2>/dev/null || true
RUN pip install -e .

# Copy source
COPY src/ ./src/
COPY config/ ./config/
COPY EXAMPLE/ ./EXAMPLE/

# Runtime directories (mounted as volumes in production)
RUN mkdir -p /app/wiki /app/content_new /app/content_processed /app/content_error /app/logs

EXPOSE 8080

CMD ["python", "-m", "src", "--help"]
