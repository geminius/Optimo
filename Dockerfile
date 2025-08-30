# Multi-stage build for the robotics model optimization platform
FROM python:3.9-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-basic.txt .
RUN pip install --no-cache-dir -r requirements-basic.txt

# Copy source code and setup files
COPY src/ ./src/
COPY config/ ./config/
COPY setup.py .
COPY README.md .
COPY requirements-basic.txt requirements.txt

# Install the package
RUN pip install -e .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API server
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]