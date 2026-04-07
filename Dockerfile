FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    LITREVIEW_DATA_DIR=/app/data

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY server/ ./server/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY ui/ ./ui/
COPY openenv.yaml .

# Expose port
EXPOSE 8501

# Health check (Streamlit has a default health endpoint)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run server
CMD ["streamlit", "run", "ui/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
