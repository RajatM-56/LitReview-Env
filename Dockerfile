FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    LITREVIEW_DATA_DIR=/app/data \
    HOME=/home/appuser

WORKDIR /app

# Install system dependencies and create non-root user (required by HF Spaces)
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -m -u 1000 appuser

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

# Expose port (Hugging Face Spaces Docker SDK requires port 7860)
EXPOSE 7860

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run server
CMD ["streamlit", "run", "ui/app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]
