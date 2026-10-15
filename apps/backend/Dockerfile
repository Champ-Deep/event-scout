FROM python:3.11-slim

WORKDIR /app

# Logs straight to the platform's log viewer (print() is the app's logger); no .pyc files in the image.
# HF_HOME inside /app so the embedding model downloaded at build time is readable by the app user.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    HF_HOME=/app/.cache/huggingface

# System packages: fonts for the business card image, zbar for QR decoding, OpenCV/Pillow runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-dejavu-core \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    libzbar0 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install CPU-only PyTorch first to prevent CUDA variant being pulled by sentence-transformers
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model during build (no network needed at startup)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Application code
COPY VERSION .
# Every backend module (app.py imports a dozen of them at startup; a missing one crashes the boot).
# Tests and one-off scripts are kept out by .dockerignore.
COPY *.py ./
# Ask AI rich answers (OpenUI component prompt). Without it the server silently answers in plain text.
COPY openui_prompt.txt .

# Scratch directories (scan uploads are temporary; everything that matters lives in the database)
RUN mkdir -p /app/temp_images /app/saved_qr

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/api/health" || exit 1

# Required at runtime (the server refuses to start without them in production — see .env.example):
#   APP_API_KEY, JWT_SECRET, DATABASE_URL (Postgres), OPENROUTER_API_KEY, PUBLIC_BASE_URL
# One worker on purpose: the live feed (SSE), CRM jobs and the search index live in this process.
# --proxy-headers: Railway / a load balancer terminates HTTPS; trust X-Forwarded-For/Proto for the
# real client IP (rate limits) and https links.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --forwarded-allow-ips='*'"]
