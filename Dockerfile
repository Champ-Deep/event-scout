FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including Tesseract OCR and zbar for QR scanning
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    tesseract-ocr \
    tesseract-ocr-eng \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the SentenceTransformer model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY app.py .

# Create directories for data persistence
RUN mkdir -p /app/temp_images /app/saved_qr /app/data /app/users

# Expose port
EXPOSE 8000

# Environment variables must be provided at runtime (no defaults for security)
# APP_API_KEY and GEMINI_API_KEY are required

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
