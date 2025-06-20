FROM python:3.10-slim

# Install system dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    libmagic1 \
    libmagic-dev \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/results /app/temp

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8753/health || exit 1

# Expose port
EXPOSE 8753

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8753", "--reload"]
