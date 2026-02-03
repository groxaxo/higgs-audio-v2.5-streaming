# Higgs Audio TTS API Server Dockerfile
# Production-ready container with GPU/CPU support

FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Set environment variables
ENV PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    git \
    wget \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY api/requirements.txt api/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir -r api/requirements.txt

# Copy application code
COPY . .

# Install the package
RUN pip3 install -e .

# Expose port
EXPOSE 8000

# Set default environment variables
ENV MODEL_PATH="bosonai/higgs-audio-v2-generation-3B-base"
ENV AUDIO_TOKENIZER_PATH="bosonai/higgs-audio-v2-tokenizer"
ENV DEVICE="auto"
ENV LOG_LEVEL="INFO"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# Run the server
CMD ["python3", "-m", "uvicorn", "api.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
