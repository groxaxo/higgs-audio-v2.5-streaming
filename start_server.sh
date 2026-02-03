#!/bin/bash
# Startup script for Higgs Audio TTS API server

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting Higgs Audio OpenAI-Compatible TTS API Server...${NC}"

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${BLUE}Installing API dependencies...${NC}"
    pip install -r api/requirements.txt
fi

if ! python -c "import torch" 2>/dev/null; then
    echo -e "${BLUE}Installing core dependencies...${NC}"
    pip install -r requirements.txt
fi

# Check if the package is installed
if ! python -c "import boson_multimodal" 2>/dev/null; then
    echo -e "${BLUE}Installing boson_multimodal package...${NC}"
    pip install -e .
fi

# Set default environment variables if not set
export MODEL_PATH=${MODEL_PATH:-"bosonai/higgs-audio-v2-generation-3B-base"}
export AUDIO_TOKENIZER_PATH=${AUDIO_TOKENIZER_PATH:-"bosonai/higgs-audio-v2-tokenizer"}
export DEVICE=${DEVICE:-"auto"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
export PORT=${PORT:-8000}

echo -e "${GREEN}Configuration:${NC}"
echo "  MODEL_PATH: $MODEL_PATH"
echo "  AUDIO_TOKENIZER_PATH: $AUDIO_TOKENIZER_PATH"
echo "  DEVICE: $DEVICE"
echo "  LOG_LEVEL: $LOG_LEVEL"
echo "  PORT: $PORT"
echo ""

# Start the server
echo -e "${GREEN}Server starting at http://0.0.0.0:${PORT}${NC}"
echo -e "${GREEN}API documentation: http://localhost:${PORT}/docs${NC}"
echo ""

python -m uvicorn api.src.main:app --host 0.0.0.0 --port ${PORT}
