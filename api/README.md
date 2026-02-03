# Higgs Audio OpenAI-Compatible TTS API

This directory contains the FastAPI-based OpenAI-compatible TTS server for Higgs Audio.

## Features

- ✅ **OpenAI API Compatible**: Drop-in replacement for OpenAI's `/v1/audio/speech` endpoint
- ✅ **Multiple Voice Support**: OpenAI voice names + native Higgs Audio voices
- ✅ **Text Sanitization**: Automatic URL, email, phone number, and symbol normalization
- ✅ **Multiple Audio Formats**: MP3, Opus, WAV, FLAC, AAC
- ✅ **Speed Control**: Adjust playback speed from 0.25x to 4.0x
- ✅ **Docker Support**: Production-ready containerization

## Quick Start

### Installation

```bash
# Install API dependencies
pip install -r requirements.txt

# Install core Higgs Audio dependencies
pip install -r ../requirements.txt
pip install -e ..
```

### Running the Server

```bash
# Start the server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Or run directly
python src/main.py
```

### Using Docker

```bash
# From the repository root
docker build -t higgs-audio-tts .
docker run -p 8000:8000 higgs-audio-tts
```

## API Endpoints

### Generate Speech (OpenAI Compatible)

```bash
POST /v1/audio/speech
```

**Request:**
```json
{
  "model": "tts-1",
  "input": "Hello, world!",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0
}
```

**Response:** Audio file in requested format

### List Voices

```bash
GET /v1/voices
```

Returns a list of available voice identifiers.

### List Models

```bash
GET /v1/models
```

Returns supported model identifiers.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_text_sanitization.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Configuration

Environment variables:

- `MODEL_PATH`: Higgs Audio model path (default: `bosonai/higgs-audio-v2-generation-3B-base`)
- `AUDIO_TOKENIZER_PATH`: Tokenizer path (default: `bosonai/higgs-audio-v2-tokenizer`)
- `DEVICE`: Device to use (`auto`, `cuda`, `cpu`, `mps`)
- `LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `PORT`: Server port (default: `8000`)

## Architecture

```
api/
├── src/
│   ├── core/           # Configuration
│   ├── routers/        # API endpoints
│   ├── services/       # Business logic
│   │   ├── tts_service.py      # TTS wrapper
│   │   └── text_processor.py   # Text sanitization
│   ├── structures/     # Pydantic schemas
│   └── main.py         # FastAPI app
└── tests/              # Test suite
```

## License

See the main repository LICENSE file.
