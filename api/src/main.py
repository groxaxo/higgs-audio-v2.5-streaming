"""
Higgs Audio OpenAI-Compatible TTS API Server
"""

import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from loguru import logger

from .core.config import settings
from .routers import openai_tts


def setup_logger():
    """Configure loguru logger."""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )


setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization."""
    logger.info("Starting Higgs Audio TTS API server...")
    logger.info(f"Model path: {settings.model_path}")
    logger.info(f"Device: {settings.device}")
    
    # Pre-initialize TTS service
    from .services.tts_service import TTSService
    tts_service = TTSService()
    await tts_service.initialize()
    logger.success("TTS service initialized successfully")
    
    yield
    
    logger.info("Shutting down server...")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
if settings.cors_enabled:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(openai_tts.router, prefix="/v1", tags=["OpenAI Compatible"])


@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API information."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Higgs Audio TTS API</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 20px;
                line-height: 1.6;
                background: #f5f5f5;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                color: #34495e;
                margin-top: 30px;
            }
            .badge {
                display: inline-block;
                padding: 4px 12px;
                background: #3498db;
                color: white;
                border-radius: 4px;
                font-size: 14px;
                margin-right: 10px;
            }
            .endpoint {
                background: #ecf0f1;
                padding: 15px;
                margin: 10px 0;
                border-radius: 4px;
                border-left: 4px solid #3498db;
            }
            code {
                background: #2c3e50;
                color: #ecf0f1;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }
            pre {
                background: #2c3e50;
                color: #ecf0f1;
                padding: 15px;
                border-radius: 4px;
                overflow-x: auto;
            }
            .example {
                margin: 20px 0;
            }
            .credits {
                background: #fff3cd;
                border: 1px solid #ffc107;
                padding: 15px;
                border-radius: 4px;
                margin-top: 30px;
            }
            a {
                color: #3498db;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎙️ Higgs Audio TTS API</h1>
            <p>
                <span class="badge">OpenAI Compatible</span>
                <span class="badge">v1.0.0</span>
            </p>
            
            <p>
                High-fidelity text-to-speech API powered by <strong>Higgs Audio v2.5</strong>.
                This server provides OpenAI-compatible endpoints for seamless integration.
            </p>
            
            <h2>📡 Available Endpoints</h2>
            
            <div class="endpoint">
                <strong>POST /v1/audio/speech</strong>
                <p>Generate speech from text (OpenAI-compatible)</p>
            </div>
            
            <div class="endpoint">
                <strong>GET /v1/voices</strong>
                <p>List available voices</p>
            </div>
            
            <div class="endpoint">
                <strong>GET /v1/models</strong>
                <p>List available models</p>
            </div>
            
            <div class="endpoint">
                <strong>GET /docs</strong>
                <p>Interactive API documentation (Swagger UI)</p>
            </div>
            
            <h2>🎯 Example Usage</h2>
            
            <div class="example">
                <h3>Basic Text-to-Speech</h3>
                <pre>curl -X POST http://localhost:8000/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test of the Higgs Audio TTS system.",
    "voice": "alloy",
    "response_format": "mp3"
  }' \\
  --output speech.mp3</pre>
            </div>
            
            <div class="example">
                <h3>With Speed Control</h3>
                <pre>curl -X POST http://localhost:8000/v1/audio/speech \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "tts-1",
    "input": "The quick brown fox jumps over the lazy dog.",
    "voice": "nova",
    "speed": 1.25,
    "response_format": "mp3"
  }' \\
  --output speech.mp3</pre>
            </div>
            
            <div class="example">
                <h3>List Available Voices</h3>
                <pre>curl http://localhost:8000/v1/voices</pre>
            </div>
            
            <h2>🎤 Supported Voices</h2>
            <p><strong>OpenAI-compatible voices:</strong> alloy, echo, fable, onyx, nova, shimmer</p>
            <p><strong>Native Higgs Audio voices:</strong> af_heart, af_nicole, af_sarah, am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis</p>
            
            <h2>🎵 Supported Formats</h2>
            <p>mp3, opus, aac, flac, wav</p>
            
            <h2>📚 Documentation</h2>
            <p>
                Visit <a href="/docs">/docs</a> for interactive API documentation and detailed parameter information.
            </p>
            
            <div class="credits">
                <h2>🙏 Credits & Acknowledgements</h2>
                <p>
                    This API is powered by <strong>Higgs Audio v2.5</strong> from Boson AI.
                    Higgs Audio is a state-of-the-art audio foundation model achieving 
                    exceptional performance on TTS benchmarks.
                </p>
                <p>
                    <strong>Model Citation:</strong><br>
                    Boson AI. (2025). Higgs Audio V2: Redefining Expressiveness in Audio Generation.
                    <a href="https://github.com/boson-ai/higgs-audio" target="_blank">
                        https://github.com/boson-ai/higgs-audio
                    </a>
                </p>
                <p>
                    <strong>Architecture inspired by:</strong>
                    <a href="https://github.com/groxaxo/Kokoro-FastAPI" target="_blank">
                        Kokoro-FastAPI
                    </a>
                </p>
                <p>
                    <em>This implementation does not imply endorsement by the original model creators.</em>
                </p>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "higgs-audio-tts"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
