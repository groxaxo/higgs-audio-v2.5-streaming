"""
Configuration settings for the Higgs Audio OpenAI-compatible TTS API.
"""

import os
from typing import List
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False
    )
    
    # API metadata
    api_title: str = "Higgs Audio TTS API"
    api_description: str = "OpenAI-compatible Text-to-Speech API powered by Higgs Audio"
    api_version: str = "1.0.0"
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Model paths
    model_path: str = os.getenv("MODEL_PATH", "bosonai/higgs-audio-v2-generation-3B-base")
    audio_tokenizer_path: str = os.getenv("AUDIO_TOKENIZER_PATH", "bosonai/higgs-audio-v2-tokenizer")
    
    # Device configuration
    device: str = os.getenv("DEVICE", "auto")  # auto, cuda, cpu
    
    # Generation defaults
    default_temperature: float = 0.3
    default_top_p: float = 0.95
    default_top_k: int = 50
    default_max_new_tokens: int = 1024
    
    # CORS settings
    cors_enabled: bool = True
    cors_origins: List[str] = ["*"]
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Supported audio formats
    supported_formats: List[str] = ["mp3", "opus", "wav", "flac", "aac"]
    default_format: str = "mp3"
    
    # Supported voices (can be expanded)
    default_voice: str = "af_heart"


settings = Settings()
