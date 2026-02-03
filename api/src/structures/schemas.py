"""
Pydantic schemas for OpenAI-compatible TTS API.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class NormalizationOptions(BaseModel):
    """Options for text normalization/sanitization."""
    
    normalize: bool = Field(
        default=True,
        description="Enable text normalization"
    )
    url_normalization: bool = Field(
        default=True,
        description="Normalize URLs for proper pronunciation"
    )
    email_normalization: bool = Field(
        default=True,
        description="Normalize email addresses for proper pronunciation"
    )
    phone_normalization: bool = Field(
        default=True,
        description="Normalize phone numbers for proper pronunciation"
    )
    number_normalization: bool = Field(
        default=True,
        description="Convert numbers to words"
    )
    replace_symbols: bool = Field(
        default=True,
        description="Replace symbols with word equivalents"
    )


class OpenAISpeechRequest(BaseModel):
    """
    OpenAI-compatible TTS request schema.
    Based on: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """
    
    model: str = Field(
        default="tts-1",
        description="TTS model to use. Supported: tts-1, tts-1-hd"
    )
    input: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The text to generate audio for (max 4096 chars)"
    )
    voice: str = Field(
        default="alloy",
        description="Voice to use. OpenAI voices: alloy, echo, fable, onyx, nova, shimmer. Or use Higgs Audio voice names."
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav"] = Field(
        default="mp3",
        description="Audio format to return"
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed of generated audio (0.25 to 4.0)"
    )
    
    # Higgs Audio specific extensions
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation (Higgs Audio extension)"
    )
    ref_audio: Optional[str] = Field(
        default=None,
        description="Reference audio for voice cloning (Higgs Audio extension)"
    )
    scene_description: Optional[str] = Field(
        default=None,
        description="Scene description for context (Higgs Audio extension)"
    )
    normalization_options: Optional[NormalizationOptions] = Field(
        default=None,
        description="Text normalization options"
    )


class VoiceInfo(BaseModel):
    """Information about an available voice."""
    
    id: str = Field(..., description="Voice identifier")
    name: str = Field(..., description="Human-readable voice name")
    description: Optional[str] = Field(None, description="Voice description")
    language: Optional[str] = Field(None, description="Primary language")


class VoicesResponse(BaseModel):
    """Response schema for voice listing."""
    
    voices: List[VoiceInfo] = Field(..., description="List of available voices")


class ModelInfo(BaseModel):
    """Information about an available model."""
    
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Human-readable model name")
    description: Optional[str] = Field(None, description="Model description")


class ModelsResponse(BaseModel):
    """Response schema for model listing."""
    
    models: List[ModelInfo] = Field(..., description="List of available models")


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
