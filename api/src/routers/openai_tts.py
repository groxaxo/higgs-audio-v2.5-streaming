"""
OpenAI-compatible TTS endpoint router.
"""

from fastapi import APIRouter, HTTPException, Response, Depends
from fastapi.responses import StreamingResponse
from loguru import logger
from typing import Optional

from ..structures.schemas import (
    OpenAISpeechRequest,
    VoicesResponse,
    VoiceInfo,
    ModelsResponse,
    ModelInfo,
    ErrorResponse
)
from ..services.tts_service import TTSService, convert_audio_format
from ..services.text_processor import sanitize_text


router = APIRouter(tags=["OpenAI Compatible TTS"])

# Global TTS service instance
_tts_service: Optional[TTSService] = None


async def get_tts_service() -> TTSService:
    """Dependency to get or create TTS service."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
        await _tts_service.initialize()
    return _tts_service


@router.post(
    "/audio/speech",
    response_class=Response,
    responses={
        200: {
            "content": {
                "audio/mpeg": {},
                "audio/opus": {},
                "audio/wav": {},
                "audio/flac": {},
                "audio/aac": {},
            },
            "description": "Generated audio file"
        },
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)
async def create_speech(
    request: OpenAISpeechRequest,
    tts_service: TTSService = Depends(get_tts_service)
):
    """
    Generate audio from text using OpenAI-compatible API.
    
    This endpoint is compatible with OpenAI's /v1/audio/speech endpoint.
    See: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """
    try:
        # Sanitize input text
        sanitized_text = sanitize_text(
            request.input,
            request.normalization_options
        )
        
        # Log request (without full text to avoid logging sensitive data)
        logger.info(f"Generating speech: voice={request.voice}, format={request.response_format}, text_length={len(request.input)}")
        
        # Generate audio
        audio_array, sample_rate = await tts_service.generate_audio(
            text=sanitized_text,
            voice=request.voice,
            temperature=request.temperature,
            ref_audio=request.ref_audio,
            scene_description=request.scene_description,
            speed=request.speed,
        )
        
        # Convert to requested format
        audio_bytes = await convert_audio_format(
            audio_array,
            sample_rate,
            request.response_format
        )
        
        # Determine content type
        content_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "flac": "audio/flac",
            "wav": "audio/wav",
        }
        content_type = content_types.get(request.response_format, "audio/mpeg")
        
        return Response(
            content=audio_bytes,
            media_type=content_type,
            headers={
                "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
            }
        )
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/voices",
    response_model=VoicesResponse,
    responses={
        200: {"model": VoicesResponse},
        500: {"model": ErrorResponse},
    }
)
async def list_voices(tts_service: TTSService = Depends(get_tts_service)):
    """
    List available voices.
    
    Returns a list of voice identifiers that can be used with the speech endpoint.
    """
    try:
        voices = await tts_service.get_available_voices()
        
        voice_infos = [
            VoiceInfo(
                id=voice,
                name=voice.replace("_", " ").title(),
                description=f"Higgs Audio voice: {voice}",
                language="en"
            )
            for voice in voices
        ]
        
        return VoicesResponse(voices=voice_infos)
        
    except Exception as e:
        logger.error(f"Error listing voices: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get(
    "/models",
    response_model=ModelsResponse,
    responses={
        200: {"model": ModelsResponse},
        500: {"model": ErrorResponse},
    }
)
async def list_models():
    """
    List available TTS models.
    
    Returns model identifiers compatible with OpenAI's model names.
    """
    try:
        models = [
            ModelInfo(
                id="tts-1",
                name="TTS-1 (Higgs Audio)",
                description="Fast, high-quality text-to-speech powered by Higgs Audio"
            ),
            ModelInfo(
                id="tts-1-hd",
                name="TTS-1-HD (Higgs Audio)",
                description="Higher quality text-to-speech powered by Higgs Audio"
            ),
        ]
        
        return ModelsResponse(models=models)
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
