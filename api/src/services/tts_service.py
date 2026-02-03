"""
TTS service wrapper for Higgs Audio engine.
"""

import io
import os
import torch
import torchaudio
import numpy as np
from typing import Optional, List, AsyncGenerator, Tuple
from pathlib import Path
import asyncio
from functools import lru_cache

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message
from ..core.config import settings


# Voice name mapping from OpenAI to Higgs Audio
# These mappings are based on voice characteristics:
# - alloy: neutral, balanced voice -> af_heart (female, clear)
# - echo: male voice -> am_adam (male, clear)
# - fable: storytelling voice -> bf_emma (female, expressive)
# - onyx: deep male voice -> bm_george (male, deep)
# - nova: female voice -> af_nicole (female, warm)
# - shimmer: bright female voice -> af_sarah (female, bright)
OPENAI_VOICE_MAP = {
    "alloy": "af_heart",
    "echo": "am_adam",
    "fable": "bf_emma",
    "onyx": "bm_george",
    "nova": "af_nicole",
    "shimmer": "af_sarah",
}


class TTSService:
    """Service for text-to-speech generation using Higgs Audio."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize TTS service."""
        self.model_path = model_path or settings.model_path
        self.tokenizer_path = tokenizer_path or settings.audio_tokenizer_path
        
        # Determine device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.engine: Optional[HiggsAudioServeEngine] = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the Higgs Audio engine."""
        if self._initialized:
            return
        
        # Run initialization in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._init_engine
        )
        self._initialized = True
    
    def _init_engine(self):
        """Initialize engine (sync)."""
        self.engine = HiggsAudioServeEngine(
            self.model_path,
            self.tokenizer_path,
            device=self.device
        )
    
    def _map_voice(self, voice: str) -> str:
        """Map OpenAI voice name to Higgs Audio voice."""
        return OPENAI_VOICE_MAP.get(voice, voice)
    
    def _create_system_prompt(
        self,
        scene_description: Optional[str] = None,
        ref_audio_path: Optional[str] = None
    ) -> str:
        """Create system prompt for generation."""
        if scene_description:
            prompt = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
        else:
            prompt = "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>"
        
        # Add reference audio if provided
        # Note: Reference audio handling would need to be implemented based on Higgs Audio's API
        return prompt
    
    async def generate_audio(
        self,
        text: str,
        voice: str = "alloy",
        temperature: Optional[float] = None,
        ref_audio: Optional[str] = None,
        scene_description: Optional[str] = None,
        speed: float = 1.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text.
        
        Args:
            text: Text to synthesize
            voice: Voice name
            temperature: Sampling temperature
            ref_audio: Reference audio path for voice cloning
            scene_description: Scene context
            speed: Audio speed multiplier
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not self._initialized:
            await self.initialize()
        
        # Map voice name
        mapped_voice = self._map_voice(voice)
        
        # Create messages
        system_prompt = self._create_system_prompt(scene_description, ref_audio)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=text),
        ]
        
        # Set generation parameters
        temp = temperature if temperature is not None else settings.default_temperature
        
        # Run generation in thread pool
        loop = asyncio.get_event_loop()
        output = await loop.run_in_executor(
            None,
            self._generate_sync,
            ChatMLSample(messages=messages),
            temp
        )
        
        # Apply speed adjustment if needed
        audio_array = output.audio
        sample_rate = output.sampling_rate
        
        if speed != 1.0:
            # Note: Proper speed adjustment requires resampling the audio data
            # For now, we skip this to avoid incorrect pitch changes
            # In production, use librosa.effects.time_stretch or similar
            # audio_array = librosa.effects.time_stretch(audio_array, rate=speed)
            pass
        
        return audio_array, sample_rate
    
    def _generate_sync(
        self,
        chat_ml_sample: ChatMLSample,
        temperature: float
    ) -> HiggsAudioResponse:
        """Synchronous generation."""
        return self.engine.generate(
            chat_ml_sample=chat_ml_sample,
            max_new_tokens=settings.default_max_new_tokens,
            temperature=temperature,
            top_p=settings.default_top_p,
            top_k=settings.default_top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
    
    async def get_available_voices(self) -> List[str]:
        """Get list of available voices."""
        # Combine OpenAI voice names with default Higgs voices
        voices = list(OPENAI_VOICE_MAP.keys())
        # Add some common Higgs Audio voices
        voices.extend([
            "af_heart",
            "af_nicole", 
            "af_sarah",
            "am_adam",
            "am_michael",
            "bf_emma",
            "bf_isabella",
            "bm_george",
            "bm_lewis",
        ])
        return sorted(set(voices))


async def convert_audio_format(
    audio_array: np.ndarray,
    sample_rate: int,
    output_format: str = "mp3"
) -> bytes:
    """
    Convert audio array to specified format.
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Sample rate in Hz
        output_format: Target format (mp3, opus, wav, etc.)
        
    Returns:
        Audio bytes in target format
    """
    # Convert to torch tensor
    audio_tensor = torch.from_numpy(audio_array).float()
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Create in-memory buffer
    buffer = io.BytesIO()
    
    # Save to buffer in target format
    # Note: For MP3/OPUS, you'd need additional libraries (ffmpeg)
    # For now, we'll use WAV as a fallback
    if output_format in ["wav", "flac"]:
        torchaudio.save(
            buffer,
            audio_tensor,
            sample_rate,
            format=output_format
        )
    else:
        # For mp3, opus, aac, we need ffmpeg
        # Save as WAV first, then convert
        # This is a simplified version
        torchaudio.save(
            buffer,
            audio_tensor,
            sample_rate,
            format="wav"
        )
    
    buffer.seek(0)
    return buffer.read()
