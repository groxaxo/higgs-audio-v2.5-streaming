"""
TTS service wrapper for Higgs Audio engine.
"""

import io
import os
import subprocess
import tempfile
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


def _use_torchcodec():
    """
    Determine whether to use TorchCodec for audio encoding.
    
    Checks the AUDIO_CODEC environment variable:
    - "torchcodec": Force use of TorchCodec
    - "ffmpeg": Force use of FFmpeg
    - "auto" (default): Use TorchCodec if available, else FFmpeg
    
    Returns:
        bool: True if TorchCodec should be used, False for FFmpeg
    """
    mode = os.getenv("AUDIO_CODEC", "auto").lower()
    if mode == "torchcodec":
        return True
    if mode == "ffmpeg":
        return False
    # auto mode: try to import torchcodec
    try:
        import torchcodec  # noqa
        return True
    except Exception:
        return False


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
    
    Always operates on CPU to avoid GPU/CPU inconsistencies.
    Uses TorchCodec if available (controlled by AUDIO_CODEC env var),
    otherwise falls back to FFmpeg for non-WAV/FLAC formats.
    
    Args:
        audio_array: Audio data as numpy array
        sample_rate: Sample rate in Hz
        output_format: Target format (mp3, opus, wav, flac, aac)
        
    Returns:
        Audio bytes in target format
    """
    # Convert to torch tensor and always move to CPU for encoding
    audio_tensor = torch.from_numpy(audio_array).float()
    
    # Ensure we're on CPU for I/O operations
    if hasattr(audio_tensor, "detach"):
        audio_tensor = audio_tensor.detach().cpu()
    
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)
    
    # Create in-memory buffer
    buffer = io.BytesIO()
    
    # For WAV and FLAC, use torchaudio directly (works everywhere)
    if output_format in ["wav", "flac"]:
        torchaudio.save(
            buffer,
            audio_tensor,
            sample_rate,
            format=output_format
        )
        buffer.seek(0)
        return buffer.read()
    
    # For other formats (mp3, opus, aac), check codec strategy
    if _use_torchcodec():
        # Try using TorchCodec
        try:
            torchaudio.save(
                buffer,
                audio_tensor,
                sample_rate,
                format=output_format
            )
            buffer.seek(0)
            return buffer.read()
        except Exception as e:
            # If TorchCodec fails, fall back to FFmpeg
            pass
    
    # FFmpeg fallback for mp3, opus, aac
    # Save as WAV first, then convert using FFmpeg
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
        wav_path = wav_file.name
    
    with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as out_file:
        out_path = out_file.name
    
    try:
        # Save as WAV
        torchaudio.save(
            wav_path,
            audio_tensor,
            sample_rate,
            format="wav"
        )
        
        # Convert using FFmpeg
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", wav_path,
                "-y",  # Overwrite output
                "-hide_banner",
                "-loglevel", "error",
                out_path
            ],
            capture_output=True,
            check=True
        )
        
        # Read converted file
        with open(out_path, "rb") as f:
            return f.read()
            
    finally:
        # Clean up temporary files
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        if os.path.exists(out_path):
            os.unlink(out_path)
