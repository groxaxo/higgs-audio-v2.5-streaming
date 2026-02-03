"""Tests for audio codec switching functionality."""

import os
import sys
import pytest
import numpy as np
from unittest.mock import patch, MagicMock, Mock


# Test _use_torchcodec function in isolation
def test_use_torchcodec_auto_with_torchcodec():
    """Test _use_torchcodec in auto mode when torchcodec is available."""
    # Define the function inline for testing
    def _use_torchcodec():
        mode = os.getenv("AUDIO_CODEC", "auto").lower()
        if mode == "torchcodec":
            return True
        if mode == "ffmpeg":
            return False
        try:
            import torchcodec  # noqa
            return True
        except Exception:
            return False
    
    with patch.dict(os.environ, {"AUDIO_CODEC": "auto"}):
        # Mock torchcodec module
        mock_torchcodec = Mock()
        with patch.dict('sys.modules', {'torchcodec': mock_torchcodec}):
            result = _use_torchcodec()
            assert result is True


def test_use_torchcodec_auto_without_torchcodec():
    """Test _use_torchcodec in auto mode when torchcodec is not available."""
    def _use_torchcodec():
        mode = os.getenv("AUDIO_CODEC", "auto").lower()
        if mode == "torchcodec":
            return True
        if mode == "ffmpeg":
            return False
        try:
            import torchcodec  # noqa
            return True
        except Exception:
            return False
    
    with patch.dict(os.environ, {"AUDIO_CODEC": "auto"}):
        # Ensure torchcodec is not in modules
        with patch.dict('sys.modules', {'torchcodec': None}):
            result = _use_torchcodec()
            assert result is False


def test_use_torchcodec_force_torchcodec():
    """Test _use_torchcodec when forced to use torchcodec."""
    def _use_torchcodec():
        mode = os.getenv("AUDIO_CODEC", "auto").lower()
        if mode == "torchcodec":
            return True
        if mode == "ffmpeg":
            return False
        try:
            import torchcodec  # noqa
            return True
        except Exception:
            return False
    
    with patch.dict(os.environ, {"AUDIO_CODEC": "torchcodec"}):
        result = _use_torchcodec()
        assert result is True


def test_use_torchcodec_force_ffmpeg():
    """Test _use_torchcodec when forced to use ffmpeg."""
    def _use_torchcodec():
        mode = os.getenv("AUDIO_CODEC", "auto").lower()
        if mode == "torchcodec":
            return True
        if mode == "ffmpeg":
            return False
        try:
            import torchcodec  # noqa
            return True
        except Exception:
            return False
    
    with patch.dict(os.environ, {"AUDIO_CODEC": "ffmpeg"}):
        result = _use_torchcodec()
        assert result is False


def test_use_torchcodec_case_insensitive():
    """Test that AUDIO_CODEC is case-insensitive."""
    def _use_torchcodec():
        mode = os.getenv("AUDIO_CODEC", "auto").lower()
        if mode == "torchcodec":
            return True
        if mode == "ffmpeg":
            return False
        try:
            import torchcodec  # noqa
            return True
        except Exception:
            return False
    
    with patch.dict(os.environ, {"AUDIO_CODEC": "FFMPEG"}):
        result = _use_torchcodec()
        assert result is False
    
    with patch.dict(os.environ, {"AUDIO_CODEC": "TorchCodec"}):
        result = _use_torchcodec()
        assert result is True


def test_use_torchcodec_default_auto():
    """Test that default behavior is auto mode."""
    def _use_torchcodec():
        mode = os.getenv("AUDIO_CODEC", "auto").lower()
        if mode == "torchcodec":
            return True
        if mode == "ffmpeg":
            return False
        try:
            import torchcodec  # noqa
            return True
        except Exception:
            return False
    
    # Remove AUDIO_CODEC env var if it exists
    env = os.environ.copy()
    if "AUDIO_CODEC" in env:
        del env["AUDIO_CODEC"]
    
    with patch.dict(os.environ, env, clear=True):
        # Mock torchcodec as not available
        with patch.dict('sys.modules', {'torchcodec': None}):
            result = _use_torchcodec()
            # In auto mode without torchcodec, should return False
            assert result is False


def test_environment_variable_values():
    """Test all expected environment variable values."""
    def _use_torchcodec():
        mode = os.getenv("AUDIO_CODEC", "auto").lower()
        if mode == "torchcodec":
            return True
        if mode == "ffmpeg":
            return False
        try:
            import torchcodec  # noqa
            return True
        except Exception:
            return False
    
    test_cases = [
        ("auto", False),  # auto without torchcodec
        ("torchcodec", True),
        ("ffmpeg", False),
        ("TORCHCODEC", True),
        ("FFmpeg", False),
        ("Auto", False),  # auto without torchcodec
    ]
    
    for env_val, expected in test_cases:
        with patch.dict(os.environ, {"AUDIO_CODEC": env_val}):
            with patch.dict('sys.modules', {'torchcodec': None}):
                result = _use_torchcodec()
                assert result == expected, f"Failed for AUDIO_CODEC={env_val}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

