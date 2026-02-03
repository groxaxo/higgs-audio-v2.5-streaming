"""Basic API endpoint tests."""

import pytest


def test_config_import():
    """Test that config can be imported."""
    from api.src.core.config import settings
    assert settings.api_title == "Higgs Audio TTS API"


def test_schemas_import():
    """Test that schemas can be imported."""
    from api.src.structures.schemas import (
        OpenAISpeechRequest,
        VoicesResponse,
        ModelsResponse,
        NormalizationOptions
    )
    # Test creating instances
    request = OpenAISpeechRequest(input="test")
    assert request.input == "test"
    assert request.model == "tts-1"
    assert request.voice == "alloy"


def test_text_processor_import():
    """Test that text processor can be imported."""
    from api.src.services.text_processor import sanitize_text
    result = sanitize_text("test")
    assert result == "test"


# NOTE: The following tests would require torch and full model initialization
# They are commented out but show how to test the full API when dependencies are available

# @pytest.mark.skipif(not _has_torch(), reason="requires torch")  
# def test_health_endpoint():
#     """Test the health endpoint."""
#     from api.src.main import app
#     from fastapi.testclient import TestClient
#     client = TestClient(app)
#     response = client.get("/health")
#     assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
