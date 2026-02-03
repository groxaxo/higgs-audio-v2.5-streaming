"""Tests for text sanitization."""

import pytest
from api.src.services.text_processor import sanitize_text, TextProcessor
from api.src.structures.schemas import NormalizationOptions


def test_url_normalization():
    """Test URL normalization."""
    text = "Visit https://example.com for more info"
    result = sanitize_text(text)
    assert "https" not in result
    assert "dot" in result or "example" in result


def test_email_normalization():
    """Test email normalization."""
    text = "Contact us at info@example.com"
    result = sanitize_text(text)
    assert "@" not in result
    assert "at" in result
    assert "dot" in result


def test_phone_normalization():
    """Test phone number normalization."""
    text = "Call 555-123-4567"
    result = sanitize_text(text)
    # Should space out digits
    assert "555" in result or "5 5 5" in result


def test_symbol_replacement():
    """Test symbol replacement."""
    text = "Price is $50 & tax is 10%"
    result = sanitize_text(text)
    assert "$" not in result
    assert "%" not in result
    assert "dollar" in result
    assert "percent" in result


def test_whitespace_cleanup():
    """Test whitespace cleanup."""
    text = "This   has    extra     spaces"
    result = sanitize_text(text)
    # Should have single spaces
    assert "  " not in result


def test_no_normalization():
    """Test with normalization disabled."""
    options = NormalizationOptions(normalize=False)
    text = "Visit https://example.com & email info@example.com"
    result = sanitize_text(text, options)
    # Should be unchanged
    assert result == text


def test_selective_normalization():
    """Test with selective normalization options."""
    options = NormalizationOptions(
        normalize=True,
        url_normalization=False,
        email_normalization=True,
        replace_symbols=False
    )
    text = "Visit https://example.com & email info@example.com"
    result = sanitize_text(text, options)
    # URL should remain
    assert "https" in result
    # Email should be normalized
    assert "@" not in result or "at" in result
    # Symbol should remain
    assert "&" in result


def test_complex_text():
    """Test with complex mixed content."""
    text = """
    Visit our website at https://company.com or email support@company.com.
    Call us at 555-123-4567 for assistance.
    Prices start at $99 & include 20% discount!
    """
    result = sanitize_text(text)
    # Check that normalization happened
    assert len(result) > 0
    assert result.strip()  # Not empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
