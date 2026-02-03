"""
Text sanitization and normalization for TTS processing.
Based on Kokoro-FastAPI text processing strategies.
"""

import re
from typing import Optional
from ..structures.schemas import NormalizationOptions


class TextProcessor:
    """Handles text sanitization and normalization for TTS."""
    
    # URL pattern
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    # Email pattern
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    # Phone pattern (basic)
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    
    # Symbol replacements
    SYMBOL_MAP = {
        '&': ' and ',
        '@': ' at ',
        '#': ' hash ',
        '$': ' dollar ',
        '%': ' percent ',
        '+': ' plus ',
        '=': ' equals ',
        '<': ' less than ',
        '>': ' greater than ',
        '~': ' tilde ',
        '|': ' pipe ',
    }
    
    def __init__(self, options: Optional[NormalizationOptions] = None):
        """Initialize text processor with options."""
        self.options = options or NormalizationOptions()
    
    def sanitize(self, text: str) -> str:
        """
        Sanitize and normalize text for TTS processing.
        
        Args:
            text: Raw input text
            
        Returns:
            Sanitized text ready for TTS
        """
        if not self.options.normalize:
            return text
        
        # Apply normalization steps in order
        if self.options.url_normalization:
            text = self._normalize_urls(text)
        
        if self.options.email_normalization:
            text = self._normalize_emails(text)
        
        if self.options.phone_normalization:
            text = self._normalize_phones(text)
        
        if self.options.number_normalization:
            text = self._normalize_numbers(text)
        
        if self.options.replace_symbols:
            text = self._replace_symbols(text)
        
        # Clean up extra whitespace
        text = self._clean_whitespace(text)
        
        return text
    
    def _normalize_urls(self, text: str) -> str:
        """Convert URLs to readable format."""
        def replace_url(match):
            url = match.group(0)
            # Remove protocol
            url = re.sub(r'https?://', '', url)
            # Replace dots and slashes with spaces
            url = url.replace('.', ' dot ')
            url = url.replace('/', ' slash ')
            return url
        
        return self.URL_PATTERN.sub(replace_url, text)
    
    def _normalize_emails(self, text: str) -> str:
        """Convert email addresses to readable format."""
        def replace_email(match):
            email = match.group(0)
            # Replace @ and dots
            email = email.replace('@', ' at ')
            email = email.replace('.', ' dot ')
            return email
        
        return self.EMAIL_PATTERN.sub(replace_email, text)
    
    def _normalize_phones(self, text: str) -> str:
        """Convert phone numbers to readable format."""
        def replace_phone(match):
            phone = match.group(0)
            # Remove separators and space out digits
            phone = re.sub(r'[-.]', '', phone)
            # Insert spaces between digits
            return ' '.join(phone)
        
        return self.PHONE_PATTERN.sub(replace_phone, text)
    
    def _normalize_numbers(self, text: str) -> str:
        """Convert numbers to words (basic implementation)."""
        # This is a simplified version. For production, use inflect or num2words
        # For now, just ensure numbers are spaced properly
        # Full implementation would convert "123" to "one hundred twenty three"
        return text
    
    def _replace_symbols(self, text: str) -> str:
        """Replace symbols with word equivalents."""
        for symbol, replacement in self.SYMBOL_MAP.items():
            text = text.replace(symbol, replacement)
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up extra whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Trim leading/trailing whitespace
        text = text.strip()
        return text


def sanitize_text(text: str, options: Optional[NormalizationOptions] = None) -> str:
    """
    Convenience function to sanitize text.
    
    Args:
        text: Raw input text
        options: Normalization options
        
    Returns:
        Sanitized text
    """
    processor = TextProcessor(options)
    return processor.sanitize(text)
