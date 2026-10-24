"""
Text preprocessing utilities for agricultural farmer query classification.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Cleans and normalizes query text:
    - Converts text to lowercase
    - Removes punctuation and special symbols while retaining alphanumeric words
    - Collapses multiple whitespace characters into single space
    - Strips leading and trailing whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    # Replace hyphens and slashes with spaces to avoid merging compound agricultural words
    text = re.sub(r"[\-/]", " ", text)
    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_meaningful_query(text: str, min_words: int = 2, min_chars: int = 4) -> bool:
    """
    Validates whether a user query contains sufficient content for meaningful classification.
    Returns False for empty strings, single random characters, or pure numbers.
    """
    cleaned = clean_text(text)
    if not cleaned or len(cleaned) < min_chars:
        return False

    tokens = cleaned.split()
    # Check if there are only numbers
    has_letters = any(any(c.isalpha() for c in token) for token in tokens)
    if not has_letters:
        return False

    return len(tokens) >= min_words
