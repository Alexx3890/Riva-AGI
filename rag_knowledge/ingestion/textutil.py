"""Text processing and normalization utilities for RAG ingestion."""

import re
from typing import Optional


def normalize_whitespace(text: Optional[str]) -> str:
    """Collapses multiple spaces and strips whitespace."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def to_title_case(name: str) -> str:
    """Converts student/mentor names to clean Title Case."""
    clean = normalize_whitespace(name)
    if not clean:
        return ""
    parts = clean.split(" ")
    capitalized = []
    for p in parts:
        if p.upper() in {"I", "II", "III", "IV", "CSE", "CSIT", "IT", "ECE", "ME", "AIML", "AI"}:
            capitalized.append(p.upper())
        else:
            capitalized.append(p.capitalize())
    return " ".join(capitalized)


def clean_identifier(val: Optional[str]) -> str:
    """Sanitizes an ID or roll number (uppercase, alphanumeric stripped of spaces)."""
    if not val:
        return ""
    return re.sub(r"[^A-Za-z0-9]", "", str(val)).upper()


def slugify(text: str) -> str:
    """Converts a string into a clean lowercase slug suitable for document IDs."""
    if not text:
        return ""
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text).lower()).strip("_")
