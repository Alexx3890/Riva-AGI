"""Configuration and environment getters for RAG ingestion."""

import os
from pathlib import Path
from typing import List

from .. import load_env


def get_vision_api_key() -> str:
    """Retrieves dedicated vision API key, falling back to general GEMINI_API_KEY."""
    load_env()
    return os.getenv("GEMINI_VISION_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()


def get_vision_model() -> str:
    """Retrieves dedicated vision model from environment."""
    load_env()
    return (
        os.getenv("GEMINI_VISION_MODEL", "").strip()
        or os.getenv("GEMINI_RAG_MODEL", "").strip()
        or os.getenv("GEMINI_MODEL", "").strip()
        or os.getenv("GEMINI_DEFAULT_MODEL", "").strip()
    )


def get_vision_fallback_models() -> List[str]:
    """Retrieves fallback models for vision transcription."""
    load_env()
    raw = os.getenv("GEMINI_VISION_FALLBACK_MODELS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    from ..clients.gemini_client import get_fallback_gemini_models
    return get_fallback_gemini_models()


def is_cloud_vision_permitted(filepath: Path, allow_cloud_vision: bool = False) -> bool:
    """Checks whether cloud vision is permitted for this specific file or run."""
    if allow_cloud_vision:
        return True
    raw_env = os.getenv("ALLOW_CLOUD_VISION", "0").strip().lower()
    if raw_env in ("1", "true", "yes", "all"):
        allowlist = os.getenv("ALLOW_CLOUD_VISION_ALLOWLIST", "").strip()
        if not allowlist:
            return True
        allowed_tokens = [tok.strip().lower() for tok in allowlist.split(",") if tok.strip()]
        target_str = f"{filepath.parent.name}/{filepath.name}".lower()
        return any(tok in target_str for tok in allowed_tokens)
    return False
