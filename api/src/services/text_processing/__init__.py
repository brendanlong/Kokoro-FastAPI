"""Text processing pipeline."""

from .normalization import get_normalizer, normalize_text
from .text_processor import smart_split

__all__ = [
    "get_normalizer",
    "normalize_text",
    "smart_split",
]
