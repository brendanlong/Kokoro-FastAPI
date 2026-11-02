"""Text to Kokoro phonemes, with one G2P per language shared by chunking and generation."""

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

from kokoro import KPipeline
from kokoro.foreign import CJK_PUNCTUATION
from misaki.en import MToken

from ..core.config import settings


@dataclass
class Phonemes:
    text: str
    phonemes: str
    # English only: the words with their phonemes, which generation turns into word timestamps
    tokens: Optional[List[MToken]] = None


_pipelines: Dict[str, KPipeline] = {}
_locks: Dict[str, threading.Lock] = {}
_creating = threading.Lock()


def pipeline(lang_code: str) -> KPipeline:
    """A model-free KPipeline: its G2P, plus generate_from_tokens when handed a model."""
    with _creating:
        if lang_code not in _pipelines:
            _pipelines[lang_code] = KPipeline(
                lang_code=lang_code, repo_id=settings.model_repo_id, model=False
            )
            _locks[lang_code] = threading.Lock()
        return _pipelines[lang_code]


def raw(text: str, lang_code: str):
    """The language's G2P output as is. Blocking; G2P state (spaCy, espeak) isn't shared across threads."""
    p = pipeline(lang_code)
    with _locks[lang_code]:
        return p.g2p(text)


def phonemize(text: str, lang_code: str) -> Phonemes:
    """Blocking; call off the event loop.

    Runs of other scripts (Chinese in English text, say) are read by their own language's G2P
    in this voice, rather than as symbol names like "Chinese letter".
    """
    p = pipeline(lang_code)
    graphemes = text if p.lang_code in "jz" else text.translate(CJK_PUNCTUATION)
    with _locks[lang_code]:
        if p.lang_code in "ab":
            tokens = p.en_g2p_tokens(graphemes)
        else:
            ps = p.g2p_phonemes(graphemes)
    if p.lang_code in "ab":
        for token in tokens:
            token.phonemes = token.phonemes or ""
        return Phonemes(text, KPipeline.tokens_to_ps(tokens), tokens)
    return Phonemes(text, (ps or "").strip())
