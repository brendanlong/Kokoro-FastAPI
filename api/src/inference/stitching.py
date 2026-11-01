"""Render phonemes too long for one model call as overlapping pieces, keeping each piece's own audio.

A piece rendered alone ends like a finished utterance, which sounds like a pause mid-phrase. So each
piece is rendered with the words around it, and only its own stretch is kept: located with the
model's per-phoneme durations, cut mid-way through the space between two words, and crossfaded.
"""

from typing import Callable, Dict, Iterator, List, Tuple

import numpy as np

MODEL_LIMIT = 510  # phonemes the model takes in one call
SAMPLES_PER_FRAME = 600  # one predicted-duration unit at 24 kHz
CROSSFADE = 480  # 20 ms
WORDS_AHEAD = 4
MAX_PIECE = MODEL_LIMIT - 60  # room for the seam before a piece and the words after it

# (phonemes, style row) -> (audio, per-phoneme durations with the start and end tokens' at either end)
Render = Callable[[str, int], Tuple[np.ndarray, np.ndarray]]
# phoneme index -> [start, end) samples within the audio yielded with it
Times = Dict[int, Tuple[int, int]]


def pieces(phonemes: str, max_len: int) -> List[Tuple[int, int]]:
    """[start, end) ranges of at most max_len phonemes, ending at spaces where possible.

    The space between two pieces is in neither: it's the seam.
    """
    out, start = [], 0
    while start < len(phonemes):
        end = start + max_len
        if end >= len(phonemes):
            end = len(phonemes)
        else:
            space = phonemes.rfind(" ", start + 1, end + 1)
            end = space if space > start else end
        out.append((start, end))
        start = end + 1 if end < len(phonemes) and phonemes[end] == " " else end
    return out


def _context(phonemes: str, start: int, end: int) -> Tuple[int, int]:
    stop = end
    for _ in range(WORDS_AHEAD + 1):
        if stop >= len(phonemes):
            stop = len(phonemes)
            break
        nxt = phonemes.find(" ", stop + 1)
        stop = len(phonemes) if nxt == -1 else nxt
    first = max(0, stop - MODEL_LIMIT)
    if first > 0:
        word = phonemes.find(" ", first)
        first = word + 1 if 0 <= word < start else first
    # the seam before a piece is cut inside its context, so the context must reach it
    first = min(first, max(start - 1, 0))
    return first, min(stop, first + MODEL_LIMIT)


def render_stitched(
    phonemes: str, max_len: int, render: Render
) -> Iterator[Tuple[np.ndarray, Times]]:
    """Yield each piece's audio as soon as it's rendered, with the times of the phonemes it covers.

    The yields concatenate into one continuous rendering; only a crossfade's worth of each piece is
    held back, to blend with the next.
    """
    row = min(len(phonemes), MODEL_LIMIT)
    ranges = pieces(phonemes, min(max_len, MAX_PIECE))
    held = None
    for index, (start, end) in enumerate(ranges):
        first, stop = _context(phonemes, start, end)
        audio, dur = render(phonemes[first:stop], row)
        # frames[1 + i] is where phoneme first + i starts in audio
        frames = np.concatenate([[0], np.cumsum(dur)]) * SAMPLES_PER_FRAME

        def cut(boundary):
            """Sample where the seam before phoneme `boundary` falls: mid-space, or between phonemes."""
            if phonemes[boundary - 1] == " ":
                i = boundary - 1 - first
                return int((frames[1 + i] + frames[2 + i]) // 2)
            return int(frames[1 + boundary - first])

        last = index == len(ranges) - 1
        begin = 0 if index == 0 else max(cut(start) - CROSSFADE // 2, 0)
        finish = len(audio) if last else cut(ranges[index + 1][0]) - CROSSFADE // 2
        out = np.array(audio[begin:finish], dtype=np.float32)
        if held is not None:
            n = min(len(held), len(out))
            ramp = np.linspace(0, 1, n, dtype=np.float32)
            out[:n] = held[:n] * (1 - ramp) + out[:n] * ramp
        held = None if last else audio[finish : finish + CROSSFADE]

        # a seam's space belongs to the piece before it
        owned = stop if last else ranges[index + 1][0]
        times = {
            first + i: (int(frames[1 + i]) - begin, int(frames[2 + i]) - begin)
            for i in range(start - first, owned - first)
        }
        yield out, times
