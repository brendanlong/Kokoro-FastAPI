import numpy as np
import pytest

from api.src.inference.stitching import (
    MAX_PIECE,
    MODEL_LIMIT,
    SAMPLES_PER_FRAME,
    _context,
    pieces,
    render_stitched,
)

BOS, EOS = 3, 5


def duration(char: str) -> int:
    return 2 if char == " " else 1 + ord(char) % 4


def fake_render(calls=None, limit=MODEL_LIMIT):
    """A model whose audio for any phonemes is that stretch of one continuous render.

    Each phoneme's samples depend only on the phoneme and its position within it, so a
    substring rendered with context matches the same stretch of the full render exactly.
    """

    def render(ps, row):
        assert len(ps) <= limit
        if calls is not None:
            calls.append((ps, row))
        durations = [BOS] + [duration(c) for c in ps] + [EOS]
        parts = [np.zeros(BOS * SAMPLES_PER_FRAME, np.float32)]
        for c in ps:
            n = duration(c) * SAMPLES_PER_FRAME
            parts.append((ord(c) % 7 + np.arange(n) % 11).astype(np.float32) / 20)
        parts.append(np.zeros(EOS * SAMPLES_PER_FRAME, np.float32))
        return np.concatenate(parts), np.array(durations)

    return render


def words(n: int, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    return " ".join(
        "".join(rng.choice(list("abcdefghij"), rng.integers(2, 9))) for _ in range(n)
    )


@pytest.mark.parametrize("n_words, max_len", [(10, 20), (120, 100), (400, 450)])
def test_pieces_cover_everything_but_the_seams_and_end_at_spaces(n_words, max_len):
    text = words(n_words)
    ranges = pieces(text, max_len)

    assert ranges[0][0] == 0 and ranges[-1][1] == len(text)
    assert all(end - start <= max_len for start, end in ranges)
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        assert text[end] == " " and start == end + 1


def test_pieces_cut_a_word_too_long_to_fit():
    assert pieces("a" * 25, 10) == [(0, 10), (10, 20), (20, 25)]


def stitch(text, max_len, render):
    """All yields joined, with each phoneme's times moved onto the joined audio."""
    audio, times, offset = [], {}, 0
    for piece, piece_times in render_stitched(text, max_len, render):
        times.update({i: (a + offset, b + offset) for i, (a, b) in piece_times.items()})
        audio.append(piece)
        offset += len(piece)
    return audio, np.concatenate(audio), times


@pytest.mark.parametrize(
    "text, max_len",
    [
        (words(10), 20),
        (words(120), 100),
        (words(400), 450),
        (words(400), 2000),  # capped so a piece and its context fit the model
        (f"ab cd {'x' * 150} ef gh", 100),  # a word longer than a piece is cut mid-word
        ("a" * 510 + " " + "b" * 510, 510),  # words as long as the model's whole input
    ],
)
def test_stitching_a_consistent_model_reproduces_one_continuous_render(text, max_len):
    """The cutting and crossfading add and drop nothing: every sample is the model's."""
    full_audio, full_durations = fake_render(limit=len(text))(text, 0)
    starts = np.cumsum(full_durations) * SAMPLES_PER_FRAME

    pieces_audio, audio, times = stitch(text, max_len, fake_render())

    assert len(pieces_audio) == len(pieces(text, min(max_len, MAX_PIECE)))
    assert len(audio) == len(full_audio)
    assert np.allclose(audio, full_audio)
    assert sorted(times) == list(range(len(text)))
    assert all(times[i] == (starts[i], starts[i + 1]) for i in range(len(text)))


def test_each_piece_sees_the_words_ahead_and_as_much_before_as_fits():
    text = words(300)
    calls = []
    list(render_stitched(text, 100, fake_render(calls)))

    assert len(calls) > 1
    rows = {row for _, row in calls}
    assert rows == {min(len(text), MODEL_LIMIT)}
    for (start, end), (ps, _) in zip(pieces(text, 100), calls):
        first, stop = _context(text, start, end)
        assert ps == text[first:stop]
        assert first <= max(start - 1, 0)
        assert stop == len(text) or text[end:stop].count(" ") >= 4
        assert len(ps) <= MODEL_LIMIT
