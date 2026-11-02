"""Tests for AudioService"""

import asyncio
import io
import threading
from unittest.mock import patch

import av
import numpy as np
import pytest

from api.src.inference.base import AudioChunk
from api.src.services.audio import AudioNormalizer, AudioService
from api.src.services.streaming_audio_writer import StreamingAudioWriter


@pytest.fixture(autouse=True)
def mock_settings():
    """Mock settings for all tests"""
    with patch("api.src.services.audio.settings") as mock_settings:
        mock_settings.gap_trim_ms = 250
        yield mock_settings


@pytest.fixture
def sample_audio():
    """Generate a simple sine wave for testing"""
    sample_rate = 24000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    return np.sin(2 * np.pi * frequency * t).astype(np.float32), sample_rate


def decode(blob: bytes, format: str) -> np.ndarray:
    if format == "pcm":
        return np.frombuffer(blob, np.int16)
    with av.open(io.BytesIO(blob)) as container:
        return np.concatenate(
            [f.to_ndarray().ravel() for f in container.decode(audio=0)]
        )


@pytest.mark.asyncio
async def test_convert_to_wav(sample_audio):
    """Test converting to WAV format"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("wav", sample_rate=24000)
    # Write and finalize in one step for WAV
    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "wav", writer, is_last_chunk=False
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # Check WAV header
    assert audio_chunk.output.startswith(b"RIFF")
    assert b"WAVE" in audio_chunk.output[:12]


@pytest.mark.asyncio
async def test_convert_to_mp3(sample_audio):
    """Test converting to MP3 format"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("mp3", sample_rate=24000)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "mp3", writer
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # Check MP3 header (ID3 or MPEG frame sync)
    assert audio_chunk.output.startswith(b"ID3") or audio_chunk.output.startswith(
        b"\xff\xfb"
    )


@pytest.mark.asyncio
async def test_convert_to_opus(sample_audio):
    """Test converting to Opus format"""

    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("opus", sample_rate=24000)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "opus", writer
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # Check OGG header
    assert audio_chunk.output.startswith(b"OggS")


@pytest.mark.asyncio
async def test_convert_to_flac(sample_audio):
    """Test converting to FLAC format"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("flac", sample_rate=24000)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "flac", writer
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # Check FLAC header
    assert audio_chunk.output.startswith(b"fLaC")


@pytest.mark.asyncio
async def test_convert_to_aac(sample_audio):
    """Test converting to M4A format"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("aac", sample_rate=24000)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "aac", writer
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # Check ADTS header (AAC)
    assert audio_chunk.output.startswith(b"\xff\xf0") or audio_chunk.output.startswith(
        b"\xff\xf1"
    )


@pytest.mark.asyncio
async def test_convert_to_pcm(sample_audio):
    """Test converting to PCM format"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter("pcm", sample_rate=24000)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "pcm", writer
    )

    writer.close()

    assert isinstance(audio_chunk.output, bytes)
    assert isinstance(audio_chunk, AudioChunk)
    assert len(audio_chunk.output) > 0
    # PCM is raw bytes, so no header to check


@pytest.mark.asyncio
async def test_convert_to_invalid_format_raises_error(sample_audio):
    """Test that converting to an invalid format raises an error"""
    # audio_data, sample_rate = sample_audio
    with pytest.raises(ValueError, match="Unsupported format: invalid"):
        StreamingAudioWriter("invalid", sample_rate=24000)


@pytest.mark.asyncio
@pytest.mark.parametrize("format", ["wav", "pcm"])
async def test_normalization_clips_to_int16(sample_audio, format):
    """Samples outside the int16 range land on the rails instead of wrapping"""
    audio_data, sample_rate = sample_audio

    writer = StreamingAudioWriter(format, sample_rate=sample_rate)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data * 1e5), format, writer, is_last_chunk=True
    )

    samples = decode(audio_chunk.output, format)
    assert samples.max() == 32767
    assert samples.min() == -32768


@pytest.mark.asyncio
async def test_empty_audio_writes_nothing():
    """An empty chunk produces no bytes and no error"""
    writer = StreamingAudioWriter("wav", sample_rate=24000)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(np.array([], dtype=np.float32)), "wav", writer
    )

    writer.close()

    assert audio_chunk.output == b""


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [8000, 16000, 44100, 48000])
async def test_different_sample_rates(sample_audio, rate):
    """The writer's sample rate is the one in the file header"""
    audio_data, _ = sample_audio

    writer = StreamingAudioWriter("wav", sample_rate=rate)

    audio_chunk = await AudioService.convert_audio(
        AudioChunk(audio_data), "wav", writer, is_last_chunk=True
    )

    with av.open(io.BytesIO(audio_chunk.output)) as container:
        assert container.streams.audio[0].rate == rate


@pytest.mark.parametrize("format", ["wav", "flac", "mp3", "opus", "aac"])
def test_finalize_preserves_tail(format):
    """Test that no audio is lost across chunks or at finalize (issue #497)"""
    sample_rate = 16000
    t = np.arange(int(0.2 * sample_rate)) / sample_rate
    audio = (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16)

    writer = StreamingAudioWriter(format, sample_rate=sample_rate)
    blob = (
        writer.write_chunk(audio)
        + writer.write_chunk(audio)
        + writer.write_chunk(finalize=True)
    )
    decoded = len(decode(blob, format))

    if format in ("wav", "flac"):
        assert decoded == 2 * len(audio)
    else:
        # lossy encoders pad with encoder delay, but must not truncate
        assert decoded >= 2 * len(audio)


def test_trim_keeps_the_sound_and_pads_its_edges(mock_settings):
    mock_settings.dynamic_gap_trim_padding_ms = 410
    mock_settings.dynamic_gap_trim_padding_char_multiplier = {".": 1}
    rate = 24000
    tone = (np.sin(np.arange(rate // 5) / 5) * 20000).astype(np.int16)
    silence = np.zeros(rate // 2, np.int16)
    audio = np.concatenate([silence, tone, silence])

    normalizer = AudioNormalizer()
    start, end = normalizer.find_first_last_non_silent(audio, "Hi.", 1.0)

    loud = np.flatnonzero(np.abs(audio.astype(np.int32)) > 32767 * 10 ** (-45 / 20))
    assert start == loud[0] - normalizer.samples_to_pad_start
    assert end == loud[-1] + int(0.41 * rate) - normalizer.samples_to_pad_start


def test_trim_leaves_all_silent_audio_alone(mock_settings):
    mock_settings.dynamic_gap_trim_padding_ms = 410
    mock_settings.dynamic_gap_trim_padding_char_multiplier = {}
    audio = np.zeros(4800, np.int16)
    assert AudioNormalizer().find_first_last_non_silent(audio, "", 1.0) == (0, 4800)


def _blocking_writer(sample_rate):
    """A writer whose write_chunk parks in its thread until released, recording when it finishes."""
    writer = StreamingAudioWriter("pcm", sample_rate=sample_rate)
    started, release, events = threading.Event(), threading.Event(), []

    def write_chunk(*args, **kwargs):
        started.set()
        release.wait(5)
        events.append("write finished")
        return b"x"

    writer.write_chunk = write_chunk
    return writer, started, release, events


@pytest.mark.asyncio
async def test_convert_audio_does_not_block_the_event_loop(sample_audio):
    """Encoding runs in a thread, so other requests keep being served meanwhile."""
    audio_data, sample_rate = sample_audio
    writer, started, release, _ = _blocking_writer(sample_rate)

    task = asyncio.create_task(
        AudioService.convert_audio(
            AudioChunk(audio_data), "pcm", writer, trim_audio=False
        )
    )
    await asyncio.to_thread(started.wait, 5)
    # The encode is parked in its thread; this coroutine still gets to run.
    assert not task.done()
    release.set()
    assert (await task).output == b"x"


@pytest.mark.asyncio
async def test_cancelled_convert_audio_waits_for_its_encode(sample_audio):
    """A disconnect cancels the request, which then closes the writer: that must not happen mid-write."""
    audio_data, sample_rate = sample_audio
    writer, started, release, events = _blocking_writer(sample_rate)

    async def request():
        try:
            await AudioService.convert_audio(
                AudioChunk(audio_data), "pcm", writer, trim_audio=False
            )
        finally:
            events.append("writer closed")

    task = asyncio.create_task(request())
    await asyncio.to_thread(started.wait, 5)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events == ["write finished", "writer closed"]


def test_trim_handles_the_most_negative_sample(mock_settings):
    mock_settings.dynamic_gap_trim_padding_ms = 0
    mock_settings.dynamic_gap_trim_padding_char_multiplier = {}
    for dtype in (np.int16, np.int32):
        audio = np.zeros(4800, dtype)
        audio[2400] = np.iinfo(dtype).min
        normalizer = AudioNormalizer()
        normalizer.samples_to_pad_start = 0
        assert normalizer.find_first_last_non_silent(
            audio, "", 1.0, is_last_chunk=True
        ) == (2400, 2400)
