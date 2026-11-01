from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, patch

import pytest
import torch

from api.src.core.config import settings
from api.src.inference.g2p import Phonemes
from api.src.inference.kokoro_v1 import KokoroV1, _configure_rocm_backend

TEST = Phonemes("test", "tˈɛst")


def _inputs(method):
    """What each generation method takes: a phonemized chunk, or raw phonemes."""
    return TEST if method == "generate" else TEST.phonemes


def _pipeline(run):
    return MagicMock(generate_from_tokens=MagicMock(side_effect=run))


@pytest.fixture
def kokoro_backend():
    """Create a KokoroV1 instance for testing."""
    return KokoroV1()


def test_initial_state(kokoro_backend):
    """Test initial state of KokoroV1."""
    assert not kokoro_backend.is_loaded
    assert kokoro_backend._model is None
    # Device should be set based on settings
    assert kokoro_backend.device in ["cuda", "cpu", "mps"]


@patch("torch.cuda.is_available", return_value=True)
@patch("torch.cuda.memory_allocated", return_value=5e9)
def test_memory_management(mock_memory, mock_cuda, kokoro_backend):
    """Test GPU memory management functions."""
    # Patch backend so it thinks we have cuda
    with patch.object(kokoro_backend, "_device", "cuda"):
        # Test memory check
        with patch("api.src.inference.kokoro_v1.model_config") as mock_config:
            mock_config.pytorch_gpu.memory_threshold = 4
            assert kokoro_backend._check_memory() == True

            mock_config.pytorch_gpu.memory_threshold = 6
            assert kokoro_backend._check_memory() == False


@patch("torch.mps.driver_allocated_memory", return_value=5e9)
def test_mps_memory_management(mock_memory, kokoro_backend):
    with patch.object(kokoro_backend, "_device", "mps"):
        with patch("api.src.inference.kokoro_v1.model_config") as mock_config:
            mock_config.pytorch_gpu.memory_threshold = 4
            assert kokoro_backend._check_memory()

            mock_config.pytorch_gpu.memory_threshold = 6
            assert not kokoro_backend._check_memory()


@pytest.mark.parametrize("method", ["generate", "generate_from_tokens"])
async def test_generate_clears_mps_memory_over_threshold(method, kokoro_backend):
    kokoro_backend._model = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.generate_from_tokens.return_value = iter([])
    with (
        patch.object(kokoro_backend, "_device", "mps"),
        patch.object(kokoro_backend, "_check_memory", return_value=True),
        patch.object(kokoro_backend, "_clear_memory") as mock_clear,
        patch("api.src.core.paths.load_voice_tensor", return_value=torch.ones(1)),
        patch("api.src.core.paths.save_voice_tensor"),
        patch("api.src.inference.g2p.pipeline", return_value=mock_pipeline),
    ):
        async for _ in getattr(kokoro_backend, method)(_inputs(method), "af_voice"):
            pass
    mock_clear.assert_called_once()


async def test_voice_cache_evicts_least_recently_used(kokoro_backend, monkeypatch):
    monkeypatch.setattr("api.src.inference.kokoro_v1.VOICE_CACHE_SIZE", 2)
    with patch("api.src.core.paths.load_voice_tensor", return_value=torch.ones(1)):
        await kokoro_backend._get_voice_tensor("a.pt")
        await kokoro_backend._get_voice_tensor("b.pt")
        await kokoro_backend._get_voice_tensor("a.pt")
        await kokoro_backend._get_voice_tensor("c.pt")
    assert [k.split(":")[0] for k in kokoro_backend._voice_cache] == ["a.pt", "c.pt"]


async def test_generation_does_not_block_event_loop(kokoro_backend):
    """Model calls run in a worker thread, so other requests and /health keep being served."""
    import asyncio
    import time

    kokoro_backend._model = MagicMock()

    def slow_pipeline(*args, **kwargs):
        for _ in range(3):
            time.sleep(0.1)
            yield SimpleNamespace(audio=torch.zeros(4), tokens=None, phonemes=None)

    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    with (
        patch("api.src.core.paths.load_voice_tensor", return_value=torch.ones(1)),
        patch("api.src.core.paths.save_voice_tensor"),
        patch(
            "api.src.inference.g2p.pipeline",
            return_value=_pipeline(slow_pipeline),
        ),
    ):
        task = asyncio.create_task(ticker())
        chunks = [c async for c in kokoro_backend.generate(TEST, "af_voice")]
        task.cancel()
    assert len(chunks) == 3
    assert ticks > 10


async def test_concurrent_generations_take_turns_on_the_model(kokoro_backend):
    import asyncio
    import threading
    import time

    kokoro_backend._model = MagicMock()
    active = 0
    peak = 0
    guard = threading.Lock()

    def pipeline(*args, **kwargs):
        nonlocal active, peak
        for _ in range(3):
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with guard:
                active -= 1
            yield SimpleNamespace(audio=torch.zeros(4), tokens=None, phonemes=None)

    async def run():
        return [c async for c in kokoro_backend.generate(TEST, "af_voice")]

    with (
        patch("api.src.core.paths.load_voice_tensor", return_value=torch.ones(1)),
        patch("api.src.core.paths.save_voice_tensor"),
        patch(
            "api.src.inference.g2p.pipeline",
            return_value=_pipeline(pipeline),
        ),
    ):
        results = await asyncio.gather(run(), run(), run())
    assert [len(r) for r in results] == [3, 3, 3]
    assert peak == 1


def _oom_backend(kokoro_backend, outcomes):
    """Pipeline whose calls play out outcomes in order: an exception is raised, a list is iterated."""
    kokoro_backend._model = MagicMock()

    def run(*args, **kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return iter(outcome)

    return _pipeline(run)


def _audio_result():
    return SimpleNamespace(audio=torch.zeros(4), tokens=None, phonemes=None)


@pytest.mark.parametrize("method", ["generate", "generate_from_tokens"])
async def test_oom_retries_once_then_raises(method, kokoro_backend):
    outcomes = [RuntimeError("CUDA out of memory")] * 3
    mock_pipeline = _oom_backend(kokoro_backend, outcomes)
    with (
        patch.object(kokoro_backend, "_device", "cuda"),
        patch.object(kokoro_backend, "_check_memory", return_value=False),
        patch.object(kokoro_backend, "_clear_memory") as mock_clear,
        patch("api.src.core.paths.load_voice_tensor", return_value=torch.ones(1)),
        patch("api.src.core.paths.save_voice_tensor"),
        patch("api.src.inference.g2p.pipeline", return_value=mock_pipeline),
    ):
        with pytest.raises(RuntimeError, match="out of memory"):
            async for _ in getattr(kokoro_backend, method)(_inputs(method), "af_voice"):
                pass
    assert len(outcomes) == 1
    mock_clear.assert_called_once()


@pytest.mark.parametrize("method", ["generate", "generate_from_tokens"])
async def test_oom_retry_success_does_not_raise(method, kokoro_backend):
    outcomes = [RuntimeError("CUDA out of memory"), [_audio_result()]]
    mock_pipeline = _oom_backend(kokoro_backend, outcomes)
    with (
        patch.object(kokoro_backend, "_device", "cuda"),
        patch.object(kokoro_backend, "_check_memory", return_value=False),
        patch.object(kokoro_backend, "_clear_memory"),
        patch("api.src.core.paths.load_voice_tensor", return_value=torch.ones(1)),
        patch("api.src.core.paths.save_voice_tensor"),
        patch("api.src.inference.g2p.pipeline", return_value=mock_pipeline),
    ):
        chunks = [
            c
            async for c in getattr(kokoro_backend, method)(_inputs(method), "af_voice")
        ]
    assert len(chunks) == 1


async def test_oom_after_audio_is_not_retried(kokoro_backend):
    def run(*args, **kwargs):
        yield _audio_result()
        raise RuntimeError("CUDA out of memory")

    kokoro_backend._model = MagicMock()
    mock_pipeline = _pipeline(run)
    with (
        patch.object(kokoro_backend, "_device", "cuda"),
        patch.object(kokoro_backend, "_check_memory", return_value=False),
        patch.object(kokoro_backend, "_clear_memory") as mock_clear,
        patch("api.src.core.paths.load_voice_tensor", return_value=torch.ones(1)),
        patch("api.src.core.paths.save_voice_tensor"),
        patch("api.src.inference.g2p.pipeline", return_value=mock_pipeline),
    ):
        chunks = []
        with pytest.raises(RuntimeError, match="out of memory"):
            async for chunk in kokoro_backend.generate(TEST, "af_voice"):
                chunks.append(chunk)
    assert len(chunks) == 1
    assert mock_pipeline.generate_from_tokens.call_count == 1
    mock_clear.assert_not_called()


@patch("torch.cuda.empty_cache")
@patch("torch.cuda.synchronize")
def test_clear_memory(mock_sync, mock_clear, kokoro_backend):
    """Test memory clearing."""
    with patch.object(kokoro_backend, "_device", "cuda"):
        kokoro_backend._clear_memory()
        mock_clear.assert_called_once()
        mock_sync.assert_called_once()


def test_unload_releases_model_and_voices(kokoro_backend):
    kokoro_backend._model = MagicMock()
    kokoro_backend._voice_cache = {"af_heart.pt:cpu": MagicMock()}
    assert kokoro_backend.is_loaded

    # Test unload
    kokoro_backend.unload()
    assert not kokoro_backend.is_loaded
    assert kokoro_backend._model is None
    assert kokoro_backend._voice_cache == {}  # Voice tensors should be released


@pytest.mark.asyncio
async def test_generate_validation(kokoro_backend):
    """Test generation validation."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        async for _ in kokoro_backend.generate(TEST, "voice"):
            pass


@pytest.mark.asyncio
async def test_generate_from_tokens_validation(kokoro_backend):
    """Test token generation validation."""
    with pytest.raises(RuntimeError, match="Model not loaded"):
        async for _ in kokoro_backend.generate_from_tokens("test tokens", "voice"):
            pass


def test_get_pipeline_is_the_shared_g2p_pipeline(kokoro_backend):
    """Generation uses the same per-language pipeline chunking phonemized with."""
    kokoro_backend._model = MagicMock()
    mock_pipeline = MagicMock()
    with patch("api.src.inference.g2p.pipeline", return_value=mock_pipeline) as factory:
        assert kokoro_backend._get_pipeline("e") is mock_pipeline
    factory.assert_called_once_with("e")


@pytest.mark.asyncio
async def test_generate_uses_correct_pipeline(kokoro_backend):
    """Test that generate uses correct pipeline for language code."""
    # Mock loaded state
    kokoro_backend._model = MagicMock()

    # Mock voice path handling
    with (
        patch("api.src.core.paths.load_voice_tensor") as mock_load_voice,
        patch("api.src.core.paths.save_voice_tensor"),
    ):
        mock_load_voice.return_value = torch.ones(1)

        mock_pipeline = MagicMock()
        mock_pipeline.generate_from_tokens.return_value = iter([])
        with patch(
            "api.src.inference.g2p.pipeline", return_value=mock_pipeline
        ) as factory:
            async for _ in kokoro_backend.generate(TEST, "ef_voice", lang_code="e"):
                pass

            factory.assert_called_with("e")
            mock_pipeline.generate_from_tokens.assert_called_with(
                tokens=TEST.phonemes,
                voice=ANY,
                speed=1.0,
                model=kokoro_backend._model,
            )
            # The cached tensor goes to the pipeline directly, so nothing is
            # staged in the temp dir or kept in pipeline.voices
            voice = mock_pipeline.generate_from_tokens.call_args[1]["voice"]
            assert torch.equal(voice, torch.ones(1))


@pytest.mark.asyncio
async def test_generate_skips_untimed_tokens(kokoro_backend):
    """A token misaki gave no phonemes has no start_ts; later words keep their timestamps (issue #249)."""
    kokoro_backend._model = MagicMock()
    tokens = [
        SimpleNamespace(text="What", start_ts=0.0, end_ts=0.2),
        SimpleNamespace(text="--", start_ts=None, end_ts=None),
        SimpleNamespace(text=" ", start_ts=0.2, end_ts=0.3),
        SimpleNamespace(text="key", start_ts=0.3, end_ts=0.5),
        SimpleNamespace(text="tail"),
    ]
    result = SimpleNamespace(
        audio=torch.zeros(24000), tokens=tokens, pred_dur=torch.ones(8), phonemes=""
    )

    with (
        patch("api.src.core.paths.load_voice_tensor", return_value=torch.ones(1)),
        patch("api.src.core.paths.save_voice_tensor"),
    ):
        mock_pipeline = MagicMock()
        mock_pipeline.generate_from_tokens.return_value = iter([result])
        with patch("api.src.inference.g2p.pipeline", return_value=mock_pipeline):
            chunks = [
                c
                async for c in kokoro_backend.generate(
                    Phonemes("What--key", "wʌt kˈi", tokens),
                    "af_voice",
                    lang_code="a",
                    return_timestamps=True,
                )
            ]

    assert [t.word for t in chunks[0].word_timestamps] == ["What", "key"]


def test_espeak_word_timestamps_basic():
    """Word times derived from pred_dur for an espeak (non-English) result."""
    from api.src.inference.kokoro_v1 import _espeak_word_timestamps

    # phonemes "ab cd": BOS=8, a=4, b=4, space=8, c=4, d=4, EOS=0
    # scale is 2/80 s per unit -> BOS 0.2s, each char 0.1s, space 0.2s
    pred_dur = torch.tensor([8, 4, 4, 8, 4, 4, 0])
    ts = _espeak_word_timestamps("hola mundo", "ab cd", pred_dur)

    assert [t.word for t in ts] == ["hola", "mundo"]
    assert ts[0].start_time == pytest.approx(0.2)
    assert ts[0].end_time == pytest.approx(0.4)
    assert ts[1].start_time == pytest.approx(0.6)
    assert ts[1].end_time == pytest.approx(0.8)


def test_espeak_word_timestamps_expansion():
    """Words espeak expands (numbers) are merged back via per-word g2p."""
    from api.src.inference.kokoro_v1 import _espeak_word_timestamps

    # Three text words but four phoneme groups ("1863" speaks as two words).
    phonemes = "a bb cc d"
    pred_dur = torch.tensor([0, 4] + [8, 4, 4] + [8, 4, 4] + [8, 4] + [0])

    def g2p(word):
        return ("bb cc" if word == "1863" else "x", None)

    ts = _espeak_word_timestamps("en 1863 el", phonemes, pred_dur, g2p=g2p)

    assert [t.word for t in ts] == ["en", "1863", "el"]
    # "1863" spans both expanded groups.
    assert ts[1].start_time == pytest.approx(0.3)
    assert ts[1].end_time == pytest.approx(0.9)
    # Whole-string timing stays monotonic and inside the clip.
    assert ts[0].end_time <= ts[1].start_time <= ts[2].start_time


def test_espeak_word_timestamps_unreconcilable():
    """Group/word count mismatch that g2p can't explain yields None."""
    from api.src.inference.kokoro_v1 import _espeak_word_timestamps

    phonemes = "a b c"
    pred_dur = torch.tensor([0, 4, 8, 4, 8, 4, 0])

    # g2p says every word is a single group: 2 != 3 -> give up.
    ts = _espeak_word_timestamps("uno dos", phonemes, pred_dur, g2p=lambda w: (w, None))
    assert ts is None

    # No g2p available -> also None.
    assert _espeak_word_timestamps("uno dos", phonemes, pred_dur) is None


def test_espeak_word_timestamps_degenerate_inputs():
    """Missing durations or empty text return None instead of raising."""
    from api.src.inference.kokoro_v1 import _espeak_word_timestamps

    assert _espeak_word_timestamps("hola", "abc", None) is None
    assert _espeak_word_timestamps("hola", "", torch.tensor([0, 0])) is None
    assert _espeak_word_timestamps("", "abc", torch.tensor([0, 4, 4, 4, 0])) is None
    # pred_dur too short for the phoneme string.
    assert _espeak_word_timestamps("hola", "abcdef", torch.tensor([0, 4])) is None


@pytest.fixture
def restore_cudnn():
    """Restore the global flag that _configure_rocm_backend mutates."""
    previous = torch.backends.cudnn.enabled
    yield
    torch.backends.cudnn.enabled = previous


def test_configure_rocm_backend_noop_on_cuda(restore_cudnn, monkeypatch):
    """A CUDA build must keep cuDNN enabled."""
    monkeypatch.setattr(torch.version, "hip", None)
    torch.backends.cudnn.enabled = True
    _configure_rocm_backend()
    assert torch.backends.cudnn.enabled is True


def test_configure_rocm_backend_disables_miopen_on_rocm(restore_cudnn, monkeypatch):
    """A ROCm build disables MIOpen to avoid per-shape kernel compilation."""
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(settings, "enable_miopen", False)
    torch.backends.cudnn.enabled = True
    _configure_rocm_backend()
    assert torch.backends.cudnn.enabled is False


def test_configure_rocm_backend_opt_out(restore_cudnn, monkeypatch):
    """ENABLE_MIOPEN keeps MIOpen active."""
    monkeypatch.setattr(torch.version, "hip", "7.2.0")
    monkeypatch.setattr(settings, "enable_miopen", True)
    torch.backends.cudnn.enabled = True
    _configure_rocm_backend()
    assert torch.backends.cudnn.enabled is True


@pytest.mark.parametrize(
    ("phonemes", "sent"),
    [
        (Phonemes("Hi.", "hˈI.", ["token"]), ["token"]),
        (Phonemes("hola", "ˈola"), "ˈola"),
    ],
)
async def test_generate_feeds_the_chunks_own_phonemes(phonemes, sent, kokoro_backend):
    """English sends its words (for timestamps); other languages the phoneme string. No G2P again."""
    kokoro_backend._model = MagicMock()
    mock_pipeline = MagicMock()
    mock_pipeline.generate_from_tokens.return_value = iter([])
    with (
        patch("api.src.core.paths.load_voice_tensor", return_value=torch.ones(1)),
        patch("api.src.core.paths.save_voice_tensor"),
        patch("api.src.inference.g2p.pipeline", return_value=mock_pipeline),
    ):
        async for _ in kokoro_backend.generate(phonemes, "af_voice"):
            pass
    assert mock_pipeline.generate_from_tokens.call_args.kwargs["tokens"] == sent
    mock_pipeline.g2p.assert_not_called()


class FakeModel:
    """Each phoneme lasts two frames and sounds like its own constant; anything outside the
    vocabulary is dropped, as KModel does."""

    vocab = {c: i for i, c in enumerate(" .abcdefghijklmnopqrstuvwxyzˈ")}

    def __init__(self):
        self.calls = []

    def __call__(self, ps, ref_s, speed, return_output=False):
        self.calls.append(ps)
        kept = [c for c in ps if c in self.vocab]
        dur = torch.tensor([2] * (len(kept) + 2))
        audio = torch.cat(
            [torch.zeros(1200)]
            + [torch.full((1200,), self.vocab[c] / 40) for c in kept]
            + [torch.zeros(1200)]
        )
        return SimpleNamespace(audio=audio, pred_dur=dur)


@pytest.mark.parametrize("dropped", ["", "¿"])
async def test_over_long_chunk_streams_in_pieces_with_word_timestamps(
    dropped, kokoro_backend, monkeypatch
):
    monkeypatch.setattr(settings, "absolute_max_tokens", 40)
    kokoro_backend._model = FakeModel()
    words = [f"word{i}" for i in range(30)]
    phonemes = Phonemes(
        " ".join(words),
        dropped + " ".join("wˈord" + "abcdefgh"[i % 8] for i in range(30)),
    )

    with (
        patch(
            "api.src.core.paths.load_voice_tensor", return_value=torch.ones(510, 256)
        ),
        patch("api.src.core.paths.save_voice_tensor"),
        patch("api.src.inference.g2p.pipeline"),
    ):
        chunks = [
            c
            async for c in kokoro_backend.generate(
                phonemes, "af_voice", lang_code="e", return_timestamps=True
            )
        ]

    assert len(chunks) == len(kokoro_backend._model.calls) > 1
    stamps, offset = [], 0.0
    for chunk in chunks:
        stamps += [(t.word, t.start_time + offset) for t in chunk.word_timestamps]
        offset += len(chunk.audio) / 24000
    assert [word for word, _ in stamps] == words
    assert all(a < b for (_, a), (_, b) in zip(stamps, stamps[1:]))
    assert offset == pytest.approx(
        len(phonemes.phonemes.lstrip("¿")) * 2 * 600 / 24000 + 0.1
    )


def test_word_spans_point_at_each_words_phonemes():
    from api.src.inference import g2p
    from api.src.inference.kokoro_v1 import _word_spans

    phonemes = g2p.phonemize("Hello there, wide world.", "a")
    spans = _word_spans(phonemes)

    assert [w for w, _, _ in spans] == ["Hello", "there", ",", "wide", "world", "."]
    tokens = [t for t in phonemes.tokens if t.phonemes]
    assert [phonemes.phonemes[a : b + 1] for _, a, b in spans] == [
        t.phonemes for t in tokens
    ]
