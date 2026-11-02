import time

import pytest

from api.src.inference.g2p import Phonemes
from api.src.services.text_processing import text_processor
from api.src.services.text_processing.normalization import Normalizer
from api.src.services.text_processing.text_processor import (
    check_speakable,
    join_lines,
    smart_split,
    split_by_voice,
    strip_emoji,
)
from api.src.structures.schemas import NormalizationOptions


class ShoutingNormalizer(Normalizer):
    lang_codes = ("xx",)

    def numbers(self, text: str) -> str:
        return text.upper()


def fake_phonemize(text: str, lang_code: str) -> Phonemes:
    """One phoneme per character, so chunk sizes are easy to reason about."""
    return Phonemes(text, text)


@pytest.fixture
def fake_g2p(monkeypatch):
    calls = []

    def phonemize(text, lang_code):
        calls.append(text)
        return fake_phonemize(text, lang_code)

    monkeypatch.setattr(text_processor, "phonemize", phonemize)
    return calls


@pytest.fixture
def shouting_language(monkeypatch, fake_g2p):
    registry = {"xx": ShoutingNormalizer()}
    monkeypatch.setattr(text_processor, "get_normalizer", registry.get)
    monkeypatch.setattr(text_processor.settings, "advanced_text_normalization", True)


async def first_chunk_text(**kwargs) -> str:
    async for chunk_text, _, _ in smart_split("hello world", **kwargs):
        return chunk_text


@pytest.mark.asyncio
async def test_smart_split_normalizes_with_the_registered_language(shouting_language):
    assert await first_chunk_text(lang_code="xx") == "HELLO WORLD"


@pytest.mark.asyncio
async def test_smart_split_skips_unregistered_language(shouting_language):
    assert await first_chunk_text(lang_code="yy") == "hello world"


@pytest.mark.asyncio
async def test_smart_split_honors_request_normalize_switch(shouting_language):
    options = NormalizationOptions(normalize=False)
    assert (
        await first_chunk_text(lang_code="xx", normalization_options=options)
        == "hello world"
    )


@pytest.mark.asyncio
async def test_smart_split_honors_global_normalization_switch(
    shouting_language, monkeypatch
):
    monkeypatch.setattr(text_processor.settings, "advanced_text_normalization", False)
    assert await first_chunk_text(lang_code="xx") == "hello world"


@pytest.mark.asyncio
async def test_smart_split_keeps_custom_phonemes_out_of_the_normalizer(
    shouting_language,
):
    text = "hello [Kokoro](/kˈOkəɹO/) world"
    async for chunk_text, _, _ in smart_split(text, lang_code="xx"):
        assert chunk_text == "HELLO [Kokoro](/kˈOkəɹO/) WORLD"
        break


# issue #353
REMOVE_EMOJI = NormalizationOptions(remove_emoji=True)


@pytest.mark.asyncio
async def test_smart_split_keeps_emoji_by_default():
    async for chunk_text, _, _ in smart_split("hello 😊 world", lang_code="a"):
        assert "😊" in chunk_text


@pytest.mark.asyncio
@pytest.mark.parametrize("lang_code", ["a", "z"])
async def test_smart_split_remove_emoji_option(lang_code):
    async for chunk_text, _, _ in smart_split(
        "hello 😊 world", lang_code=lang_code, normalization_options=REMOVE_EMOJI
    ):
        assert chunk_text == "hello world"
    assert [
        chunk
        async for chunk in smart_split(
            "😊", lang_code=lang_code, normalization_options=REMOVE_EMOJI
        )
    ] == []


def test_check_speakable():
    check_speakable("hi")
    check_speakable("[pause:1s]")
    check_speakable("[voice:af_bella]")
    check_speakable("😊")
    for text in ["😊", "⏰ 1️⃣", "  ", "[voice:af_bella] [voice:bm_george]"]:
        with pytest.raises(ValueError, match="no speakable text"):
            check_speakable(
                text, allow_voice_tags=True, normalization_options=REMOVE_EMOJI
            )


def test_strip_emoji_sequences():
    england = "\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
    for emoji in ["😊", "⏰", "1️⃣", "👨‍👩‍👧", "🇺🇸", "👍🏽", england]:
        assert strip_emoji(f"a {emoji} b") == "a b"
    assert strip_emoji("call 1 or # now") == "call 1 or # now"


def test_strip_emoji_leaves_joiners_inside_words():
    conjunct = "क्‍ष"
    assert strip_emoji(f"a {conjunct} b") == f"a {conjunct} b"
    assert strip_emoji("a ‍ b") == "a ‍ b"


def test_strip_emoji_flood_is_fast():
    start = time.monotonic()
    strip_emoji("1\ufe0f" * 20_000 + "😊" * 20_000)
    assert time.monotonic() - start < 2.0


@pytest.mark.asyncio
async def test_smart_split_phonemizes_each_sentence_once(fake_g2p):
    chunks = [c async for c in smart_split("One here. Two here!", lang_code="a")]

    assert [(text, phonemes.phonemes) for text, phonemes, _ in chunks] == [
        ("One here.", "One here."),
        ("Two here!", "Two here!"),
    ]
    assert fake_g2p == ["One here.", "Two here!"]


@pytest.mark.asyncio
async def test_smart_split_uses_the_languages_own_phonemes():
    """English phonemes come from misaki, with the words kept for timestamps."""
    chunks = [c async for c in smart_split("Hello world.", lang_code="a")]

    assert len(chunks) == 1
    phonemes = chunks[0][1]
    assert phonemes.phonemes == "həlˈO wˈɜɹld."
    assert [t.text for t in phonemes.tokens] == ["Hello", "world", "."]


@pytest.mark.asyncio
async def test_smart_split_abbreviations_are_not_sentence_breaks(fake_g2p):
    """Abbreviations, decimals, and ellipses do not force bogus sentence breaks (issue #308, pr #520)."""
    text = (
        "This, that, the other thing, etc. Another sentence... A, b, c, etc., and "
        "more. D, e, f, etc. and more. One, i. e. two. Three, i. e., four. Five, "
        "i.e. six. You have 4.2 messages. Property access: `a.b.c`. Bring pencils, "
        "paper, erasers, etc. Then we can start the exam. Some fruits, e.g. apples, "
        "keep well. Data (Eastern Harbor vs. outer harbor) may vary. Keep that in mind."
    )

    sentences = [
        c
        async for c, _, _ in smart_split(
            text, normalization_options=NormalizationOptions(normalize=False)
        )
    ]

    assert sentences == [
        "This, that, the other thing, etc.",
        "Another sentence...",
        "A, b, c, etc., and more.",
        "D, e, f, etc. and more.",
        "One, i. e. two.",
        "Three, i. e., four.",
        "Five, i.e. six.",
        "You have 4.2 messages.",
        "Property access: `a.b.c`.",
        "Bring pencils, paper, erasers, etc.",
        "Then we can start the exam.",
        "Some fruits, e.g. apples, keep well.",
        "Data (Eastern Harbor vs. outer harbor) may vary.",
        "Keep that in mind.",
    ]


@pytest.mark.asyncio
async def test_smart_split_first_chunk_skips_rest_of_text(fake_g2p):
    """Time to first chunk stays flat: it must not phonemize the whole input (pr #513)."""
    text = " ".join(f"This is sentence {i}." for i in range(200))
    chunks = text_processor.smart_split(text)
    try:
        await anext(chunks)
    finally:
        await chunks.aclose()

    assert fake_g2p == ["This is sentence zero."]


@pytest.mark.asyncio
async def test_smart_split_short_text():
    """Test smart splitting with text under max tokens."""
    text = "This is a short test sentence."
    chunks = []
    async for chunk_text, phonemes, _ in smart_split(text):
        chunks.append((chunk_text, phonemes))

    assert len(chunks) == 1
    assert chunks[0][0] == text
    assert chunks[0][1].phonemes


@pytest.mark.asyncio
async def test_smart_custom_phenomes():
    """Custom phoneme tags survive splitting (issue #348, pr #350)."""
    text = "This is a short test sentence. [Kokoro](/kˈOkəɹO/) has a feature called custom phenomes. This is made possible by [Misaki](/misˈɑki/), the custom phenomizer that [Kokoro](/kˈOkəɹO/) version 1.0 uses"
    chunks = [c async for c in smart_split(text)]

    assert [c for c, _, _ in chunks] == [
        "This is a short test sentence.",
        "[Kokoro](/kˈOkəɹO/) has a feature called custom phenomes.",
        "This is made possible by [Misaki](/misˈɑki/), the custom phenomizer that [Kokoro](/kˈOkəɹO/) version one uses",
    ]
    assert chunks[1][1].phonemes.startswith("kˈOkəɹO ")
    assert "misˈɑki" in chunks[2][1].phonemes


@pytest.mark.asyncio
async def test_smart_split_only_phenomes():
    """Input made entirely of phoneme annotations stays one chunk (pr #350)."""
    text = "[Kokoro](/kˈOkəɹO/) [Misaki 1.2](/misˈɑki/) [Test](/tɛst/)"
    chunks = [c async for c in smart_split(text, max_tokens=10)]

    assert [c for c, _, _ in chunks] == [text]
    assert chunks[0][1].phonemes == "kˈOkəɹO misˈɑki tɛst"


@pytest.mark.asyncio
async def test_smart_split_long_text():
    """Test smart splitting with longer text."""
    # Create text that should split into multiple chunks
    text = ". ".join(["This is test sentence number " + str(i) for i in range(20)])

    chunks = [c async for c in smart_split(text)]

    assert len(chunks) == 20
    assert all(chunk_text and phonemes.phonemes for chunk_text, phonemes, _ in chunks)


@pytest.mark.asyncio
async def test_smart_split_with_punctuation():
    """Test smart splitting handles punctuation correctly."""
    text = "First sentence! Second sentence? Third sentence; Fourth sentence: Fifth sentence."

    chunks = []
    async for chunk_text, chunk_tokens, _ in smart_split(text):
        chunks.append(chunk_text)

    # Verify punctuation is preserved
    assert all(any(p in chunk for p in "!?;:.") for chunk in chunks)


@pytest.mark.asyncio
async def test_smart_split_chinese_sentences():
    """Chinese splits at its own sentence punctuation (pr #321)."""
    text = "这是一个句子。这是第二个句子！第三个问题？"
    chunks = [c async for c in smart_split(text, lang_code="z")]

    assert [c for c, _, _ in chunks] == [
        "这是一个句子。",
        "这是第二个句子！",
        "第三个问题？",
    ]
    assert all(phonemes.phonemes for _, phonemes, _ in chunks)


@pytest.mark.asyncio
async def test_smart_split_chinese_short():
    """Test Chinese smart splitting with short text (pr #321)."""
    text = "这是一句话。"
    chunks = [c async for c in smart_split(text, lang_code="z")]

    assert len(chunks) == 1
    assert chunks[0][0] == text
    assert chunks[0][1].phonemes


@pytest.mark.asyncio
async def test_smart_split_chinese_long():
    """Test Chinese smart splitting with longer text (pr #321)."""
    text = "。".join([f"测试句子 {i}" for i in range(20)])

    chunks = [c async for c in smart_split(text, lang_code="z")]

    assert len(chunks) == 20
    assert all(chunk_text and phonemes.phonemes for chunk_text, phonemes, _ in chunks)


@pytest.mark.asyncio
async def test_smart_split_chinese_punctuation():
    """Test Chinese smart splitting with punctuation preservation (pr #321)."""
    text = "第一句！第二问？第三句；第四句：第五句。"

    chunks = []
    async for chunk_text, _, _ in smart_split(text, lang_code="z"):
        chunks.append(chunk_text)

    # Verify Chinese punctuation is preserved
    assert all(any(p in chunk for p in "！？；：。") for chunk in chunks)


@pytest.mark.asyncio
async def test_smart_split_with_pause():
    """Test smart splitting with pause tags (pr #322)."""
    text = "Hello world [pause:2.5s] How are you?"

    chunks = []
    async for chunk_text, phonemes, pause_duration in smart_split(text):
        chunks.append((chunk_text, phonemes, pause_duration))

    # Should have 3 chunks: text, pause, text
    assert len(chunks) == 3

    # First chunk: text
    assert chunks[0][2] is None  # No pause
    assert "Hello world" in chunks[0][0]
    assert chunks[0][1].phonemes

    # Second chunk: pause
    assert chunks[1][2] == 2.5  # 2.5 second pause
    assert chunks[1][0] == ""  # Empty text
    assert chunks[1][1] is None

    # Third chunk: text
    assert chunks[2][2] is None  # No pause
    assert "How are you?" in chunks[2][0]
    assert chunks[2][1].phonemes


@pytest.mark.asyncio
async def test_smart_split_with_two_pause():
    """Test smart splitting with two pause tags."""
    text = "[pause:0.5s][pause:1.67s]0.5"

    chunks = []
    async for chunk_text, phonemes, pause_duration in smart_split(text):
        chunks.append((chunk_text, phonemes, pause_duration))

    # Should have 3 chunks: pause, pause, text
    assert len(chunks) == 3

    # First chunk: pause
    assert chunks[0][2] == 0.5  # 0.5 second pause
    assert chunks[0][0] == ""  # Empty text
    assert chunks[0][1] is None

    # Second chunk: pause
    assert chunks[1][2] == 1.67  # 1.67 second pause
    assert chunks[1][0] == ""  # Empty text
    assert chunks[1][1] is None

    # Third chunk: text
    assert chunks[2][2] is None  # No pause
    assert "zero point five" in chunks[2][0]
    assert chunks[2][1].phonemes


def test_split_by_voice_no_tags():
    """Text without voice tags is a single segment on the default voice."""
    segments = split_by_voice("Just plain text.", "af_heart")
    assert segments == [("af_heart", 1.0, "Just plain text.")]


def test_split_by_voice_multiple_speakers():
    """Each tag starts a new segment."""
    text = "[voice:af_bella] Hello there. [voice:am_michael] Hi back."
    segments = split_by_voice(text, "af_heart")

    assert segments == [
        ("af_bella", 1.0, "Hello there."),
        ("am_michael", 1.0, "Hi back."),
    ]


def test_split_by_voice_leading_text_uses_default():
    """Text ahead of the first tag belongs to the request voice."""
    text = "Narrator opens. [voice:af_bella] Then Bella speaks."
    segments = split_by_voice(text, "af_heart")

    assert segments == [
        ("af_heart", 1.0, "Narrator opens."),
        ("af_bella", 1.0, "Then Bella speaks."),
    ]


def test_split_by_voice_merges_repeated_voice():
    """A tag that doesn't change speaker shouldn't fragment chunking."""
    text = "[voice:af_bella] One. [voice:af_bella] Two."
    segments = split_by_voice(text, "af_heart")

    assert segments == [("af_bella", 1.0, "One.  Two.")]


def test_split_by_voice_merge_keeps_paragraph_break():
    """Merging same-voice runs keeps the blank line for join_lines."""
    text = "[voice:af_bella]Heading\n\n[voice:af_bella]Body."
    segments = split_by_voice(text, "af_heart")

    assert segments == [("af_bella", 1.0, "Heading\n\nBody.")]


def test_split_by_voice_merge_keeps_words_apart():
    """A no-op tag with no whitespace around it must not glue the words."""
    assert split_by_voice("Hello[voice:af_bella]world", "af_bella") == [
        ("af_bella", 1.0, "Hello world")
    ]


def test_split_by_voice_accepts_combined_voices():
    """Weighted combine syntax survives the tag round trip."""
    segments = split_by_voice("[voice: af_bella(2)+af_sky ] Mixed.", "af_heart")
    assert segments == [("af_bella(2)+af_sky", 1.0, "Mixed.")]


def test_split_by_voice_keeps_pause_tags():
    """Pause tags stay in the text for smart_split to handle."""
    text = "[voice:af_bella] One. [pause:0.5s] [voice:am_michael] Two."
    segments = split_by_voice(text, "af_heart")

    assert segments == [
        ("af_bella", 1.0, "One. [pause:0.5s]"),
        ("am_michael", 1.0, "Two."),
    ]


def test_split_by_voice_empty_text():
    """Empty input still yields one segment so callers keep their fallback."""
    assert split_by_voice("", "af_heart") == [("af_heart", 1.0, "")]


def test_split_by_voice_tags_only():
    """Tags with no speech yield nothing rather than reading the markup aloud."""
    assert split_by_voice("[voice:af_bella] [voice:am_michael]", "af_heart") == []


def test_split_by_rate_segments():
    """Rate tags open segments on the same voice, reverting when re-tagged."""
    text = "Normal. [rate:1.5] Fast. [rate:1.0] Normal again."
    assert split_by_voice(text, "af_heart") == [
        ("af_heart", 1.0, "Normal."),
        ("af_heart", 1.5, "Fast."),
        ("af_heart", 1.0, "Normal again."),
    ]


def test_split_by_rate_clamps_to_speed_bounds():
    assert split_by_voice("[rate:99] Whoa.", "af_heart") == [("af_heart", 4.0, "Whoa.")]
    assert split_by_voice("[rate:0.01] Crawl.", "af_heart") == [
        ("af_heart", 0.25, "Crawl.")
    ]


def test_split_voice_tag_resets_rate():
    """A pace stays with the voice it was set on, a voice change reverts to 1.0."""
    text = "[voice:af_bella] [rate:0.75] Slow Bella. [voice:am_michael] Normal Michael."
    assert split_by_voice(text, "af_heart") == [
        ("af_bella", 0.75, "Slow Bella."),
        ("am_michael", 1.0, "Normal Michael."),
    ]


def test_split_rate_scales_baserate():
    """An explicit rate stays relative to the voice's calibrated base pace."""
    text = "[baserate:0.5] Base. [rate:1.5] Faster. [rate:1.0] Base again."
    assert split_by_voice(text, "af_heart") == [
        ("af_heart", 0.5, "Base."),
        ("af_heart", 0.75, "Faster."),
        ("af_heart", 0.5, "Base again."),
    ]


def test_split_baserate_product_clamps_to_speed_bounds():
    assert split_by_voice("[baserate:2.5] [rate:2.0] Whoa.", "af_heart") == [
        ("af_heart", 4.0, "Whoa.")
    ]


# issue #519, pr #525
JOIN_LINES_CASES = [
    (
        "Steven Erikson\n\nFive riders drew rein in the pass.",
        "Steven Erikson. Five riders drew rein in the pass.",
    ),
    ("End of chapter.\n\nNew chapter begins.", "End of chapter. New chapter begins."),
    ("Really?\n\nYes!\n\nOk", "Really? Yes! Ok"),
    ('"Really?"\n\nYes.', '"Really?" Yes.'),
    ("“Really?”\n\nYes.", "“Really?” Yes."),
    ("It read as follows:\n\nText.", "It read as follows: Text."),
    ("He paused…\n\nThen spoke.", "He paused… Then spoke."),
    ("Heading  \r\n\r\n  Body.", "Heading. Body."),
    ("Heading\n \n\t\nBody.", "Heading. Body."),
    ("Heading\n\xa0\nBody.", "Heading. Body."),
    ("Heading\n\f\nBody.", "Heading. Body."),
    ("Five riders drew\nrein in the pass.", "Five riders drew rein in the pass."),
    ("No trailing period", "No trailing period"),
    ("  padded  ", "padded"),
]


@pytest.mark.parametrize("text, expected", JOIN_LINES_CASES)
def test_join_lines(text, expected):
    assert join_lines(text) == expected


def sizes(pieces):
    return [(p.text, len(p.phonemes)) for p in pieces]


def test_split_clauses_keeps_separators_and_long_clauses_whole(fake_g2p):
    """A clause that's still too long isn't cut between words: generation stitches it."""
    assert sizes(text_processor.split_clauses("a b, c d; e", "a")) == [
        ("a b,", 4),
        ("c d;", 4),
        ("e", 1),
    ]
    assert sizes(text_processor.split_clauses("abcdefgh ij, k", "a")) == [
        ("abcdefgh ij,", 12),
        ("k", 1),
    ]
    assert sizes(text_processor.split_clauses("a,, b", "a")) == [
        ("a,", 2),
        ("b", 1),
    ]


def test_oversized_sentence_packs_its_clauses(fake_g2p):
    """A sentence over the cap is cut into as few pieces as fit, at clause punctuation first."""
    sentence = "aaaa, bbbb, cccc, dddd."
    assert sizes(text_processor.sentence_chunks(sentence, 12, "a")) == [
        ("aaaa, bbbb,", 11),
        ("cccc, dddd.", 11),
    ]


@pytest.mark.asyncio
async def test_smart_split_caps_unpunctuated_runs():
    """A long run with no punctuation stays one chunk for generation to stitch, never truncated (issue #71, #95)."""
    text = "hello\n" * 100
    chunks = [
        (c, p)
        async for c, p, _ in smart_split(
            text, normalization_options=NormalizationOptions(normalize=False)
        )
    ]

    assert len(chunks) == 1
    assert chunks[0][0].split() == ["hello"] * 100
    assert len(chunks[0][1].phonemes) > text_processor.settings.absolute_max_tokens


@pytest.mark.asyncio
async def test_smart_split_paragraph_break_is_a_sentence_boundary():
    """A blank line reads as a sentence break (issue #519)."""
    text = "Steven Erikson\n\nFive riders drew rein in the pass."
    chunks = [c async for c, _, _ in smart_split(text)]

    assert chunks == ["Steven Erikson.", "Five riders drew rein in the pass."]


@pytest.mark.asyncio
async def test_smart_split_normalize_off_still_joins_lines():
    """Line handling is chunking, not normalization, so it applies either way."""
    text = "Steven Erikson\n\nFive riders drew\nrein in the pass. Then 3 more."
    chunks = [
        c
        async for c, _, _ in smart_split(
            text, normalization_options=NormalizationOptions(normalize=False)
        )
    ]

    assert chunks == [
        "Steven Erikson.",
        "Five riders drew rein in the pass.",
        "Then 3 more.",
    ]


@pytest.mark.asyncio
async def test_long_chinese_sentence_without_commas_stays_one_chunk():
    """No clause to cut at: generation renders it in overlapping pieces instead of failing."""
    text = "我们今天早上一起去公园散步然后在湖边看到很多人在跑步和打太极拳" * 4 + "。"
    chunks = [c async for c in smart_split(text, lang_code="z")]

    assert [c for c, _, _ in chunks] == [text]
    assert len(chunks[0][1].phonemes) > text_processor.settings.absolute_max_tokens


def test_phonemize_english_keeps_the_words_for_timestamps():
    from api.src.inference import g2p

    phonemes = g2p.phonemize("Hi there.", "a")
    assert phonemes.phonemes == "hˈI ðˈɛɹ."
    assert [t.text for t in phonemes.tokens] == ["Hi", "there", "."]
    assert g2p.phonemize("", "a").phonemes == ""
    assert g2p.phonemize("hola", "e").tokens is None


@pytest.mark.asyncio
async def test_sentence_split_keeps_custom_pronunciations_whole():
    """An annotation's text can end a "sentence" ([Hello! World]), but must reach G2P in one piece."""
    text = "Say [Hello! World](/həlˈO wˈɜɹld/) again. Then stop."
    chunks = [c async for c in smart_split(text)]

    assert [c for c, _, _ in chunks] == [
        "Say [Hello! World](/həlˈO wˈɜɹld/) again.",
        "Then stop.",
    ]
    assert "həlˈO wˈɜɹld" in chunks[0][1].phonemes


def test_english_voice_reads_other_scripts_in_their_own_language():
    """Not "Chinese letter Chinese letter": each script goes to its own language's G2P."""
    from api.src.inference import g2p

    chinese = g2p.phonemize("We had dinner in 北京 last year.", "a")
    assert "bˈeiʨˈiŋ" in chinese.phonemes
    assert "lˈɛtəɹ" not in chinese.phonemes
    assert "北京" in [t.text for t in chinese.tokens]

    russian = g2p.phonemize("The word Москва means Moscow.", "a")
    assert "mʌskvˈɑ" in russian.phonemes

    punctuation = g2p.phonemize("He said 你好，世界！ and left.", "a")
    assert ", " in punctuation.phonemes and "!" in punctuation.phonemes
