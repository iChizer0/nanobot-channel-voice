"""Cloud/compat TTS adapter guards: URL resolution, the retry ladder and its budget,
and the lying-server defenses (text bodies, encoded audio for pcm, WAV-in-pcm strips).
These paths only run when a server misbehaves, so they are exactly what manual testing
against a healthy endpoint never exercises."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes
from nanobot_channel_voice.config import TtsConfig
from nanobot_channel_voice.tts import make_tts
from nanobot_channel_voice.tts.openai_compat import OpenAITtsAdapter, _speech_url


def _run(coro):
    return asyncio.run(coro)


def _adapter(handler=None, *, api_key="k", api_base=None, fmt="wav", **kw) -> OpenAITtsAdapter:
    tts = OpenAITtsAdapter(
        api_key=api_key, api_base=api_base, model="m", voice="v", audio_format=fmt, **kw
    )
    if handler is not None:
        tts._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return tts


async def _synth(tts: OpenAITtsAdapter, text="hello", *, pcm=False) -> bytes:
    try:
        return await (tts.synthesize_pcm(text) if pcm else tts.synthesize(text))
    finally:
        await tts.aclose()


_WAV = pcm_to_wav_bytes(b"\x01\x02" * 800, 24000)


def test_speech_url_resolution():
    # Chat-style base gets the path appended (the form users copy); full URL kept.
    assert _speech_url(None) == "https://api.openai.com/v1/audio/speech"
    assert _speech_url("http://h:8880/v1") == "http://h:8880/v1/audio/speech"
    assert _speech_url("http://h:8880/v1/audio/speech") == "http://h:8880/v1/audio/speech"


def test_transient_status_is_retried_then_wins(monkeypatch):
    from nanobot_channel_voice.tts import openai_compat

    monkeypatch.setattr(openai_compat, "_BACKOFF_S", (0.0, 0.0))
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, content=_WAV)

    assert _run(_synth(_adapter(handler))) == _WAV
    assert len(seen) == 2
    assert seen[0].headers["authorization"] == "Bearer k"


def test_auth_errors_fail_fast_without_retry():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    # Retrying a 401 only wastes quota: one request, empty result.
    assert _run(_synth(_adapter(handler))) == b""
    assert len(seen) == 1


def test_transport_errors_retry_then_give_up(monkeypatch):
    from nanobot_channel_voice.tts import openai_compat

    monkeypatch.setattr(openai_compat, "_BACKOFF_S", (0.0, 0.0))
    seen = []

    def handler(request):
        seen.append(request)
        raise httpx.ConnectError("refused")

    assert _run(_synth(_adapter(handler))) == b""
    assert len(seen) == 3  # _RETRIES + 1


def test_ladder_budget_bounds_total_wall_time(monkeypatch):
    from nanobot_channel_voice.tts import openai_compat

    # Backoff longer than the whole 2x-timeout budget: the clamp must land the sleep
    # on the deadline and the NEXT attempt must refuse to start.
    monkeypatch.setattr(openai_compat, "_BACKOFF_S", (60.0, 60.0))
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(503, text="overloaded")

    assert _run(_synth(_adapter(handler, timeout_s=0.05))) == b""
    assert len(seen) == 1


def test_text_body_with_200_is_not_audio():
    def handler(request):
        return httpx.Response(200, json={"detail": "queued"})

    assert _run(_synth(_adapter(handler))) == b""


def test_wav_mode_drops_non_wav_bodies():
    def handler(request):
        return httpx.Response(200, content=b"\xff\xf3garbage-mp3")

    assert _run(_synth(_adapter(handler))) == b""


def test_pcm_mode_drops_encoded_audio():
    def handler(request):
        return httpx.Response(200, content=b"ID3\x04rest-of-an-mp3")

    # Streamed raw, an mp3 is sustained full-scale noise: both entry points drop it.
    assert _run(_synth(_adapter(handler, fmt="pcm"), pcm=True)) == b""
    assert _run(_synth(_adapter(handler, fmt="pcm"))) == b""


def test_pcm_mode_strips_a_wav_lie_and_rejects_bad_geometry():
    frames = b"\x01\x02" * 1200

    def wav_handler(request):
        return httpx.Response(200, content=pcm_to_wav_bytes(frames, 24000))

    assert _run(_synth(_adapter(wav_handler, fmt="pcm"), pcm=True)) == frames
    # Blob mode takes the same body: unstripped, the inner 44-byte RIFF header played
    # as ~22 full-scale samples at the head of every chunk.
    assert _run(_synth(_adapter(wav_handler, fmt="pcm"))) == pcm_to_wav_bytes(frames, 24000)

    # A WAV at another rate: the pcm stream cannot fix it, but the blob carries the
    # WAV's OWN rate so the sink plays it at speed.
    def slow_handler(request):
        return httpx.Response(200, content=pcm_to_wav_bytes(frames, 22050))

    assert _run(_synth(_adapter(slow_handler, fmt="pcm"))) == pcm_to_wav_bytes(frames, 22050)
    assert _run(_synth(_adapter(slow_handler, fmt="pcm"), pcm=True)) == frames

    def stereo_handler(request):
        return httpx.Response(200, content=pcm_to_wav_bytes(frames, 24000, channels=2))

    assert _run(_synth(_adapter(stereo_handler, fmt="pcm"), pcm=True)) == b""
    assert _run(_synth(_adapter(stereo_handler, fmt="pcm"))) == b""


def test_pcm_mode_trims_a_torn_trailing_byte():
    def handler(request):
        return httpx.Response(200, content=b"\x01\x02\x03\x04\x05")

    # An odd byte count misaligns every later S16 sample in the stream.
    assert _run(_synth(_adapter(handler, fmt="pcm"), pcm=True)) == b"\x01\x02\x03\x04"
    assert _run(_synth(_adapter(handler, fmt="pcm"))) == pcm_to_wav_bytes(
        b"\x01\x02\x03\x04", 24000
    )


def test_pcm_format_synthesize_wraps_to_wav_at_the_declared_rate():
    def handler(request):
        return httpx.Response(200, content=b"\x01\x02" * 400)

    out = _run(_synth(_adapter(handler, fmt="pcm", pcm_sample_rate=22050)))
    assert out == pcm_to_wav_bytes(b"\x01\x02" * 400, 22050)


# ---- factory policy ---------------------------------------------------------


def test_keyless_default_endpoint_falls_back_to_system(monkeypatch):
    from nanobot_channel_voice.tts.system import SystemTtsAdapter

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Billed default endpoint with no key would 401 every synthesis: the build refuses
    # and the one fallback policy speaks instead of a mute channel.
    tts = make_tts(TtsConfig.model_validate({"provider": "openai"}))
    assert isinstance(tts, SystemTtsAdapter)
    # A keyless COMPAT server is a real configuration: the adapter builds, no auth header.
    tts = make_tts(TtsConfig.model_validate({"provider": "openai", "apiBase": "http://h/v1"}))
    assert isinstance(tts, OpenAITtsAdapter)


@pytest.mark.parametrize(
    ("cfg", "expected"),
    [
        ({"provider": "openai", "apiKey": "k"}, False),                      # auto: billed
        ({"provider": "openai", "apiBase": "http://h/v1"}, True),            # auto: presumed local
        ({"provider": "openai", "apiBase": "http://h/v1", "probe": "off"}, False),
        ({"provider": "openai", "apiKey": "k", "probe": "on"}, True),
        ({"provider": "system", "probe": "off"}, False),                     # operator intent wins
    ],
)
def test_probe_override(cfg, expected):
    tts = make_tts(TtsConfig.model_validate(cfg))
    assert tts is not None and tts.probe_ok is expected
