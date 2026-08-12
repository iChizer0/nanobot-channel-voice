"""Build-time degrade for aec="webrtc" (channel assembly seam): a blob-mode sink
never feeds the reference tap, so the channel builds the canceller only when the
resolved TTS is raw-PCM (stream-mode sink) and otherwise degrades to soft-duplex —
a reference-starved canceller cancels nothing while its warmup hold never releases.
Exercised through the real _build_local() with the null audio backend: construction
only, no device is claimed.
"""

from __future__ import annotations

import pytest
from nanobot.bus.queue import MessageBus

import nanobot_channel_voice.aec as aec_mod
from nanobot_channel_voice.channel import VoiceChannel
from nanobot_channel_voice.config import VoiceConfig


def _channel(audio_format: str) -> VoiceChannel:
    cfg = VoiceConfig.model_validate({
        "aec": "webrtc",
        "audio": {"backend": "null"},
        # openai wav => blob sink (output_rate None); pcm => stream sink.
        "tts": {"provider": "openai", "audioFormat": audio_format, "apiKey": "k"},
    })
    return VoiceChannel(cfg, MessageBus())


def test_wav_blob_tts_degrades_to_soft_without_building_a_canceller(monkeypatch):
    monkeypatch.setattr(
        aec_mod, "make_echo_canceller",
        lambda *a, **k: pytest.fail("built a canceller a blob sink can never feed"),
    )
    ch = _channel("wav")
    ch._build_local()
    backend = ch._backend
    assert backend._aec is None            # no reference-starved canceller
    assert backend._open_mic               # degrade is to SOFT: the mic stays open
    assert not backend._full_duplex        # soft wiring, not asserted-AEC full duplex
    assert not backend._sink.stream_mode


def test_pcm_tts_builds_and_taps_the_canceller(monkeypatch):
    stub = object()
    monkeypatch.setattr(aec_mod, "make_echo_canceller", lambda *a, **k: stub)
    ch = _channel("pcm")
    ch._build_local()
    backend = ch._backend
    assert backend._aec is stub
    assert backend._sink._ref_tap is stub  # the sink will feed it every block
    assert backend._sink.stream_mode
    assert backend._full_duplex
