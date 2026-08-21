"""First-turn cold-start coverage: what warmup touches, and what it must never do
(billable calls, raised exceptions, worker-state interference)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from nanobot_channel_voice.audio.base import PlaybackSink, PlaybackStream
from nanobot_channel_voice.backend.audio_sink import AudioSink


def _run(coro):
    return asyncio.run(coro)


# ---- on-device TTS: every declared language warms its own frontend ----------


def test_ondevice_warmup_covers_every_declared_language():
    from nanobot_channel_voice.tts.ondevice_base import OnDeviceTtsAdapter

    class _Fake(OnDeviceTtsAdapter):
        output_rate = 16000

        def __init__(self, language, languages=()):
            self.spoken_language = language
            if languages:
                self.spoken_languages = languages
            self.texts: list[str] = []

        async def synthesize(self, text, *, voice=None):
            self.texts.append(text)
            return b"RIFF"

        def _synthesize_floats(self, text):  # abstract in the shell
            raise NotImplementedError

    # A zh-en single model (matcha is_zh_en) routes scripts through DIFFERENT
    # sub-frontends: warming only zh leaves the espeak English fold cold.
    bi = _Fake("zh", ("zh", "en"))
    _run(bi.warmup())
    assert bi.texts == ["好的。", "Okay."]
    mono = _Fake("zh")
    _run(mono.warmup())
    assert mono.texts == ["好的。"]


# ---- STT warmups: zipformer/sensevoice now decode once, like whisper --------


def test_stt_warmups_run_one_silent_decode():
    from nanobot_channel_voice.stt.sensevoice import SenseVoiceOnDeviceStt
    from nanobot_channel_voice.stt.zipformer import ZipformerOnDeviceStt

    for cls in (ZipformerOnDeviceStt, SenseVoiceOnDeviceStt):
        stt = object.__new__(cls)
        calls: list[tuple[int, int]] = []

        async def transcribe(pcm, sample_rate, _c=calls):
            _c.append((len(pcm), sample_rate))
            return ""

        stt.transcribe = transcribe  # type: ignore[method-assign]
        _run(stt.warmup())
        assert calls == [(16000, 16000)], cls.__name__  # 0.5 s of silence


# ---- cloud TTS: connection-only warmup, never a synthesis -------------------


def test_openai_compat_warmup_touches_origin_only():
    from nanobot_channel_voice.tts.openai_compat import OpenAITtsAdapter

    tts = OpenAITtsAdapter(api_key="k", api_base=None, model="m", voice="v")
    heads: list[str] = []

    class _Client:
        async def head(self, url, timeout=None):
            heads.append(str(url))
            return SimpleNamespace(status_code=404)  # any answer = connection warm

    tts._get_client = lambda: _Client()  # type: ignore[method-assign]
    _run(tts.warmup())
    assert heads == ["https://api.openai.com/"]  # origin only, no /audio/speech


def test_openai_compat_warmup_swallows_dead_endpoints():
    from nanobot_channel_voice.tts.openai_compat import OpenAITtsAdapter

    # 127.0.0.1:9 (discard) refuses instantly; warmup must not raise.
    tts = OpenAITtsAdapter(
        api_key=None, api_base="http://127.0.0.1:9/v1", model="m", voice="v",
        timeout_s=0.5,
    )
    try:
        _run(tts.warmup())
    finally:
        _run(tts.aclose())


# ---- playback prewarm -------------------------------------------------------


class _RecStream(PlaybackStream):
    def __init__(self, log):
        self.log = log

    async def write(self, pcm: bytes) -> None:
        self.log.append(("write", len(pcm)))

    async def drain(self) -> None:
        self.log.append(("drain",))

    async def kill(self) -> None:
        self.log.append(("kill",))


class _RecSink(PlaybackSink):
    def __init__(self, fail=False):
        self.log: list = []
        self.fail = fail

    async def play_wav(self, wav_bytes: bytes) -> bool:
        if self.fail:
            raise RuntimeError("no such device")
        self.log.append(("play_wav", len(wav_bytes)))
        return True

    async def open_stream(self, rate: int) -> PlaybackStream:
        if self.fail:
            raise RuntimeError("no such device")
        self.log.append(("open", rate))
        return _RecStream(self.log)

    async def abort(self) -> None:
        pass


def test_sink_prewarm_plays_silence_through_the_real_path():
    dev = _RecSink()
    sink = AudioSink(dev, mode="stream")
    _run(sink.prewarm(22050))
    # 40 ms at 22050 Hz S16, through open -> write -> drain; worker slot untouched.
    assert dev.log == [("open", 22050), ("write", int(22050 / 1000 * 40) * 2), ("drain",)]
    assert sink._stream is None

    dev = _RecSink()
    blob = AudioSink(dev, mode="blob")
    _run(blob.prewarm(16000))
    [(op, size)] = dev.log
    assert op == "play_wav" and size > 44  # a real (tiny) WAV


def test_sink_prewarm_fails_loudly_but_never_raises():
    # A wrong playbackDevice must surface at STARTUP in the log, not take the
    # channel down — and not wait for the first reply to be discovered.
    sink = AudioSink(_RecSink(fail=True), mode="stream")
    _run(sink.prewarm(16000))  # no raise


def test_backend_prewarm_playback_uses_the_tts_rate_and_skips_tts_off():
    from nanobot_channel_voice.audio.null import NullPlayback
    from nanobot_channel_voice.backend.base import VoiceState
    from nanobot_channel_voice.backend.local import LocalBackend
    from nanobot_channel_voice.config import VoiceConfig
    from nanobot_channel_voice.vad.base import Vad

    class _V(Vad):
        def is_speech(self, frame):
            return False

    async def _t():
        async def nop(*a, **k):
            return ""

        sink = AudioSink(NullPlayback(), mode="stream")
        b = LocalBackend(
            VoiceConfig(), vad=_V(), tts=None, sink=sink,
            transcribe=nop, publish_text=nop, interrupt=nop,
        )
        called: list[int] = []

        async def prewarm(rate):
            called.append(rate)

        sink.prewarm = prewarm  # type: ignore[method-assign]
        await b.prewarm_playback()
        assert called == []  # tts off: nothing will ever play
        b._tts = SimpleNamespace(output_rate=22050)
        await b.prewarm_playback()
        assert called == [22050]
        # The warmup task runs with the mic live: a fast first reply may already
        # hold the device (warm by definition) — a second open collides on hw:.
        b._turn = VoiceState.SPEAKING
        await b.prewarm_playback()
        assert called == [22050]

    _run(_t())


# ---- calibration seeds the JIT cost model -----------------------------------


def test_calibration_seeds_the_jit_cost_ema_once():
    from nanobot_channel_voice.audio.null import NullPlayback
    from nanobot_channel_voice.backend.local import LocalBackend
    from nanobot_channel_voice.config import VoiceConfig
    from nanobot_channel_voice.vad.base import Vad

    class _V(Vad):
        def is_speech(self, frame):
            return False

    async def nop(*a, **k):
        return ""

    b = LocalBackend(
        VoiceConfig(), vad=_V(), tts=None,
        sink=AudioSink(NullPlayback(), mode="stream"),
        transcribe=nop, publish_text=nop, interrupt=nop,
    )
    # Unseeded, the whole first reply bypasses JIT scheduling ("whole candidate"):
    # the probe's measured cost closes exactly that gap.
    assert b._synth_mpc is None
    b.apply_calibration(
        stt_cost_ms=None, tts_rtf=1.2, tts_ms_per_char=5.0, chunk_floor_pinned=False,
    )
    assert b._synth_mpc == 5.0
    b.apply_calibration(  # real observations own the EMA after the first seed
        stt_cost_ms=None, tts_rtf=1.2, tts_ms_per_char=9.0, chunk_floor_pinned=False,
    )
    assert b._synth_mpc == 5.0
