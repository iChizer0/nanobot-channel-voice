"""debug.dumpAudio: capture segments leave as verdict-named WAVs (with a pre-AEC
raw twin under software AEC), so false barge-ins can be diagnosed by ear."""

from __future__ import annotations

import asyncio
import wave

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import VoiceState
from nanobot_channel_voice.backend.local import LocalBackend
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.dump import AudioDumper
from nanobot_channel_voice.vad.base import Vad

_FRAME = b"\x01\x00" * 320  # 20 ms @ 16 kHz


class _ScriptedVad(Vad):
    def __init__(self, script: list[bool]) -> None:
        self.script = list(script)

    def is_speech(self, frame: bytes) -> bool:
        return self.script.pop(0) if self.script else False

    def scale_floor(self, factor: float) -> None:
        pass


class _IdentityAec:
    """Stand-in canceller: presence turns the raw ring on; identity keeps the
    pair comparable byte-for-byte."""

    def process(self, pcm: bytes) -> bytes:
        return pcm


class _Harness:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self.transcript = ""


def _build(vad: Vad, tmp, *, mode: str = "duck", aec=None) -> _Harness:
    cfg = VoiceConfig.model_validate(
        {
            "aec": "soft",
            "duckDb": -12.0,
            "bargeIn": {"mode": mode},
            "debug": {"dumpAudio": True, "dumpDir": str(tmp), "dumpMaxMb": 50},
        }
    )
    harness = _Harness()

    async def transcribe(pcm: bytes) -> str:
        return harness.transcript

    async def publish(text: str, token: str) -> None:
        harness.published.append((text, token))

    async def interrupt() -> None:
        pass

    harness.backend = LocalBackend(
        cfg,
        vad=vad,
        tts=None,
        sink=AudioSink(NullPlayback(), mode="stream"),
        transcribe=transcribe,
        publish_text=publish,
        interrupt=interrupt,
        aec=aec,
    )
    return harness


async def _start(backend: LocalBackend) -> None:
    """Spin the utterance/TTS workers up: the dump-at-verdict hook lives in the
    worker, so these tests need the real pump -> queue -> worker path."""

    async def _sink_events(event) -> None:
        pass

    await backend.start(instructions=None, tools=[], on_event=_sink_events)


async def _drain_and_close(backend: LocalBackend) -> list[str]:
    await backend._utt_queue.join()
    dump_dir = backend._dumper.dir
    await backend.close()  # joins the writer thread, so every submit is on disk
    return sorted(p.name for p in dump_dir.glob("*.wav"))


# ---- dumper unit -------------------------------------------------------------

def test_dumper_writes_verdict_named_pairs(tmp_path):
    d = AudioDumper(tmp_path, 16000, 10 * 1024 * 1024)
    d.submit("empty", b"\x01\x00" * 1600, b"\x02\x00" * 1600)
    d.submit("publish", b"\x03\x00" * 800)  # no raw twin
    d.close()
    names = sorted(p.name for p in d.dir.glob("*.wav"))
    assert len(names) == 3
    assert names[0].startswith("utt-0001-") and names[0].endswith("-empty.raw.wav")
    assert names[1].startswith("utt-0001-") and names[1].endswith("-empty.wav")
    assert names[2].startswith("utt-0002-") and names[2].endswith("-publish.wav")
    with wave.open(str(d.dir / names[1]), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getnframes() == 1600  # 100 ms


def test_dumper_session_cap_deletes_oldest_first(tmp_path):
    # Each segment ~2 KB on disk; a 5 KB cap keeps roughly the newest two.
    d = AudioDumper(tmp_path, 16000, 5 * 1024)
    for _ in range(4):
        d.submit("empty", b"\x01\x00" * 1000)
    d.close()
    names = sorted(p.name for p in d.dir.glob("*.wav"))
    assert names and "utt-0004-" in names[-1]  # newest survived
    assert not any("utt-0001-" in n for n in names)  # oldest pruned


def test_dumper_prunes_old_sessions_at_startup(tmp_path):
    stale = tmp_path / "20200101-000000"
    stale.mkdir()
    (stale / "utt-0001-000000-empty.wav").write_bytes(b"\x00" * 8192)
    keep = tmp_path / "not-a-session"
    keep.mkdir()
    (keep / "sample.wav").write_bytes(b"\x00" * 8192)
    d = AudioDumper(tmp_path, 16000, 4 * 1024)
    d.close()
    assert not stale.exists()  # over the cap -> whole old session pruned
    assert keep.exists()  # non-timestamped dirs are never touched


# ---- backend integration -----------------------------------------------------

def test_empty_verdict_segment_is_dumped(tmp_path):
    async def _case():
        # 12 speech frames (240 ms >= minUtteranceMs) + hangover -> close; STT
        # decodes nothing -> the classic leak-shaped false barge-in.
        vad = _ScriptedVad([True] * 12 + [False] * 35)
        h = _build(vad, tmp_path)
        await _start(h.backend)
        h.transcript = ""
        for _ in range(46):
            await h.backend.push_audio(_FRAME)
        return await _drain_and_close(h.backend)

    names = asyncio.run(_case())
    assert any(n.endswith("-empty.wav") for n in names)


def test_published_utterance_dumps_with_publish_verdict(tmp_path):
    async def _case():
        vad = _ScriptedVad([True] * 12 + [False] * 35)
        h = _build(vad, tmp_path)
        await _start(h.backend)
        h.transcript = "what time is it"
        for _ in range(46):
            await h.backend.push_audio(_FRAME)
        names = await _drain_and_close(h.backend)
        assert h.published  # sanity: the verdict really was publish
        return names

    names = asyncio.run(_case())
    assert any(n.endswith("-publish.wav") for n in names)


def test_blip_reject_is_dumped(tmp_path):
    async def _case():
        # Onset confirms at 5 frames but only 6 flagged (120 ms < 200 ms min):
        # the min filter rejects, which without the dump left nothing to hear.
        vad = _ScriptedVad([True] * 6 + [False] * 35)
        h = _build(vad, tmp_path)
        await _start(h.backend)
        for _ in range(42):
            await h.backend.push_audio(_FRAME)
        return await _drain_and_close(h.backend)

    names = asyncio.run(_case())
    assert any(n.endswith("-blip.wav") for n in names)


def test_probe_drop_is_dumped(tmp_path):
    async def _case():
        # The pause-probe scenario from test_false_barge_in: the candidate dies
        # inside the leak-death window and is dropped whole, pre-verdict.
        vad = _ScriptedVad([True] * 5 + [False] * 40)
        h = _build(vad, tmp_path, mode="pause")
        await _start(h.backend)
        h.backend._turn = VoiceState.SPEAKING
        for _ in range(18):
            await h.backend.push_audio(_FRAME)
        assert h.backend._metrics.counters.get("barge_in_false_resume.probe") == 1
        return await _drain_and_close(h.backend)

    names = asyncio.run(_case())
    assert any(n.endswith("-probe.wav") for n in names)


def test_aec_active_writes_matching_raw_twin(tmp_path):
    async def _case():
        vad = _ScriptedVad([True] * 12 + [False] * 35)
        h = _build(vad, tmp_path, aec=_IdentityAec())
        await _start(h.backend)
        h.transcript = "hello"
        for _ in range(46):
            await h.backend.push_audio(_FRAME)
        dump_dir = h.backend._dumper.dir
        await _drain_and_close(h.backend)
        main = next(p for p in dump_dir.glob("*-publish.wav"))
        twin = next(p for p in dump_dir.glob("*-publish.raw.wav"))
        with wave.open(str(main), "rb") as a, wave.open(str(twin), "rb") as b:
            pcm_a = a.readframes(a.getnframes())
            pcm_b = b.readframes(b.getnframes())
        return pcm_a, pcm_b

    pcm_a, pcm_b = asyncio.run(_case())
    assert pcm_a == pcm_b  # identity AEC: the raw span aligns byte-for-byte


def test_flooring_aec_keeps_the_twin_aligned(tmp_path):
    class _FlooringAec:
        """Like AEC3: emits only whole 10 ms blocks, dropping a ragged remainder."""

        def process(self, pcm: bytes) -> bytes:
            return pcm[: len(pcm) // 320 * 320]

    async def _case():
        vad = _ScriptedVad([True] * 12 + [False] * 35)
        h = _build(vad, tmp_path, aec=_FlooringAec())
        await _start(h.backend)
        h.transcript = "hello"
        for i in range(46):
            # One ragged frame mid-utterance: its dropped tail must not enter the
            # raw ring, or every later twin slices shifted.
            await h.backend.push_audio(_FRAME + b"\x07\x00" if i == 3 else _FRAME)
        dump_dir = h.backend._dumper.dir
        await _drain_and_close(h.backend)
        main = next(p for p in dump_dir.glob("*-publish.wav"))
        twin = next(p for p in dump_dir.glob("*-publish.raw.wav"))
        with wave.open(str(main), "rb") as a, wave.open(str(twin), "rb") as b:
            return a.readframes(a.getnframes()), b.readframes(b.getnframes())

    pcm_a, pcm_b = asyncio.run(_case())
    assert pcm_a == pcm_b


def test_dump_off_keeps_endpointer_lean(tmp_path):
    cfg = VoiceConfig.model_validate({"aec": "soft"})

    async def _noop_text(text: str, token: str) -> None:
        pass

    async def _noop_stt(pcm: bytes) -> str:
        return ""

    async def _noop() -> None:
        pass

    backend = LocalBackend(
        cfg,
        vad=_ScriptedVad([]),
        tts=None,
        sink=AudioSink(NullPlayback(), mode="stream"),
        transcribe=_noop_stt,
        publish_text=_noop_text,
        interrupt=_noop,
    )
    assert backend._dumper is None
    assert backend._dump_raw is None
    assert backend._endpointer.keep_rejected is False


def test_debug_config_parses_camel_case():
    cfg = VoiceConfig.model_validate(
        {"debug": {"dumpAudio": True, "dumpDir": "/tmp/x", "dumpMaxMb": 10}}
    )
    assert cfg.debug.dump_audio is True
    assert cfg.debug.dump_dir == "/tmp/x"
    assert cfg.debug.dump_max_mb == 10
