"""Pure audio/text helpers: trim, gain, WAV codec, budget splitter, tokens."""

from __future__ import annotations

import asyncio
import io
import os
import struct
import tempfile
import wave

import pytest

import nanobot_channel_voice.backend.audio_sink as sink_mod
import nanobot_channel_voice.stt.base as stt_base
from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes
from nanobot_channel_voice.backend.audio_sink import AudioSink, scale_pcm, trim_lead_silence
from nanobot_channel_voice.backend.base import OutputAudio
from nanobot_channel_voice.stt.base import pcm_to_float_mono, read_token_table, write_temp_wav
from nanobot_channel_voice.tts.base import is_wav, split_for_budget

RATE = 16000


def tone(n: int, amp: int = 16384) -> bytes:
    return struct.pack("<h", amp) * n


def silence(n: int) -> bytes:
    return b"\x00\x00" * n


# ---- trim_lead_silence ------------------------------------------------------

def test_trim_caps_leading_silence_keeps_preroll():
    pcm = silence(3200) + tone(1600)  # 200 ms silence then speech @16k
    out = trim_lead_silence(pcm, RATE, cap_ms=20.0)
    assert len(out) == (320 + 1600) * 2  # 20 ms preroll + the speech
    assert out[: 320 * 2] == silence(320)


def test_trim_leaves_short_leads_and_pure_silence_alone():
    short = silence(100) + tone(100)
    assert trim_lead_silence(short, RATE, cap_ms=20.0) == short  # under the cap
    quiet = silence(4000)
    assert trim_lead_silence(quiet, RATE, cap_ms=20.0) == quiet  # intentional silence


def test_trim_pure_python_path_matches(monkeypatch):
    monkeypatch.setattr(sink_mod, "_np", None)
    pcm = silence(3200) + tone(1600)
    assert trim_lead_silence(pcm, RATE, cap_ms=20.0) == (
        silence(320) + tone(1600)
    )


# ---- scale_pcm --------------------------------------------------------------

def test_scale_pcm_applies_gain_and_is_identity_at_unity():
    pcm = struct.pack("<3h", 1000, -2000, 30000)
    assert scale_pcm(pcm, 1.0) is pcm
    assert struct.unpack("<3h", scale_pcm(pcm, 0.5)) == (500, -1000, 15000)


def test_scale_pcm_trims_odd_trailing_byte():
    pcm = struct.pack("<3h", 1, 2, 3) + b"\x7f"
    assert len(scale_pcm(pcm, 0.5)) == 6


def test_scale_pcm_pure_python_path_matches(monkeypatch):
    pcm = struct.pack("<3h", 1000, -2000, 30000)
    with_np = scale_pcm(pcm, 0.25)
    monkeypatch.setattr(sink_mod, "_np", None)
    assert scale_pcm(pcm, 0.25) == with_np


# ---- pcm_rms ----------------------------------------------------------------

def test_pcm_rms_measures_normalized_amplitude(monkeypatch):
    import nanobot_channel_voice.audio.pcm as pcm_mod
    from nanobot_channel_voice.audio.pcm import pcm_rms

    assert pcm_rms(b"") == 0.0
    assert pcm_rms(b"\x00" * 640) == 0.0
    full = struct.pack("<4h", 32000, -32000, 32000, -32000)
    assert 0.95 < pcm_rms(full) < 1.0
    assert pcm_rms(full + b"\x7f") == pcm_rms(full)  # odd trailing byte tolerated
    with_np = pcm_rms(full)
    monkeypatch.setattr(pcm_mod, "_np", None)
    assert abs(pcm_rms(full) - with_np) < 1e-9


# ---- quietest_split (wake-trim snap) ----------------------------------------

def test_quietest_split_finds_the_dip():
    from nanobot_channel_voice.audio.pcm import quietest_split

    # 100 ms loud, a 10 ms near-silent gap, 100 ms loud: the cut is the gap's end.
    pcm = tone(1600, amp=8000) + tone(160, amp=10) + tone(1600, amp=8000)
    assert quietest_split(pcm, RATE) == (1600 + 160) * 2


def test_quietest_split_ties_latest_and_short_input_passes():
    from nanobot_channel_voice.audio.pcm import quietest_split

    flat = tone(1600, amp=1)  # constant amplitude: every window ties
    assert quietest_split(flat, RATE) == len(flat)  # latest wins -> the hit cut
    assert quietest_split(tone(8), RATE) == 16      # under one window: unchanged


def test_quietest_split_only_searches_the_back_window():
    from nanobot_channel_voice.audio.pcm import quietest_split

    # The true silence lies 300 ms back, outside back_ms=240: it must not win.
    pcm = tone(160, amp=0) + tone(4800, amp=8000) + tone(160, amp=100)
    assert quietest_split(pcm, RATE) == len(pcm)


def test_quietest_split_pure_python_path_matches(monkeypatch):
    import nanobot_channel_voice.audio.pcm as pcm_mod
    from nanobot_channel_voice.audio.pcm import quietest_split

    pcm = tone(1600, amp=8000) + tone(160, amp=10) + tone(1600, amp=8000)
    with_np = quietest_split(pcm, RATE)
    monkeypatch.setattr(pcm_mod, "_np", None)
    assert quietest_split(pcm, RATE) == with_np


# ---- WAV codec --------------------------------------------------------------

def test_pcm_to_wav_bytes_roundtrip():
    blob = pcm_to_wav_bytes(tone(100), RATE)
    assert is_wav(blob)
    with wave.open(io.BytesIO(blob), "rb") as w:
        assert w.getframerate() == RATE and w.getnframes() == 100


def test_is_wav_rejects_short_and_foreign_data():
    assert not is_wav(b"RIFF")
    assert not is_wav(b"\xff" * 64)


def test_write_temp_wav_writes_parseable_mono_s16_file():
    path = write_temp_wav(tone(50), RATE)
    try:
        with wave.open(path, "rb") as w:
            # nanobot's transcribe_audio hands this file to a Whisper endpoint:
            # geometry has to survive, not just parseability.
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, RATE)
            assert w.getnframes() == 50
    finally:
        os.unlink(path)


def test_write_temp_wav_leaves_no_file_behind_when_the_write_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(stt_base, "pcm_to_wav_bytes", lambda *a, **k: (_ for _ in ()).throw(OSError))
    with pytest.raises(OSError):
        write_temp_wav(tone(50), RATE)
    assert not list(tmp_path.iterdir())  # mkstemp's empty file must be reaped


# ---- split_for_budget -------------------------------------------------------

def test_split_prefers_spaces():
    assert split_for_budget("hello world foo", 11) == ["hello world", "foo"]


def test_split_falls_back_to_cjk_punctuation():
    assert split_for_budget("你好，世界。再见吧", 4) == ["你好，", "世界。", "再见吧"]


def test_split_never_cuts_inside_a_grouped_number():
    pieces = split_for_budget("金额是1,902,567,338，请再次核对确认", 18)
    assert pieces[0].endswith("1,902,567,338，")


def test_split_hard_cuts_unbroken_runs():
    pieces = split_for_budget("a" * 10, 4)
    assert pieces == ["aaaa", "aaaa", "aa"]
    assert all(len(p) <= 4 for p in pieces)


def test_split_short_input_untouched():
    assert split_for_budget("  hi  ", 100) == ["hi"]


# ---- token table ------------------------------------------------------------

def test_read_token_table_handles_crlf_and_negative_ids(tmp_path):
    p = tmp_path / "tokens.txt"
    p.write_bytes("hello 1\nwörld 2\r\n▁x -3\n\n".encode())
    assert read_token_table(str(p)) == {1: "hello", 2: "wörld", -3: "▁x"}


def test_read_token_table_raises_loud_on_garbage(tmp_path):
    p = tmp_path / "empty.txt"
    p.write_text("not a table\n")
    with pytest.raises(ValueError, match="no tokens parsed"):
        read_token_table(str(p))


# ---- pcm_to_float_mono ------------------------------------------------------

def test_pcm_to_float_mono_scales_and_resamples():
    np = pytest.importorskip("numpy")
    pcm = struct.pack("<2h", -32768, 16384)
    out = pcm_to_float_mono(pcm, RATE, RATE)
    assert np.allclose(out, [-1.0, 0.5])
    up = pcm_to_float_mono(tone(100), RATE, RATE * 2)
    assert len(up) == 200  # linear resample doubles the samples


def test_downsample_rejects_above_nyquist_energy():
    """Regression: linear-interp downsampling folded 8-24 kHz energy into the
    speech band (aliasing); the FFT path must kill it while passing speech."""
    np = pytest.importorskip("numpy")

    def sine_pcm(freq_hz: float, rate: int, n: int) -> bytes:
        t = np.arange(n) / rate
        return (np.sin(2 * np.pi * freq_hz * t) * 16000).astype("<i2").tobytes()

    n = 48000  # 1 s at 48 kHz
    hiss = pcm_to_float_mono(sine_pcm(20000, 48000, n), 48000, 16000)  # above 8 k Nyquist
    speech = pcm_to_float_mono(sine_pcm(1000, 48000, n), 48000, 16000)
    assert len(hiss) == len(speech) == 16000
    rms = lambda a: float(np.sqrt(np.mean(a * a)))  # noqa: E731
    assert rms(hiss) < 0.01  # aliased image suppressed (linear interp left ~0.3)
    assert 0.25 < rms(speech) < 0.45  # in-band content passes at amplitude


# ---- AudioSink epoch/flush (no worker needed) -------------------------------

def test_flush_invalidates_queue_and_bumps_epoch():
    async def _run():
        sink = AudioSink(NullPlayback(), mode="blob")
        sink.enqueue(OutputAudio(epoch=0, wav=b"RIFF0000WAVE"))
        sink.enqueue(OutputAudio(epoch=0, wav=b"RIFF0000WAVE"))
        assert sink.busy
        played = await sink.flush()
        assert played == 0  # blob mode has no sub-blob accounting
        assert sink.epoch == 1
        assert not sink.busy
        assert sink.backlog_ms() == 0

    asyncio.run(_run())


def test_stream_generation_advances_when_a_cancelled_drain_reroutes_writes():
    """Regression: after a cancelled drain_stream the EOF'd handle stays
    installed while new writes open a FRESH stream whose played_ms restarts at
    0: span bases captured against the old stream are void. The generation
    counter is what lets the heard-up-to accounting detect that."""
    from nanobot_channel_voice.audio.base import PlaybackStream

    class _SlowDrainStream(PlaybackStream):
        def __init__(self):
            self.gate = asyncio.Event()

        async def write(self, pcm: bytes) -> None:
            await asyncio.sleep(0)

        async def drain(self) -> None:
            await self.gate.wait()  # a real device plays its tail out here

        async def kill(self) -> None:
            self.gate.set()

    class _SlowDrainPlayback(NullPlayback):
        async def open_stream(self, rate: int) -> PlaybackStream:
            return _SlowDrainStream()

    async def _run():
        sink = AudioSink(_SlowDrainPlayback(), mode="stream")
        await sink.start()
        pcm = OutputAudio(epoch=0, pcm=b"\x00" * 3200, rate=16000)
        g0 = sink.stream_generation
        sink.enqueue(pcm)
        await sink.wait_idle()
        assert sink.stream_generation == g0 + 1  # first write published a stream

        drain = asyncio.ensure_future(sink.drain_stream())
        await asyncio.sleep(0.01)  # parked inside stream.drain()
        drain.cancel()
        try:
            await drain
        except asyncio.CancelledError:
            pass

        sink.enqueue(pcm)  # must reroute away from the EOF'd handle
        await sink.wait_idle()
        assert sink.stream_generation == g0 + 2
        await sink.stop()

    asyncio.run(_run())


def test_duck_targets_configured_floor_only_in_range():
    sink = AudioSink(NullPlayback(), mode="stream")
    sink.configure_duck(0.25)
    sink.duck(True)
    assert sink._gain_target == 0.25
    sink.duck(False)
    assert sink._gain_target == 1.0
    sink.configure_duck(2.0)  # clamped into [0, 1]
    assert sink._duck_floor == 1.0
