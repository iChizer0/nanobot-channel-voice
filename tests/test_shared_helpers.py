"""The consolidated helpers: streamid, aio, turn-event mixin, engines, pcm, on-device TTS shell."""

from __future__ import annotations

import asyncio

import pytest

from nanobot_channel_voice.aio import (
    Throttle,
    cancel_and_wait,
    cancel_task,
    put_drop_oldest,
    wait_for_stall,
)
from nanobot_channel_voice.audio.pcm import pcm_ms, pcm_to_wav_bytes, wav_duration_ms
from nanobot_channel_voice.engines import EngineSpec, describe_build_error, missing_fields
from nanobot_channel_voice.streamid import base_of, started_ns, unique_token

# ---- streamid ---------------------------------------------------------------

NS = 1_754_000_000_000_000_000  # a plausible time_ns


def test_base_of_strips_only_the_segment():
    assert base_of(f"voice:local:{NS}:2") == f"voice:local:{NS}"
    assert base_of("solo") == "solo"
    assert base_of(None) is None
    assert base_of("") is None


def test_started_ns_accepts_full_id_and_bare_base():
    assert started_ns(f"voice:local:{NS}:2") == NS
    assert started_ns(f"voice:local:{NS}") == NS
    assert started_ns("voice:local:123:2") is None  # too short for time_ns
    assert started_ns("no timestamps") is None
    assert started_ns(None) is None
    assert started_ns("") is None


def test_unique_token_cannot_collide_within_a_clock_tick():
    """Bare str(time.time_ns()) collided inside one clock quantum (an observed
    test flake); identity tokens must never be equal."""
    minted = [unique_token() for _ in range(10_000)]
    assert len(set(minted)) == len(minted)


# ---- aio --------------------------------------------------------------------

def test_cancel_helpers_tolerate_none_done_and_failing_tasks():
    async def _case():
        cancel_task(None)
        await cancel_and_wait(None)

        async def boom():
            raise RuntimeError("dies")

        t = asyncio.create_task(boom())
        await asyncio.sleep(0)  # let it fail
        cancel_task(t)          # done task: no-op, no raise
        await cancel_and_wait(t)  # failure swallowed

        async def hang():
            await asyncio.Event().wait()

        t2 = asyncio.create_task(hang())
        await cancel_and_wait(t2)
        assert t2.cancelled()

    asyncio.run(_case())


def test_cancel_and_wait_propagates_callers_own_cancellation():
    """A caller cancelled while parked in cancel_and_wait must propagate its OWN
    CancelledError, not mistake it for the child's: teardown paths the
    ChannelManager cancels from above have to stop, not run on."""

    async def _case():
        async def stubborn():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(0.05)  # slow teardown: the caller parks here
                raise

        survived = []

        async def caller():
            child = asyncio.create_task(stubborn())
            await asyncio.sleep(0)  # child is running
            await cancel_and_wait(child)
            survived.append(True)  # must NOT be reached when caller is cancelled

        task = asyncio.create_task(caller())
        await asyncio.sleep(0.01)  # caller is parked awaiting the stubborn child
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert survived == []

    asyncio.run(_case())


def test_wait_for_stall_tracks_the_moving_stamp(monkeypatch):
    import nanobot_channel_voice.aio as aio_mod

    clock = [0.0]
    sleeps: list[float] = []
    monkeypatch.setattr(aio_mod.time, "monotonic", lambda: clock[0])

    async def fake_sleep(s):
        sleeps.append(s)
        clock[0] += s

    stamp = [0.0]

    async def _case():
        async def bump_midway(s):  # activity arrives while the deadman sleeps
            await fake_sleep(s)
            if len(sleeps) == 1:
                stamp[0] = clock[0]  # push the deadline forward once

        monkeypatch.setattr(aio_mod.asyncio, "sleep", bump_midway)
        await wait_for_stall(lambda: stamp[0], 10.0)

    asyncio.run(_case())
    # First sleep waited the full budget; the bumped stamp forced a second wait.
    assert len(sleeps) == 2 and sleeps[0] == 10.0


def test_throttle_first_call_ready_then_gated(monkeypatch):
    import nanobot_channel_voice.aio as aio_mod

    clock = [100.0]
    monkeypatch.setattr(aio_mod.time, "monotonic", lambda: clock[0])
    t = Throttle(30.0)
    assert t.ready() is True     # first call always fires
    assert t.ready() is False
    clock[0] += 29.0
    assert t.ready() is False    # suppressed calls never push the window
    clock[0] += 1.0
    assert t.ready() is True


def test_put_drop_oldest_keeps_the_newest():
    async def _case():
        q: asyncio.Queue = asyncio.Queue(maxsize=2)
        assert put_drop_oldest(q, "a") is None
        assert put_drop_oldest(q, "b") is None
        assert put_drop_oldest(q, "c") == "a"  # oldest dropped, task_done'd
        assert [q.get_nowait(), q.get_nowait()] == ["b", "c"]
        q.task_done(), q.task_done()
        await q.join()  # accounting stayed balanced through the drop

    asyncio.run(_case())


def test_turn_event_mixin_gates_on_closing():
    from nanobot_channel_voice.backend.base import StateHint, VoiceState
    from nanobot_channel_voice.backend.common import TurnEventMixin

    class Fake(TurnEventMixin):
        def __init__(self):
            self._on_event = None
            self._closing = False
            self._turn = VoiceState.IDLE

    async def _case():
        fake = Fake()
        seen = []

        async def on_event(e):
            seen.append(e)

        await fake._set_turn(VoiceState.THINKING)  # _on_event not wired: applied, nothing emitted
        fake._on_event = on_event
        await fake._set_turn(VoiceState.THINKING)  # no-op: unchanged state
        await fake._set_turn(VoiceState.SPEAKING)
        assert seen == [StateHint(VoiceState.SPEAKING)]
        fake._closing = True
        await fake._set_turn(VoiceState.IDLE)      # state applied, emit dropped
        assert fake._turn is VoiceState.IDLE and len(seen) == 1

    asyncio.run(_case())


# ---- engines ----------------------------------------------------------------

class _Nested:
    inner_path = None


class _Cfg:
    top_path = "/model.onnx"
    nested = _Nested()


def test_missing_fields_resolves_dotted_paths():
    spec = EngineSpec(
        build=lambda cfg: cfg,
        required=(("top_path", "topPath"), ("nested.inner_path", "nested.innerPath")),
    )
    assert missing_fields(_Cfg(), spec) == ["nested.innerPath"]


def test_describe_build_error_names_the_extra():
    exc = ModuleNotFoundError("No module named 'kaldi_native_fbank'",
                              name="kaldi_native_fbank")
    assert "[ondevice]" in describe_build_error(exc)
    sub = ModuleNotFoundError(
        "No module named 'numpy.random._pickle'", name="numpy.random._pickle"
    )
    assert "[ondevice]" in describe_build_error(sub)  # top-level module decides
    plain = ValueError("bad model header")
    assert describe_build_error(plain) == "bad model header"
    unknown = ModuleNotFoundError("No module named 'left_pad'", name="left_pad")
    assert "pip install" not in describe_build_error(unknown)


# ---- audio.pcm --------------------------------------------------------------

def test_pcm_ms_and_wav_duration_agree():
    pcm = b"\x00\x00" * 1600  # 100 ms @ 16 kHz
    assert pcm_ms(len(pcm), 16000) == 100.0
    assert pcm_ms(len(pcm), 0) == 0.0
    assert wav_duration_ms(pcm_to_wav_bytes(pcm, 16000)) == 100.0
    assert wav_duration_ms(b"") == 0.0
    assert wav_duration_ms(b"not a wav at all") == 0.0


def test_wav_duration_never_exceeds_physical_payload():
    """A non-seekable WAV writer (espeak-ng --stdout) stamps a placeholder
    data-chunk size (0x7ffff000); trusting it feeds ~13.5 hours per chunk into
    echo holds, sink backlog and calibration."""
    import io
    import wave

    pcm = b"\x01\x00" * 1600  # 100 ms @ 16 kHz
    blob = bytearray(pcm_to_wav_bytes(pcm, 16000))
    # Forge espeak-ng's stdout header: RIFF and data sizes both placeholders.
    assert blob[36:40] == b"data"
    blob[4:8] = (0x7FFFF024).to_bytes(4, "little")
    blob[40:44] = (0x7FFFF000).to_bytes(4, "little")
    with wave.open(io.BytesIO(bytes(blob)), "rb") as w:
        assert w.getnframes() > len(pcm) // 2  # the header really does lie
    assert wav_duration_ms(bytes(blob)) == 100.0


def test_espeak_stdout_wav_is_rewrapped_with_real_sizes():
    from nanobot_channel_voice.tts.system import _rewrap_stdout_wav

    pcm = b"\x01\x00" * 2205  # 100 ms @ 22050 Hz (espeak-ng's native rate)
    blob = bytearray(pcm_to_wav_bytes(pcm, 22050))
    blob[4:8] = (0x7FFFF024).to_bytes(4, "little")
    blob[40:44] = (0x7FFFF000).to_bytes(4, "little")
    fixed = _rewrap_stdout_wav(bytes(blob))
    assert fixed == pcm_to_wav_bytes(pcm, 22050)  # payload kept, sizes healed
    assert _rewrap_stdout_wav(b"garbage") == b"garbage"  # unparseable: unchanged


# ---- on-device TTS shell ----------------------------------------------------

def test_ondevice_tts_shell_split_join_and_degrade():
    np = pytest.importorskip("numpy")
    from nanobot_channel_voice.tts.ondevice_base import OnDeviceTtsAdapter

    class Fake(OnDeviceTtsAdapter):
        output_rate = 1000
        _label = "Fake"
        _join_gap_s = 0.1  # 100 samples at 1 kHz
        pieces: list[str]

        def __init__(self, fail=False):
            self.pieces = []
            self._fail = fail

        def _piece_budget(self) -> int:
            return 5

        def _synthesize_piece(self, text: str):
            if self._fail:
                raise RuntimeError("model exploded")
            self.pieces.append(text)
            return np.ones(10, dtype=np.float32)

    async def _case():
        fake = Fake()
        assert await fake.synthesize("") == b""
        pcm = await fake.synthesize_pcm("abcd efgh")     # two 5-char pieces
        assert fake.pieces == ["abcd", "efgh"]
        # piece + 100-sample gap + piece, S16_LE
        assert len(pcm) == (10 + 100 + 10) * 2
        wav = await fake.synthesize("abc")
        assert wav[:4] == b"RIFF"

        broken = Fake(fail=True)
        assert await broken.synthesize("hello") == b""   # degrade, never raise
        assert await broken.synthesize_pcm("hello") == b""

    asyncio.run(_case())
