"""VoiceShell: the edge contracts the backends rely on.

Barge-in ORDERING (flush first, its return value is the played-ms handed to the
backend), the mic gate, tool-outcome classification, and the late-event gate.
"""

from __future__ import annotations

import asyncio

from nanobot_channel_voice.audio.base import CaptureSource
from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import (
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    StateHint,
    ToolCall,
    TurnDone,
    VoiceState,
)
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.shell import VoiceShell


class _StubCapture(CaptureSource):
    def __init__(self, frames: list[bytes]) -> None:
        self._frames = list(frames)
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.starts += 1

    async def read_frame(self) -> bytes:
        return self._frames.pop(0) if self._frames else b""

    async def stop(self) -> None:
        self.stops += 1


class _StubBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.raise_on_barge_in = False

    async def start(self, *, instructions, tools, on_event) -> None:
        self.calls.append(("start", None))
        self.on_event = on_event

    async def push_audio(self, pcm: bytes) -> None:
        self.calls.append(("push", pcm))

    async def barge_in(self, played_ms: int) -> None:
        self.calls.append(("barge_in", played_ms))
        if self.raise_on_barge_in:
            raise RuntimeError("socket is gone")

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        self.calls.append(("result", (call_id, output)))

    async def close(self) -> None:
        self.calls.append(("close", None))


def _shell(**kw) -> tuple[VoiceShell, _StubBackend, AudioSink]:
    backend = _StubBackend()
    sink = AudioSink(NullPlayback(), mode="stream")
    shell = VoiceShell(
        VoiceConfig(),
        capture=kw.pop("capture", _StubCapture([])),
        sink=sink,
        backend=backend,
        open_mic=kw.pop("open_mic", False),
        **kw,
    )
    return shell, backend, sink


def _run(coro):
    return asyncio.run(coro)


# ---- barge-in ordering ------------------------------------------------------

def test_cloud_barge_in_hands_the_backend_the_flush_result():
    """played_ms must come FROM the flush, not from a separate read: reading it
    after the flush yields 0 and mis-truncates the model's memory."""
    order: list[str] = []

    async def _case():
        shell, backend, sink = _shell()
        await sink.start()
        sink.enqueue(OutputAudio(epoch=sink.epoch, pcm=b"\x00" * 6400, rate=16000))
        await sink.wait_idle()
        await asyncio.sleep(0.05)  # let some of it become "heard"

        real_flush = sink.flush

        async def spy_flush():
            order.append("flush")
            return await real_flush()

        sink.flush = spy_flush

        async def on_barge_in():
            order.append("abandon")

        shell._on_barge_in = on_barge_in
        await shell._cloud_barge_in()
        await sink.stop()
        return backend

    backend = _run(_case())
    kinds = [c[0] for c in backend.calls]
    assert kinds == ["barge_in"]
    assert order == ["flush", "abandon"]  # flush -> backend -> abandon
    (_, played_ms), = [c for c in backend.calls if c[0] == "barge_in"]
    assert played_ms > 0  # a real number, not the post-flush zero


def test_delegation_is_abandoned_even_if_the_wire_step_fails():
    abandoned: list[bool] = []

    async def _case():
        shell, backend, sink = _shell()
        backend.raise_on_barge_in = True

        async def on_barge_in():
            abandoned.append(True)

        shell._on_barge_in = on_barge_in
        try:
            await shell._cloud_barge_in()
        except RuntimeError:
            pass

    _run(_case())
    assert abandoned == [True]


# ---- mic gate ---------------------------------------------------------------

def test_mic_gate_follows_state_only_when_the_mic_is_not_open():
    shell, _, _ = _shell(open_mic=False)
    shell._state = VoiceState.SPEAKING
    assert shell._mic_gated() is True
    shell._state = VoiceState.THINKING
    assert shell._mic_gated() is False

    open_shell, _, _ = _shell(open_mic=True)
    open_shell._state = VoiceState.SPEAKING
    assert open_shell._mic_gated() is False


def test_state_is_mirrored_only_from_state_hints():
    async def _case():
        shell, _, _ = _shell()
        await shell._on_event(StateHint(VoiceState.THINKING))
        assert shell.state is VoiceState.THINKING
        # The negative half: observational events must not move the mirror.
        await shell._on_event(OutputTranscript("hi"))
        await shell._on_event(InputTranscript("hello"))
        await shell._on_event(TurnDone())
        assert shell.state is VoiceState.THINKING

    _run(_case())


def test_events_after_teardown_are_dropped():
    """The backend's rx loop can deliver a last few events between the tool-task
    sweep and close(); acting on one would spawn work nothing will cancel."""
    shell, _, _ = _shell()
    shell._stopped = True
    _run(shell._on_event(ToolCall(call_id="c1", name="x", arguments="{}")))
    assert shell._tool_tasks == set()
    _run(shell._on_event(StateHint(VoiceState.SPEAKING)))
    assert shell.state is VoiceState.IDLE


# ---- tool outcome classification -------------------------------------------

class _ErrResult(str):
    is_error = True


def test_tool_failure_is_classified_from_is_error_not_from_exceptions():
    """nanobot's ToolRegistry never raises for a tool failure: it returns a
    ToolResult (a str subclass) with is_error set. Keying success off exceptions
    would report ~0% failures forever."""
    async def exec_tool(name, args):
        return _ErrResult("boom")

    async def _case():
        shell, backend, _ = _shell(exec_tool=exec_tool)
        await shell._on_tool_call(ToolCall(call_id="c1", name="t", arguments="{}"))
        return shell, backend

    shell, backend = _run(_case())
    assert shell._metrics.counters.get("tool_error") == 1
    assert ("result", ("c1", "boom")) in backend.calls


def test_tool_exception_is_reported_back_to_the_model():
    async def exec_tool(name, args):
        raise ValueError("nope")

    async def _case():
        shell, backend, _ = _shell(exec_tool=exec_tool)
        await shell._on_tool_call(ToolCall(call_id="c2", name="t", arguments="{}"))
        return shell, backend

    shell, backend = _run(_case())
    assert shell._metrics.counters.get("tool_exception") == 1
    call_id, output = [c[1] for c in backend.calls if c[0] == "result"][0]
    assert call_id == "c2"
    assert "nope" in output


def test_missing_tool_seam_answers_the_model_instead_of_hanging():
    async def _case():
        shell, backend, _ = _shell(exec_tool=None)
        await shell._on_tool_call(ToolCall(call_id="c3", name="t", arguments="{}"))
        return shell, backend

    shell, backend = _run(_case())
    assert shell._metrics.counters.get("tool_no_seam") == 1
    assert [c for c in backend.calls if c[0] == "result"]


# ---- capture pump -----------------------------------------------------------

def test_gate_reopen_drops_the_in_hand_frame_and_flushes_the_backlog():
    """Half-duplex: the gate is applied at READ time, so under lag frames captured
    while it was closed (the bot audible) are still buffered when it reopens:
    released, they reach the VAD as fresh speech and endpoint as echo blobs. The
    pump must drop the frame in hand at the edge and flush the source's backlog
    before live audio flows again."""
    shell_box: list[VoiceShell] = []

    class _EdgeCapture(CaptureSource):
        def __init__(self) -> None:
            self.flushes = 0
            self._i = 0

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

        async def read_frame(self) -> bytes:
            self._i += 1
            if self._i == 1:
                return b"A"  # gate open: must reach the backend
            if self._i == 2:
                shell_box[0]._state = VoiceState.SPEAKING
                return b"B"  # gated: dropped
            if self._i == 3:
                shell_box[0]._state = VoiceState.IDLE
                return b"C"  # in hand at the reopen edge: dropped, flush runs
            if self._i == 4:
                return b"D"  # live edge: must reach the backend
            shell_box[0]._running = False
            return b""

        async def flush(self) -> int:
            self.flushes += 1
            return 640  # pretend a backlog existed

    async def _case():
        capture = _EdgeCapture()
        shell, backend, _ = _shell(capture=capture, open_mic=False)
        shell_box.append(shell)
        shell._running = True
        await asyncio.wait_for(shell._capture_loop(), timeout=5.0)
        return capture, shell, backend

    capture, shell, backend = _run(_case())
    assert [c[1] for c in backend.calls if c[0] == "push"] == [b"A", b"D"]
    assert capture.flushes == 1
    assert shell._metrics.counters.get("capture_gate_flush") == 1


def test_capture_that_never_comes_back_escalates_to_a_fatal_stop():
    """Permanent deafness is fatal, not a degraded mode: the session must not
    keep playing audio while presenting healthy."""
    fatal: list[bool] = []

    async def on_fatal():
        fatal.append(True)

    async def _case():
        capture = _StubCapture([])  # every read is EOF
        shell, backend, sink = _shell(capture=capture, on_fatal=on_fatal)
        shell._running = True
        # 5 EOFs per restart cycle, 3 restarts, then fatal: with the sleeps
        # patched out so the test does not pay 3 x 2 s.
        real_sleep = asyncio.sleep

        async def fast_sleep(delay, *a, **kw):
            return await real_sleep(0)

        asyncio.sleep = fast_sleep
        try:
            # Bounded: if the restart cap regresses, fail red instead of
            # busy-spinning the whole suite on the zero-delay sleep.
            await asyncio.wait_for(shell._capture_loop(), timeout=5.0)
            for _ in range(20):
                await real_sleep(0)
        finally:
            asyncio.sleep = real_sleep
        return capture

    capture = _run(_case())
    assert capture.starts == 3  # bounded restarts, then give up
    assert fatal == [True]


def test_capture_restart_tells_the_backend_about_the_gap():
    """Each restart must fire the backend's gap hook (an utterance open at mic
    death would otherwise bridge the outage: no frames means no silence run)."""
    gaps: list[bool] = []

    class _GapBackend(_StubBackend):
        async def on_capture_gap(self) -> None:
            gaps.append(True)

    async def _case():
        capture = _StubCapture([])  # every read is EOF
        shell, backend, sink = _shell(capture=capture, on_fatal=None)
        gap_backend = _GapBackend()
        shell._backend = gap_backend
        shell._running = True
        real_sleep = asyncio.sleep

        async def fast_sleep(delay, *a, **kw):
            return await real_sleep(0)

        asyncio.sleep = fast_sleep
        try:
            await asyncio.wait_for(shell._capture_loop(), timeout=5.0)
        finally:
            asyncio.sleep = real_sleep

    _run(_case())
    assert gaps == [True, True, True]  # one per bounded restart
