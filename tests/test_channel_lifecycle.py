"""VoiceChannel start()/stop(): who owns the shell, the engines and the STT adapter across
a raced stop, a failed start, and a start cancelled mid-load (core stops a channel with
stop() then task.cancel(); a thread load cannot be interrupted)."""

from __future__ import annotations

import asyncio
import threading

import pytest

from nanobot_channel_voice.backend.base import VoiceState
from nanobot_channel_voice.config import VoiceConfig


def _run(coro):
    return asyncio.run(coro)



class _FakeStt:
    max_decode_ms = None
    streaming = False

    def __init__(self) -> None:
        self.released = 0

    async def transcribe(self, pcm: bytes, rate: int) -> str:
        return ""

    async def warmup(self) -> None:
        pass

    def release(self) -> None:
        self.released += 1


class _StubShell:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self.state = VoiceState.IDLE

    async def start(self, *, instructions, tools) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1


def _channel(section: dict, monkeypatch, make_stt):
    from nanobot.bus.queue import MessageBus

    from nanobot_channel_voice import channel as channel_mod
    from nanobot_channel_voice.channel import VoiceChannel

    monkeypatch.setattr(channel_mod, "make_stt", make_stt)
    cfg = VoiceConfig.model_validate({"audio": {"backend": "null"}, **section})
    return VoiceChannel(cfg, MessageBus())


_CLOUD_SERVE = {
    "backend": "openai",
    "realtime": {"apiKey": "k"},
    "stt": {"provider": "whisper", "serve": {"enabled": True, "port": 0}},
}


async def _start_until(ch, ready, *, timeout_s: float = 5.0):
    task = asyncio.create_task(ch.start())
    for _ in range(int(timeout_s / 0.01)):
        if ready() or task.done():
            break
        await asyncio.sleep(0.01)
    assert ready() or task.done()
    return task


def test_serve_side_stt_loads_off_the_event_loop(monkeypatch):
    """cloud backend + stt.serve is the one path that built the adapter inline: a model
    load there froze every other channel, like the local path's used to."""
    adapter = _FakeStt()
    loaded_on: list[threading.Thread] = []

    def make_stt(cfg):
        loaded_on.append(threading.current_thread())
        return adapter

    ch = _channel(_CLOUD_SERVE, monkeypatch, make_stt)
    shell = _StubShell()

    async def build_cloud(kind):
        return shell, "", []

    ch._build_cloud = build_cloud  # type: ignore[method-assign]

    async def run():
        task = await _start_until(ch, lambda: ch._stt_server is not None)
        await ch.stop()
        await task

    _run(run())
    assert loaded_on and loaded_on[0] is not threading.main_thread()
    assert adapter.released == 1 and shell.stops == 1


def test_stop_landing_during_the_stt_load_still_releases_it(monkeypatch):
    """stop() during the threaded make_stt sees no adapter yet; the frame that finishes
    the load owns it (ORT/RKNN sessions otherwise live until GC)."""
    adapter = _FakeStt()
    loading, may_finish = threading.Event(), threading.Event()

    def make_stt(cfg):
        loading.set()
        may_finish.wait(5)
        return adapter

    for section in ({"tts": {"enabled": False}}, _CLOUD_SERVE):
        adapter.released = 0
        loading.clear()
        may_finish.clear()
        ch = _channel(section, monkeypatch, make_stt)
        shell = _StubShell()
        ch._build_local = lambda: (shell, None, None, [])  # type: ignore[method-assign]

        async def build_cloud(kind):
            return shell, "", []

        ch._build_cloud = build_cloud  # type: ignore[method-assign]

        async def run():
            task = await _start_until(ch, loading.is_set)
            await ch.stop()  # nothing to release yet: _stt is still None here
            may_finish.set()
            await task  # start() returns via its "stop() raced" exits

        _run(run())
        assert adapter.released == 1, section
        assert ch._stt is None and ch._stt_server is None and ch._shell is None
        assert shell.stops >= 1, section  # a built shell's engines are freed too


def test_a_start_cancelled_mid_load_frees_the_adapter_that_lands_late(monkeypatch):
    """Core's _stop_channel is stop() then task.cancel(): a cancel landing while make_stt
    runs in its thread leaves no adapter to release in the frame; the one that lands
    after must free itself (RKNN: an NPU context no GC frees)."""
    adapter = _FakeStt()
    loading, may_finish = threading.Event(), threading.Event()

    def make_stt(cfg):
        loading.set()
        may_finish.wait(5)
        return adapter

    for section in ({"tts": {"enabled": False}}, _CLOUD_SERVE):
        adapter.released = 0
        loading.clear()
        may_finish.clear()
        ch = _channel(section, monkeypatch, make_stt)
        shell = _StubShell()
        ch._build_local = lambda: (shell, None, None, [])  # type: ignore[method-assign]

        async def build_cloud(kind):
            return shell, "", []

        ch._build_cloud = build_cloud  # type: ignore[method-assign]

        async def run():
            task = await _start_until(ch, loading.is_set)
            await ch.stop()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            may_finish.set()  # the load lands in a dead frame
            for _ in range(100):
                if adapter.released:
                    break
                await asyncio.sleep(0.01)

        _run(run())
        assert adapter.released == 1, section
        assert ch._stt is None


def test_a_failed_start_releases_the_loaded_stt(monkeypatch):
    """A build or bind failure after the load re-raises AND frees the adapter, for both
    the local build and the serve endpoint."""
    from nanobot_channel_voice.stt import serve as serve_mod

    adapter = _FakeStt()
    ch = _channel({"tts": {"enabled": False}}, monkeypatch, lambda cfg: adapter)

    def broken_build():
        raise RuntimeError("no such device")

    ch._build_local = broken_build  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="no such device"):
        _run(ch.start())
    assert adapter.released == 1 and ch._stt is None

    adapter.released = 0
    ch = _channel(_CLOUD_SERVE, monkeypatch, lambda cfg: adapter)
    shell = _StubShell()

    async def build_cloud(kind):
        return shell, "", []

    ch._build_cloud = build_cloud  # type: ignore[method-assign]

    async def cannot_bind(self):
        raise OSError("address in use")

    monkeypatch.setattr(serve_mod.SttHttpServer, "start", cannot_bind)
    with pytest.raises(OSError, match="address in use"):
        _run(ch.start())
    assert adapter.released == 1 and ch._stt is None and shell.stops == 1


def test_a_restart_waits_for_the_load_it_cancelled_and_registers_no_dead_bridge(monkeypatch):
    """A WebUI save stops, cancels and restarts the channel while the old build thread
    still loads: the fresh instance waits for the orphan to land (no second model copy)
    and the orphan publishes nothing (the registry keeps the fresh bridge)."""
    from nanobot_channel_voice import channel as channel_mod
    from nanobot_channel_voice import context_tool

    old = _channel({"tts": {"enabled": False}}, monkeypatch, lambda cfg: _FakeStt())
    new = _channel({"tts": {"enabled": False}}, monkeypatch, lambda cfg: _FakeStt())
    loading, may_finish = threading.Event(), threading.Event()
    real_build = old._build_local

    def slow_build():
        loading.set()
        may_finish.wait(5)
        return real_build()

    old._build_local = slow_build  # type: ignore[method-assign]
    key = ("voice", old.config.chat_id)

    async def run():
        old_task = await _start_until(old, loading.is_set)
        await old.stop()
        old_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await old_task
        new_task = asyncio.create_task(new.start())
        await asyncio.sleep(0.2)
        assert new._shell is None and channel_mod._LOAD_LOCK.locked()  # queued behind the orphan
        may_finish.set()
        for _ in range(500):
            if new._shell is not None or new_task.done():
                break
            await asyncio.sleep(0.01)
        assert new._shell is not None
        assert context_tool._BRIDGES.get(key) is new._context_bridge
        assert old._context_bridge is None and old._backend is None
        await new.stop()
        await new_task
        assert context_tool._BRIDGES.get(key) is None

    _run(run())


def test_a_cancelled_load_frees_in_the_thread_and_leaves_no_task_behind(monkeypatch):
    """The late result is freed by the loading thread itself (under the load lock) or by
    a synchronous done-callback: no task is created, so a gateway exiting during a
    load never sees a 'Task was destroyed but it is pending' from this path."""
    adapter = _FakeStt()
    loading, may_finish = threading.Event(), threading.Event()

    def make_stt(cfg):
        loading.set()
        may_finish.wait(5)
        return adapter

    ch = _channel({"tts": {"enabled": False}}, monkeypatch, make_stt)
    tasks_at_close: list = []

    async def run():
        task = await _start_until(ch, loading.is_set)
        await ch.stop()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        may_finish.set()
        for _ in range(100):
            if adapter.released:
                break
            await asyncio.sleep(0.01)
        tasks_at_close.extend(t for t in asyncio.all_tasks() if t is not asyncio.current_task())

    _run(run())
    assert adapter.released == 1 and ch._stt is None
    assert tasks_at_close == []
