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
    busy = False

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
        ch._build_local = lambda: (shell, None, None, [], {})  # type: ignore[method-assign]

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
        ch._build_local = lambda: (shell, None, None, [], {})  # type: ignore[method-assign]

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


def test_a_start_under_a_unified_session_warns_that_turns_cross_channels(monkeypatch):
    """Under ``agents.defaults.unifiedSession`` core folds another channel's message into
    the running voice turn, and a barge-in's /stop cancels theirs: the local start says so."""
    from loguru import logger

    from nanobot_channel_voice import channel as channel_mod

    seen: list[str] = []
    sink = logger.add(lambda m: seen.append(m.record["message"]), level="WARNING")
    try:
        for unified in (False, True):
            monkeypatch.setattr(channel_mod, "unified_session", lambda u=unified: u)
            ch = _channel({"tts": {"enabled": False}}, monkeypatch, lambda cfg: _FakeStt())

            async def run(ch=ch):
                task = await _start_until(ch, lambda: ch._shell is not None)
                await ch.stop()
                await task

            _run(run())
            assert sum("unifiedSession" in m for m in seen) == (1 if unified else 0)
    finally:
        logger.remove(sink)


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


def test_start_failures_reach_the_webui_in_the_channels_own_words(monkeypatch):
    """Core's manager shows start_error_message() in the WebUI's "Failed" box and falls
    back to "Check gateway logs" on None. The refusals start() words for the operator
    (RuntimeError/OSError) go through, collapsed to one line; a bug does not."""
    from nanobot_channel_voice.weights import WeightsError

    ch = _channel({"tts": {"enabled": False}}, monkeypatch, lambda cfg: _FakeStt())
    refusal = RuntimeError("no API key for realtime provider 'xai'\n  (set realtime.apiKey)")
    assert ch.start_error_message(refusal) == (
        "no API key for realtime provider 'xai' (set realtime.apiKey)"
    )
    assert ch.start_error_message(WeightsError("vad/firered/onnx: not fetched")).startswith(
        "vad/firered/onnx"
    )
    assert ch.start_error_message(OSError("address in use")) == "address in use"
    assert ch.start_error_message(RuntimeError("")) is None
    assert ch.start_error_message(AttributeError("'NoneType' has no attribute 'x'")) is None


def test_a_local_start_publishes_what_it_started_without(monkeypatch):
    """The WebUI validator compares it with the config to tell the live channel's state;
    a stop withdraws it."""
    from nanobot_channel_voice import running

    ch = _channel(
        {"tts": {"enabled": False}, "vad": {"engine": "silero"}, "stt": {"provider": "whisper"}},
        monkeypatch, lambda cfg: None,  # the on-device build fell back
    )

    async def run():
        task = await _start_until(ch, lambda: running.current() is not None)
        published = running.current()
        await ch.stop()
        await task
        return published

    published = _run(run())
    assert published is not None and published.config is ch.config
    assert list(published.fell_back) == ["vad", "stt"]
    assert published.fell_back["vad"].unset == ("vad.silero.modelPath",)
    assert running.current() is None


def test_core_sees_the_channel_running_once_its_pipeline_is_up(monkeypatch):
    """Core's status reads is_running: "starting" through the model loads (the WebUI checks
    the setup again when it turns "running", once the fallbacks are published)."""
    from nanobot_channel_voice import running

    loading, may_finish = threading.Event(), threading.Event()

    def make_stt(cfg):
        loading.set()
        may_finish.wait(5)
        return _FakeStt()

    ch = _channel({"tts": {"enabled": False}}, monkeypatch, make_stt)

    async def run():
        task = await _start_until(ch, loading.is_set)
        seen = [ch.is_running]
        may_finish.set()
        for _ in range(500):
            if ch.is_running or task.done():
                break
            await asyncio.sleep(0.01)
        seen += [ch.is_running, running.current() is not None]
        await ch.stop()
        seen.append(ch.is_running)
        await task
        return seen

    assert _run(run()) == [False, True, True, False]


# ---- models that land after the start (models.autoFetch) ---------------------------


class _FakeKeeper:
    """What start() asks of a ModelKeeper, driven by the test: ``landed`` wakes a rebuild,
    ``release`` ends the wait for the models the start cannot do without."""

    def __init__(self, _cfg=None) -> None:
        self.landed = asyncio.Event()
        self.release = asyncio.Event()
        self.why: str | None = None
        self.started = self.closed = self.loads = 0

    def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1

    def loaded(self) -> None:
        self.loads += 1

    async def until(self, ready, keys) -> str | None:
        await self.release.wait()
        return self.why


def _keepers(monkeypatch) -> list[_FakeKeeper]:
    from nanobot_channel_voice import channel as channel_mod

    made: list[_FakeKeeper] = []
    monkeypatch.setattr(channel_mod, "ModelKeeper", lambda cfg: made.append(k := _FakeKeeper()) or k)
    monkeypatch.setattr(channel_mod, "_REST_S", 0.05)
    monkeypatch.setattr(channel_mod, "_REST_POLL_S", 0.01)
    return made


async def _until(condition, timeout_s: float = 5.0) -> None:
    for _ in range(int(timeout_s / 0.01)):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition not met")


def test_models_that_land_rebuild_the_pipeline_in_place_once_at_rest(monkeypatch):
    from nanobot_channel_voice import running

    made = _keepers(monkeypatch)
    builds: list[_FakeStt] = []
    ch = _channel({"tts": {"enabled": False}}, monkeypatch, lambda cfg: builds.append(_FakeStt()) or builds[-1])

    async def run():
        task = await _start_until(ch, lambda: ch.is_running)
        first, keeper = ch._shell, made[0]
        keeper.landed.set()
        await _until(lambda: len(builds) == 2 and ch.is_running)
        assert ch._shell is not first and running.current() is not None
        await asyncio.sleep(0.1)
        assert len(builds) == 2  # one landing, one rebuild
        await ch.stop()
        await task
        return keeper

    keeper = _run(run())
    assert (keeper.started, keeper.loads, keeper.closed) == (1, 2, 1)
    assert [stt.released for stt in builds] == [1, 1]  # the first stack went before the second


def test_a_rebuild_waits_out_an_exchange_and_an_apply(monkeypatch):
    from nanobot_channel_voice.webui_sync import VoiceSyncStore

    made = _keepers(monkeypatch)
    applying = {"on": True}
    monkeypatch.setattr(VoiceSyncStore, "applying", lambda self: applying["on"])
    builds: list[int] = []
    ch = _channel({"tts": {"enabled": False}}, monkeypatch, lambda cfg: builds.append(1) or _FakeStt())

    async def run():
        task = await _start_until(ch, lambda: ch.is_running)
        ch._shell._state = VoiceState.SPEAKING
        made[0].landed.set()
        await asyncio.sleep(0.2)
        assert len(builds) == 1  # mid-reply
        ch._shell._state = VoiceState.IDLE
        await asyncio.sleep(0.2)
        assert len(builds) == 1  # core restarts the channel after an Apply
        applying["on"] = False
        await _until(lambda: len(builds) == 2 and ch.is_running)
        await ch.stop()
        await task

    _run(run())


def test_a_stop_while_waiting_for_rest_rebuilds_nothing(monkeypatch):
    from nanobot_channel_voice import running

    made = _keepers(monkeypatch)
    builds: list[int] = []
    ch = _channel({"tts": {"enabled": False}}, monkeypatch, lambda cfg: builds.append(1) or _FakeStt())

    async def run():
        task = await _start_until(ch, lambda: ch.is_running)
        ch._shell._state = VoiceState.THINKING
        made[0].landed.set()
        await asyncio.sleep(0.1)
        await ch.stop()
        await asyncio.wait_for(task, 5)

    _run(run())
    assert builds == [1] and made[0].closed == 1
    assert ch._shell is None and running.current() is None and not ch.is_running


def test_a_message_landing_mid_rebuild_is_spoken_after_it(monkeypatch):
    """A rebuild has no pipeline for seconds of model loads: a reminder that lands then is
    spoken by the rebuilt one, and a dead turn's straggler is not."""
    from nanobot.bus.events import OutboundMessage

    from nanobot_channel_voice.streamid import TURN_META

    made = _keepers(monkeypatch)
    gate = threading.Event()
    gate.set()
    builds: list[int] = []

    def make_stt(cfg):
        builds.append(1)
        gate.wait(5)
        return _FakeStt()

    ch = _channel({"tts": {"enabled": False}}, monkeypatch, make_stt)
    announced: list[str] = []

    async def announce(self, text):
        announced.append(text)

    from nanobot_channel_voice.backend.local import LocalBackend

    monkeypatch.setattr(LocalBackend, "announce", announce)

    async def run():
        task = await _start_until(ch, lambda: ch.is_running)
        gate.clear()  # the rebuild blocks inside its model load
        made[0].landed.set()
        await _until(lambda: len(builds) == 2 and ch._backend is None)
        await ch.send(OutboundMessage(channel="voice", chat_id=ch.config.chat_id, content="Reminder: stretch"))
        await ch.send(OutboundMessage(
            channel="voice", chat_id=ch.config.chat_id, content="late",
            metadata={TURN_META: "t-dead"},
        ))
        assert announced == []
        gate.set()
        await _until(lambda: ch.is_running)
        await ch.stop()
        await task

    _run(run())
    assert announced == ["Reminder: stretch"]


def test_a_waiting_backend_message_defers_the_rebuild(monkeypatch):
    """A queued notice dies with its backend, so the rebuild waits until it was spoken."""
    made = _keepers(monkeypatch)
    builds: list[int] = []
    ch = _channel({"tts": {"enabled": False}}, monkeypatch, lambda cfg: builds.append(1) or _FakeStt())

    async def run():
        task = await _start_until(ch, lambda: ch.is_running)
        ch._backend._notices.append("waiting")
        made[0].landed.set()
        await asyncio.sleep(0.3)
        assert len(builds) == 1
        ch._backend._notices.clear()
        await _until(lambda: len(builds) == 2 and ch.is_running)
        await ch.stop()
        await task

    _run(run())


def test_the_start_waits_for_the_models_it_cannot_do_without(monkeypatch):
    """Served speech-to-text has no stand-in: core reads Starting while its model is fetched."""
    made = _keepers(monkeypatch)
    section = {**_CLOUD_SERVE, "stt": {**_CLOUD_SERVE["stt"], "whisper": {"weights": "stt/whisper/base/onnx"}}}
    ch = _channel(section, monkeypatch, lambda cfg: _FakeStt())
    shell = _StubShell()

    async def build_cloud(kind):
        return shell, "", []

    ch._build_cloud = build_cloud  # type: ignore[method-assign]

    async def run():
        task = asyncio.create_task(ch.start())
        await _until(lambda: bool(made))
        await asyncio.sleep(0.1)
        assert not ch.is_running and not task.done() and shell.starts == 0
        made[0].release.set()
        await _until(lambda: ch.is_running)
        await ch.stop()
        await task

    _run(run())
    assert shell.starts == 1


def test_a_start_whose_models_will_not_come_says_why(monkeypatch):
    made = _keepers(monkeypatch)
    section = {**_CLOUD_SERVE, "stt": {**_CLOUD_SERVE["stt"], "whisper": {"weights": "stt/whisper/base/onnx"}}}
    ch = _channel(section, monkeypatch, lambda cfg: _FakeStt())

    async def run():
        task = asyncio.create_task(ch.start())
        await _until(lambda: bool(made))
        made[0].why = "the model index does not list stt/whisper/base/onnx"
        made[0].release.set()
        with pytest.raises(RuntimeError) as refused:
            await task
        return str(refused.value)

    assert _run(run()) == (
        "the channel does not start without stt/whisper/base/onnx: the model index does not "
        "list stt/whisper/base/onnx"
    )
    assert made[0].closed == 1


def test_without_auto_fetch_the_channel_keeps_no_models(monkeypatch):
    made = _keepers(monkeypatch)
    ch = _channel({"tts": {"enabled": False}, "models": {"autoFetch": False}}, monkeypatch, lambda cfg: _FakeStt())

    async def run():
        task = await _start_until(ch, lambda: ch.is_running)
        await ch.stop()
        await task

    _run(run())
    assert made == []


def test_a_model_the_channel_fetches_itself_is_in_the_rebuilt_pipeline(store, tmp_path, monkeypatch):
    """The real keeper, from a file index: the landing, the rebuild, the model in use."""
    import hashlib
    import json
    import time

    from nanobot_channel_voice import channel as channel_mod
    from nanobot_channel_voice import running
    from nanobot_channel_voice import weights as w

    monkeypatch.setattr(channel_mod, "_REST_S", 0.05)
    monkeypatch.setattr(channel_mod, "_REST_POLL_S", 0.01)
    blob = tmp_path / "model.onnx"
    blob.write_bytes(b"not a real model")
    entry = {"files": {"model.onnx": {"url": blob.as_uri(), "sha256": hashlib.sha256(blob.read_bytes()).hexdigest()}}}
    store.mkdir(parents=True)
    (store / w.INDEX_CACHE).write_text(json.dumps({
        "fetched_unix": int(time.time()), "sources": list(w.DEFAULT_INDEX_SOURCES),
        "models": {"vad/silero/v6/onnx": entry},
    }))
    builds: list[int] = []
    ch = _channel(
        {"tts": {"enabled": False}, "vad": {"engine": "silero", "silero": {"weights": "vad/silero/v6/onnx"}}},
        monkeypatch, lambda cfg: builds.append(1) or _FakeStt(),
    )

    async def run():
        task = await _start_until(ch, lambda: len(builds) == 2 and ch.is_running)
        published = running.current()
        await ch.stop()
        await task
        return published

    published = _run(run())
    assert "vad/silero/v6/onnx" in w.installed(store) and len(builds) == 2
    # the rebuild resolved the landed files (this blob does not load, so Energy stands in)
    assert published.fell_back["vad"] is None
