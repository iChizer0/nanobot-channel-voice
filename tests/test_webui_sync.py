"""The WebUI's model flow: the cached index behind the form's Model chips, the sync plan
(fetch what the section names, remove only what this flow installed), and the connector
that runs it on core's start/poll/cancel seam with progress. HTTP downloads go to a local
server so progress, cancellation and partial cleanup are the real code paths."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nanobot_channel_voice import weights as w
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.sync import config_weights_keys, plan_sync, run_sync
from nanobot_channel_voice.sync import used_weights_keys as _used_weights_keys
from nanobot_channel_voice.webui_form import build_form
from nanobot_channel_voice.webui_sync import MANAGED_BY, VoiceSyncStore


def used_weights_keys(section):
    """The keys a section's setup runs; sync asks it of a validated config."""
    from nanobot_channel_voice.config import VoiceConfig

    return _used_weights_keys(VoiceConfig.model_validate(section))


class _Blobs(BaseHTTPRequestHandler):
    """Serves ``/<name>`` from ``server.blobs`` in 64 KiB chunks, slowly when asked, so a
    poll mid-download sees a partial byte count and a cancel lands between chunks."""

    def do_GET(self):  # noqa: N802 - http.server API
        blob = self.server.blobs.get(self.path.lstrip("/"))
        if blob is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        for i in range(0, len(blob), 1 << 16):
            self.wfile.write(blob[i:i + (1 << 16)])
            self.wfile.flush()
            time.sleep(self.server.delay)

    def log_message(self, *_args):
        pass


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Blobs)
    httpd.blobs = {}
    httpd.delay = 0.0
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    httpd.base = f"http://127.0.0.1:{httpd.server_port}"
    yield httpd
    httpd.shutdown()


def _index(server, **models):
    """``key=(name, blob)`` entries served by ``server``; sizes declared for progress."""
    out = {}
    for key, (name, blob) in models.items():
        server.blobs[name] = blob
        out[key] = {
            "files": {name.split("-")[-1]: {
                "url": f"{server.base}/{name}", "sha256": hashlib.sha256(blob).hexdigest(),
                "size": len(blob),
            }},
            "langs": ["en"], "license": "MIT",
        }
    return out


def _write_config(tmp_path, monkeypatch, section):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"channels": {"voice": section}}))
    import nanobot.config.loader as loader

    monkeypatch.setattr(loader, "get_config_path", lambda: path)
    return path


# ---- index cache + form choices ---------------------------------------------


def test_cli_loads_cache_the_index_and_the_form_reads_only_the_cache(store, tmp_path, capsys):
    from nanobot_channel_voice.cli import main as cli_main

    index_file = tmp_path / "index.json"
    index_file.write_text(json.dumps({"version": 1, "models": {
        "stt/whisper/base/onnx": {"files": {"encoder.onnx": {"url": "https://x/e", "sha256": "0" * 64, "size": 7}}},
        "stt/whisper/base/rknn.rv1126b": {"files": {"encoder.rknn": {"url": "https://x/r", "sha256": "0" * 64}}},
        "stt/whisper/small/onnx": {"files": {"encoder.onnx": {"url": "https://x/s", "sha256": "0" * 64, "size": 9}},
                                   "license": "MIT", "accept": "research only"},
        "tts/matcha/en-US/ljspeech/onnx": {"files": {"decoder.onnx": {"url": "https://x/d", "sha256": "0" * 64}}},
    }}))
    assert w.cached_index(store) is None
    assert cli_main(["--index", str(index_file), "list"]) == 0
    models, fetched_unix, sources = w.cached_index(store)
    assert set(models) == {"stt/whisper/base/onnx", "stt/whisper/base/rknn.rv1126b",
                           "stt/whisper/small/onnx", "tts/matcha/en-US/ljspeech/onnx"}
    assert fetched_unix > 0 and sources == [str(index_file)]

    fields = {f["key"]: f for s in build_form(VoiceConfig.model_validate(
        {"stt": {"provider": "whisper"}, "tts": {"provider": "matcha"}}))["sections"] for f in s["fields"]}
    model = fields["stt.whisper.weights"]
    assert model["kind"] == "weights" and model["label"] == "Model"
    # only this host's platform, labelled by what tells the entries apart
    assert model["choices"] == [
        {"value": "stt/whisper/base/onnx", "label": "base", "installed": False, "bytes": 7},
        {"value": "stt/whisper/small/onnx", "label": "small", "installed": False, "bytes": 9,
         "license": "MIT", "notice": "research only"},
    ]
    assert fields["tts.matcha.weights"]["choices"][0]["label"] == "en-US/ljspeech"
    assert "Apply downloads" in model["help"]
    # without a cache the field is the bare key input
    (store / w.INDEX_CACHE).unlink()
    fields = {f["key"]: f for s in build_form(VoiceConfig.model_validate(
        {"stt": {"provider": "whisper"}}))["sections"] for f in s["fields"]}
    assert fields["stt.whisper.weights"]["choices"] == []
    assert "No model index is cached" in fields["stt.whisper.weights"]["help"]


def test_one_pill_per_stem_with_the_builds_as_a_second_level(store):
    """A stem with a CPU and a device build is one pill whose value is the device build
    (the default) and whose ``builds`` list both, CPU first; a build for another chip is
    not offered; an installed key shows without an index; wake pills read by phrase."""
    (store / w.INDEX_CACHE).parent.mkdir(parents=True)
    (store / w.INDEX_CACHE).write_text(json.dumps({"fetched_unix": 1, "models": {
        "wake/openwakeword/alexa/onnx": {"files": {"m.onnx": {"url": "https://x/a", "sha256": "0" * 64, "size": 3}}},
        "wake/openwakeword/alexa/rknn.rv1126b": {"files": {"m.rknn": {"url": "https://x/b", "sha256": "0" * 64, "size": 2}}},
        "wake/openwakeword/alexa/rknn.rk3588": {"files": {"m.rknn": {"url": "https://x/c", "sha256": "0" * 64}}},
        "vad/smartturn/v3.2/rknn.rv1126b": {"files": {"m.rknn": {"url": "https://x/t", "sha256": "0" * 64}}},
    }}))
    d = w.store_dir("wake/openwakeword/hey-house/onnx", store)
    d.mkdir(parents=True)
    (d / w.MANIFEST).write_text(json.dumps({"key": "wake/openwakeword/hey-house/onnx", "files": {}}))
    fields = {f["key"]: f for s in build_form(VoiceConfig.model_validate({
        "device": "rv1126b", "wake": {"mode": "gate", "phrases": ["alexa"], "engine": "openwakeword"},
        "vad": {"turn": {"engine": "smartturn"}},
    }))["sections"] for f in s["fields"]}
    choices = fields["wake.openwakeword.weights"]["choices"]
    assert choices[0] == {"value": "", "label": "Transcript", "sets": {"wake.engine": "text"}}
    assert [(c["value"], c["label"], c["installed"]) for c in choices[1:]] == [
        ("wake/openwakeword/alexa/rknn.rv1126b", "alexa", False),
        ("wake/openwakeword/hey-house/onnx", "hey-house", True),  # no meta, no index: the stem
    ]
    assert [(b["value"], b["label"], b["bytes"]) for b in choices[1]["builds"]] == [
        ("wake/openwakeword/alexa/onnx", "CPU", 3), ("wake/openwakeword/alexa/rknn.rv1126b", "RV1126B", 2),
    ]
    assert "builds" not in choices[2]
    # the turn block is vad.turn in the config but vad/smartturn/ in the store
    assert [(c["value"], c["label"]) for c in fields["vad.turn.weights"]["choices"]] == [
        ("vad/smartturn/v3.2/rknn.rv1126b", "v3.2"),
    ]


def test_host_platforms_follow_the_device_alone():
    """One definition: CPU builds always, the RKNN build for the SoC the section names."""
    assert w.host_platforms(None) == ("onnx",)
    assert w.host_platforms("rv1126b") == ("onnx", "rknn.rv1126b")


def test_device_drives_the_form_and_every_blocks_rknn_target(store):
    """One ``device`` in the section: the Model pills offer its builds, every engine block's
    unset ``target`` resolves to it, an explicit block target still wins, and the dump
    stays as written."""
    _cache(store, {
        "stt/whisper/base/onnx": {"files": {"e.onnx": {"url": "https://x/e", "sha256": "0" * 64}}},
        "stt/whisper/base/rknn.rv1126b": {"files": {"e.rknn": {"url": "https://x/r", "sha256": "0" * 64}}},
        "stt/whisper/base/rknn.rk3588": {"files": {"e.rknn": {"url": "https://x/k", "sha256": "0" * 64}}},
    })
    section = {"device": "RV1126B", "stt": {"provider": "whisper"}, "vad": {"engine": "silero", "silero": {"target": "rk3588"}}}
    cfg = VoiceConfig.model_validate(section)
    assert cfg.device == "rv1126b"
    assert cfg.stt.whisper.resolved_target == "rv1126b" and cfg.stt.whisper.target is None
    assert cfg.vad.silero.resolved_target == "rk3588"  # the block's own target wins
    assert cfg.tts.matcha.resolved_target == "rv1126b"
    assert "device" not in cfg.stt.whisper.model_dump(exclude_unset=True)
    fields = {f["key"]: f for s in build_form(cfg)["sections"] for f in s["fields"]}
    assert [c["value"] for c in fields["stt.whisper.weights"]["choices"]] == ["stt/whisper/base/rknn.rv1126b"]
    assert [b["value"] for b in fields["stt.whisper.weights"]["choices"][0]["builds"]] == [
        "stt/whisper/base/onnx", "stt/whisper/base/rknn.rv1126b",
    ]
    assert fields["device"]["value"] == "rv1126b" and fields["device"]["kind"] == "string"
    # no device: CPU builds only, and nothing is probed on the host
    cfg = VoiceConfig.model_validate({"stt": {"provider": "whisper"}})
    assert cfg.device is None and cfg.stt.whisper.resolved_target is None
    fields = {f["key"]: f for s in build_form(cfg)["sections"] for f in s["fields"]}
    assert "value" not in fields["device"]
    assert [c["value"] for c in fields["stt.whisper.weights"]["choices"]] == ["stt/whisper/base/onnx"]
    # a hardware fact, advanced everywhere; its help follows the on-device models: a cloud
    # backend says it is not in use until a gated uplink or a served engine runs them
    def device(section):
        return next(f for s in build_form(VoiceConfig.model_validate(section))["sections"] for f in s["fields"] if f["key"] == "device")

    assert device({})["advanced"] is True and "Not in use" not in device({})["help"]
    assert device({"backend": "openai"})["advanced"] is True
    assert device({"backend": "openai"})["help"].endswith(" Not in use until an on-device detector or a served engine runs.")
    for section in (
        {"backend": "openai", "realtime": {"uplink": "vad"}, "vad": {"engine": "silero"}},
        {"backend": "openai", "stt": {"provider": "whisper", "serve": {"enabled": True}}},
    ):
        assert "Not in use" not in device(section)["help"]
    with pytest.raises(Exception, match="device must be a SoC name"):
        VoiceConfig.model_validate({"device": "rv1126b-evb"})


def test_the_pane_reflects_the_indexs_devices_and_languages(store):
    """The published index is mostly RKNN builds with per-model languages: Device stays a
    typed name whose help lists the chips the index has builds for, a filtered-out model
    says which chip has it, and every pill carries its languages."""
    def fields(section):
        return {f["key"]: f for s in build_form(VoiceConfig.model_validate(section))["sections"] for f in s["fields"]}

    _cache(store, {
        "stt/whisper/base/rknn.rv1126b": {"langs": ["en", "zh", "ja", "de"], "files": {"e.rknn": {"url": "https://x/r", "sha256": "0" * 64, "size": 5}}},
        "stt/zipformer/zh-en/rknn.rv1126b": {"langs": ["zh", "en"], "files": {"e.rknn": {"url": "https://x/z", "sha256": "0" * 64}}},
        "tts/mms/en/rknn.rk3588": {"langs": ["en"], "files": {"d.rknn": {"url": "https://x/m", "sha256": "0" * 64}}},
        "wake/openwakeword/alexa/onnx": {"langs": ["en"], "files": {"m.onnx": {"url": "https://x/a", "sha256": "0" * 64}}},
    })
    cpu = fields({"stt": {"provider": "whisper"}})
    assert cpu["device"]["kind"] == "string" and "choices" not in cpu["device"] and "value" not in cpu["device"]
    assert cpu["device"]["help"] == (
        "The chip the on-device models are built for, empty for CPU builds only. "
        "The index has builds for `rk3588` and `rv1126b`."
    )
    assert cpu["stt.whisper.weights"]["choices"] == []
    assert cpu["stt.whisper.weights"]["help"] == "The index has this model for `rv1126b` only. Enter it under Device in Advanced to offer it."
    assert fields({"stt": {"provider": "sensevoice"}})["stt.sensevoice.weights"]["help"].startswith("The index has no model for this engine")
    board = fields({"device": "rv1126b", "stt": {"provider": "zipformer"}})
    assert board["device"]["value"] == "rv1126b"
    assert board["stt.zipformer.weights"]["choices"] == [
        {"value": "stt/zipformer/zh-en/rknn.rv1126b", "label": "zh-en", "installed": False, "bytes": 0, "langs": ["zh", "en"]},
    ]
    assert fields({"device": "rv1126b", "stt": {"provider": "whisper"}})["stt.whisper.weights"]["choices"][0]["langs"] == ["en", "zh", "ja", "de"]
    # a chip the index has nothing for (a typo, a private build) is named as such
    assert fields({"device": "rv1126"})["device"]["help"].endswith("`rk3588` and `rv1126b`, none for `rv1126`.")
    # an index without chip builds, then no index at all
    _cache(store, {"wake/openwakeword/alexa/onnx": {"files": {"m.onnx": {"url": "https://x/a", "sha256": "0" * 64}}}})
    assert fields({})["device"]["help"].endswith("The index has no chip builds.")
    (store / w.INDEX_CACHE).unlink()
    assert fields({"stt": {"provider": "whisper"}})["stt.whisper.weights"]["help"].startswith("No model index is cached")
    assert fields({})["device"]["help"].endswith("No model index is cached, `rv1126b` is one such name.")


# ---- plan ----------------------------------------------------------------------


def test_plan_fetches_the_named_and_removes_only_what_the_flow_installed(store, tmp_path, server):
    index = _index(server, **{
        "stt/whisper/base/onnx": ("whisper-encoder.onnx", b"w" * 100),
        "tts/mms/en/onnx": ("mms-decoder.onnx", b"m" * 50),
        "vad/silero/v6/onnx": ("silero-model.onnx", b"s" * 20),
    })
    w.fetch("tts/mms/en/onnx", index["tts/mms/en/onnx"], root=store, managed_by=MANAGED_BY)
    w.fetch("vad/silero/v6/onnx", index["vad/silero/v6/onnx"], root=store)  # by hand: the user's
    section = {"stt": {"whisper": {"weights": "stt/whisper/base/onnx"}}, "wake": {"openwakeword": {"weights": "wake/x/y/onnx"}}}
    assert config_weights_keys(section) == {"stt/whisper/base/onnx", "wake/x/y/onnx"}

    plan = plan_sync(section, index, store, managed_by=MANAGED_BY)
    assert plan.fetch == ["stt/whisper/base/onnx"] and plan.fetch_bytes == 100
    assert plan.unknown == ["wake/x/y/onnx"]
    assert plan.prune == ["tts/mms/en/onnx"] and plan.prune_bytes > 0  # silero stays: not ours
    assert plan.sizes == {"stt/whisper/base/onnx": 100, "tts/mms/en/onnx": plan.prune_bytes}
    assert plan.free_bytes > 0 and plan.notices == {}
    # a store not created yet still knows its volume's free space
    assert plan_sync(section, index, tmp_path / "new" / "deep" / "store", managed_by=MANAGED_BY).free_bytes > 0
    # the CLI's --prune removes everything unnamed, whoever fetched it
    assert plan_sync(section, index, store, managed_by=None).prune == ["tts/mms/en/onnx", "vad/silero/v6/onnx"]
    assert plan_sync(section, index, store, managed_by=None, prune=False).prune == []

    fetched, freed = run_sync(plan, index, store, managed_by=MANAGED_BY)
    assert fetched == 100 and freed > 0
    assert set(w.installed(store)) == {"stt/whisper/base/onnx", "vad/silero/v6/onnx"}
    assert w.managed_by("stt/whisper/base/onnx", store) == MANAGED_BY
    assert w.managed_by("vad/silero/v6/onnx", store) is None


def test_the_panel_plans_what_the_resolved_setup_runs(store, server):
    """An engine pick leaves the previous engine's ``weights`` in the section; Apply
    follows the selected engines (a served or gate-run block under a cloud backend), so
    the plan matches the identity line, and a model the setup no longer runs is removed
    like one it no longer names. The CLI's ``sync`` keeps every named key."""
    index = _index(server, **{
        "stt/whisper/base/onnx": ("whisper-encoder.onnx", b"w" * 100),
        "stt/zipformer/zh-en/onnx": ("zip-encoder.onnx", b"z" * 60),
        "vad/silero/v6/onnx": ("silero-model.onnx", b"s" * 20),
        "tts/mms/en/onnx": ("mms-decoder.onnx", b"m" * 50),
    })
    w.fetch("stt/whisper/base/onnx", index["stt/whisper/base/onnx"], root=store, managed_by=MANAGED_BY)
    section = {
        "stt": {"provider": "zipformer", "whisper": {"weights": "stt/whisper/base/onnx"}, "zipformer": {"weights": "stt/zipformer/zh-en/onnx"}},
        "vad": {"engine": "energy", "silero": {"weights": "vad/silero/v6/onnx"}},
        "tts": {"enabled": False, "provider": "mms", "mms": {"weights": "tts/mms/en/onnx"}},
    }
    assert used_weights_keys(section) == {"stt/zipformer/zh-en/onnx"}
    plan = plan_sync(section, index, store, managed_by=MANAGED_BY, used_only=True)
    assert plan.fetch == ["stt/zipformer/zh-en/onnx"] and plan.prune == ["stt/whisper/base/onnx"]
    named = plan_sync(section, index, store, managed_by=MANAGED_BY)
    assert named.fetch == ["stt/zipformer/zh-en/onnx", "tts/mms/en/onnx", "vad/silero/v6/onnx"] and named.prune == []
    # what a block runs follows its switches: the voice on, the detector picked
    section["tts"]["enabled"] = True
    section["vad"]["engine"] = "silero"
    assert used_weights_keys(section) == {"stt/zipformer/zh-en/onnx", "tts/mms/en/onnx", "vad/silero/v6/onnx"}
    # a cloud backend runs none of them, unless the gate (its detectors) or serving (STT) does
    cloud = {**section, "backend": "openai"}
    assert used_weights_keys(cloud) == set()
    assert used_weights_keys({**cloud, "realtime": {"uplink": "vad"}}) == {"vad/silero/v6/onnx"}
    cloud["stt"] = {**cloud["stt"], "serve": {"enabled": True}}
    assert used_weights_keys(cloud) == {"stt/zipformer/zh-en/onnx"}
    # a section the schema refuses still plans every key it names
    refused = plan_sync({**section, "backend": "nova"}, index, store, managed_by=MANAGED_BY, used_only=True)
    assert refused.wanted == sorted(config_weights_keys(section))


# ---- connector -----------------------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


async def _reloaded(connector):
    """The plan after the reload an open asks for has landed."""
    await connector.handle("start", {"plan": ["true"], "refresh": ["true"]})
    await connector._reload
    return await connector.handle("start", {"plan": ["true"]})


async def _drive(connector, session_id, *, until=("succeeded", "failed", "cancelled")):
    for _ in range(200):
        payload = await connector.handle("poll", {"session_id": [session_id]})
        if payload["status"] in until:
            return payload
        await asyncio.sleep(0.02)
    raise AssertionError("sync did not finish")


def test_connector_plans_runs_with_progress_and_succeeds(store, tmp_path, monkeypatch, server):
    index = _index(server, **{
        "stt/whisper/base/onnx": ("whisper-encoder.onnx", b"w" * (4 << 20)),
        "tts/mms/en/onnx": ("mms-decoder.onnx", b"m" * 10),
    })
    w.fetch("tts/mms/en/onnx", index["tts/mms/en/onnx"], root=store, managed_by=MANAGED_BY)
    monkeypatch.setattr(w, "refresh_index", lambda sources, root, timeout: _cache(root, index, sources=sources))
    _write_config(tmp_path, monkeypatch, {
        "stt": {"provider": "whisper"},
        "importJson": {"stt": {"whisper": {"weights": "stt/whisper/base/onnx"}}},  # a pending patch counts
    })
    connector = VoiceSyncStore()

    async def scenario():
        # nothing cached yet: the panel's open asks for a reload, which runs behind the
        # answer; once it lands the plan reads the cache it wrote
        planned = await _reloaded(connector)
        assert planned["status"] == "planned" and planned["session_id"] == ""
        assert planned["index"] == {"cached_unix": planned["index"]["cached_unix"], "refreshing": False, "error": None}
        assert planned["index"]["cached_unix"] > 0
        assert [f["key"] for f in planned["plan"]["fetch"]] == ["stt/whisper/base/onnx"]
        assert planned["plan"]["fetch"][0]["bytes"] == 4 << 20
        assert [p["key"] for p in planned["plan"]["prune"]] == ["tts/mms/en/onnx"]
        assert planned["plan"]["unknown"] == []

        server.delay = 0.05
        started = await connector.handle("start", {})
        assert started["status"] == "pending" and started["session_id"]
        assert started["interval_ms"] == 1000
        with pytest.raises(Exception, match="already running"):
            await connector.handle("start", {})
        # a panel opened mid-run asks for the plan and gets the run to follow instead
        reopened = await connector.handle("start", {"plan": ["true"], "refresh": ["true"]})
        assert reopened["session_id"] == started["session_id"] and reopened["status"] == "pending"
        mid = None
        for _ in range(100):
            mid = await connector.handle("poll", {"session_id": [started["session_id"]]})
            if mid["progress"]["done_bytes"] > 0 and mid["status"] == "pending":
                break
            await asyncio.sleep(0.01)
        assert mid["status"] == "pending"
        assert mid["progress"]["key"] == "stt/whisper/base/onnx" and mid["progress"]["file"] == "encoder.onnx"
        assert 0 < mid["progress"]["done_bytes"] < mid["progress"]["total_bytes"] == 4 << 20
        done = await _drive(connector, started["session_id"])
        assert done["status"] == "succeeded"
        assert done["message"] == "Models fetched 1 (4 MB), removed 1 (0 MB)."
        assert done["progress"] == {
            "stage": "done", "key": "stt/whisper/base/onnx", "file": "encoder.onnx",
            "done_bytes": 4 << 20, "total_bytes": 4 << 20, "keys_done": 1, "keys_total": 1,
        }
        # its last word is delivered ONCE: core (re)starts the channel on every "succeeded"
        # it sees, so the poll that follows finds no session rather than a second restart
        with pytest.raises(Exception, match="no such voice sync session"):
            await connector.handle("poll", {"session_id": [started["session_id"]]})
        # nothing left to move: start answers succeeded at once (core then restarts the channel)
        again = await connector.handle("start", {})
        assert again == {"session_id": "", "status": "succeeded", "message": "Models are in place."}

    _run(scenario())
    assert set(w.installed(store)) == {"stt/whisper/base/onnx"}
    assert w.managed_by("stt/whisper/base/onnx", store) == MANAGED_BY


def test_connector_settles_a_run_nobody_watched_and_admits_one_apply_at_a_time(store, tmp_path, monkeypatch, server):
    """The panel can close mid-download: the run goes on in the gateway, and the panel that
    reopens is handed its result — core starts the channel on that one "succeeded", and on
    no later one. Two panels pressing Apply together start one run, not two."""
    index = _index(server, **{
        "stt/whisper/base/onnx": ("whisper-encoder.onnx", b"w" * (1 << 20)),
        "tts/mms/en/onnx": ("mms-decoder.onnx", b"m" * 10),
    })
    _cache(store, index)
    _write_config(tmp_path, monkeypatch, {
        "stt": {"provider": "whisper", "whisper": {"weights": "stt/whisper/base/onnx"}},
        "tts": {"provider": "mms", "mms": {"weights": "tts/mms/en/onnx"}},
    })
    server.delay = 0.02
    connector = VoiceSyncStore()

    async def scenario():
        answers = await asyncio.gather(
            connector.handle("start", {}), connector.handle("start", {}), return_exceptions=True,
        )
        started = [a for a in answers if isinstance(a, dict)]
        refused = [a for a in answers if not isinstance(a, dict)]
        assert len(started) == len(refused) == 1 and "already running" in str(refused[0])

        task = connector._session.task  # the panel is gone; the gateway finishes the run
        await task
        # the panel reopens: its plan request answers with the run's result, once
        settled = await connector.handle("start", {"plan": ["true"], "refresh": ["true"]})
        assert settled["status"] == "succeeded" and settled["session_id"] == started[0]["session_id"]
        planned = await connector.handle("start", {"plan": ["true"]})
        assert planned["status"] == "planned" and planned["plan"]["fetch"] == []

    _run(scenario())
    assert set(w.installed(store)) == {"stt/whisper/base/onnx", "tts/mms/en/onnx"}


def test_progress_only_rises_across_a_models_files(store, tmp_path, monkeypatch, server):
    """A model is several files; the bar counts the run's bytes, so a new file does not
    send it backwards."""
    blob = b"w" * (1 << 20)
    for name in ("enc.onnx", "dec.onnx"):
        server.blobs[name] = blob
    index = {"stt/whisper/base/onnx": {"files": {
        name: {"url": f"{server.base}/{name}", "sha256": hashlib.sha256(blob).hexdigest(), "size": len(blob)}
        for name in ("enc.onnx", "dec.onnx")
    }, "langs": ["en"]}}
    _cache(store, index)
    _write_config(tmp_path, monkeypatch, {"stt": {"provider": "whisper", "whisper": {"weights": "stt/whisper/base/onnx"}}})
    server.delay = 0.02
    connector = VoiceSyncStore()
    seen = []

    async def scenario():
        started = await connector.handle("start", {})
        for _ in range(400):
            payload = await connector.handle("poll", {"session_id": [started["session_id"]]})
            seen.append(payload["progress"]["done_bytes"])
            if payload["status"] != "pending":
                assert payload["status"] == "succeeded"
                return
            await asyncio.sleep(0.01)
        raise AssertionError("sync did not finish")

    _run(scenario())
    assert seen == sorted(seen) and seen[-1] == 2 << 20


def _cache(root, models, *, sources=None, fetched_unix=None):
    root.mkdir(parents=True, exist_ok=True)
    (root / w.INDEX_CACHE).write_text(json.dumps({
        "fetched_unix": int(time.time()) if fetched_unix is None else fetched_unix,
        "sources": w.index_sources() if sources is None else sources,
        "models": models,
    }))
    return models


def test_connector_refuses_unaccepted_notices_unknown_keys_and_no_space(store, tmp_path, monkeypatch, server):
    index = _index(server, **{"tts/mms/en/onnx": ("mms-decoder.onnx", b"m" * 10)})
    index["tts/mms/en/onnx"]["accept"] = "non-commercial use only"
    _cache(store, index)
    # the panel plans what the setup RUNS: the engine has to be the selected one
    _write_config(tmp_path, monkeypatch, {"tts": {"provider": "mms", "mms": {"weights": "tts/mms/en/onnx"}}})
    connector = VoiceSyncStore()

    async def scenario():
        planned = await connector.handle("start", {"plan": ["true"]})
        assert planned["plan"]["fetch"][0]["notice"] == "non-commercial use only"
        with pytest.raises(Exception, match="accept the notice for tts/mms/en/onnx"):
            await connector.handle("start", {})
        started = await connector.handle("start", {"accept": ["tts/mms/en/onnx"]})
        assert (await _drive(connector, started["session_id"]))["status"] == "succeeded"

        _write_config(tmp_path, monkeypatch, {"tts": {"provider": "mms", "mms": {"weights": "tts/mms/zz/onnx"}}})
        with pytest.raises(Exception, match="not in the model index: tts/mms/zz/onnx"):
            await connector.handle("start", {})

        _write_config(tmp_path, monkeypatch, {"tts": {"provider": "mms", "mms": {"weights": "tts/mms/en/onnx"}}})
        w.prune("tts/mms/en/onnx", store)
        monkeypatch.setattr("nanobot_channel_voice.sync.shutil.disk_usage", lambda _p: type("u", (), {"free": 3})())
        # the fetch runs before the prune, so its need is the whole download, said in units
        # that show a small shortfall
        with pytest.raises(Exception, match="not enough disk space: 10 bytes needed, 3 bytes free"):
            await connector.handle("start", {"accept": ["tts/mms/en/onnx"]})

    _run(scenario())


def test_connector_cancel_stops_between_chunks_and_leaves_no_partial(store, tmp_path, monkeypatch, server):
    index = _index(server, **{"stt/whisper/base/onnx": ("whisper-encoder.onnx", b"w" * (4 << 20))})
    _cache(store, index)
    _write_config(tmp_path, monkeypatch, {"stt": {"provider": "whisper", "whisper": {"weights": "stt/whisper/base/onnx"}}})
    server.delay = 0.05
    connector = VoiceSyncStore()

    async def scenario():
        started = await connector.handle("start", {})
        await asyncio.sleep(0.1)
        # cancel answers at once; the thread stops at its next chunk and the poll says so
        asked = await connector.handle("cancel", {"session_id": [started["session_id"]]})
        assert asked["status"] == "pending"
        cancelled = await _drive(connector, started["session_id"])
        assert cancelled["status"] == "cancelled"
        assert "download cancelled" in cancelled["message"]
        with pytest.raises(Exception, match="no such voice sync session"):
            await connector.handle("poll", {"session_id": ["nope"]})
        await connector.close()

    _run(scenario())
    assert w.installed(store) == {}
    assert not list(store.rglob(".partial-*"))


def test_connector_reports_a_crashed_run_and_a_cancel_before_the_removals(store, tmp_path, monkeypatch):
    """A run nobody awaits must end the polls whatever failed; a cancel that lands after
    the downloads must not go on to remove models."""
    (tmp_path / "d.onnx").write_bytes(b"d")
    w.fetch("tts/mms/en/onnx", {"files": {"d.onnx": {"url": (tmp_path / "d.onnx").as_uri()}}},
            root=store, managed_by=MANAGED_BY)
    index = _cache(store, {"stt/whisper/base/onnx": {"files": {"e.onnx": {"url": "https://x/e", "sha256": "0" * 64, "size": 1}}}})
    _write_config(tmp_path, monkeypatch, {"stt": {"provider": "whisper", "whisper": {"weights": "stt/whisper/base/onnx"}}})
    monkeypatch.setattr("nanobot_channel_voice.webui_sync.run_sync", lambda *a, **k: 1 / 0)
    connector = VoiceSyncStore()

    async def scenario():
        started = await connector.handle("start", {})
        done = await _drive(connector, started["session_id"])
        assert done["status"] == "failed" and done["message"] == "division by zero"

    _run(scenario())
    plan = plan_sync({}, index, store, managed_by=MANAGED_BY)  # nothing to fetch, mms to remove
    assert plan.fetch == [] and plan.prune == ["tts/mms/en/onnx"]
    with pytest.raises(w.WeightsError, match="sync cancelled"):
        run_sync(plan, index, store, managed_by=MANAGED_BY, should_stop=lambda: True)
    assert set(w.installed(store)) == {"tts/mms/en/onnx"}


def test_connector_plan_survives_an_unreachable_index(store, tmp_path, monkeypatch):
    _cache(store, {"stt/whisper/base/onnx": {"files": {"e.onnx": {"url": "https://x/e", "sha256": "0" * 64}}}},
           fetched_unix=1)

    def offline(sources, root, timeout):
        raise w.WeightsError("cannot read weights index 'https://x': offline")

    monkeypatch.setattr(w, "refresh_index", offline)
    _write_config(tmp_path, monkeypatch, {})
    planned = _run(_reloaded(VoiceSyncStore()))
    assert planned["index"] == {
        "cached_unix": 1, "refreshing": False, "error": "cannot read weights index 'https://x': offline",
    }
    assert planned["plan"]["fetch"] == [] and planned["plan"]["prune"] == []


def test_plan_answers_from_the_cache_and_reloads_behind_it(store, tmp_path, monkeypatch):
    """The plan never waits on the network: a reload runs behind the answer when the
    panel opens (``refresh``), one at a time, the status saying so; without it the cache
    stands, whatever its age. Core answers a socket's requests one at a time, so a plan
    that fetched would hold the form's own request behind it."""
    loads = []
    gate = threading.Event()

    def fake_refresh(sources, root, timeout):
        loads.append(list(sources))
        gate.wait(5)
        return _cache(root, {}, sources=list(sources))

    monkeypatch.setattr(w, "refresh_index", fake_refresh)
    monkeypatch.setenv("NANOBOT_VOICE_INDEX", "https://index.test/weights.json")
    _write_config(tmp_path, monkeypatch, {})
    _cache(store, {}, fetched_unix=1)  # a cache however old stands until asked

    async def scenario():
        connector = VoiceSyncStore()
        plan = lambda **q: connector.handle("start", {"plan": ["true"], **{k: [v] for k, v in q.items()}})  # noqa: E731
        first = await plan()
        assert first["index"] == {"cached_unix": 1, "refreshing": False, "error": None} and loads == []
        gate.clear()
        opened = await plan(refresh="true")
        assert opened["index"]["refreshing"] is True and opened["index"]["cached_unix"] == 1  # answered at once
        again = await plan(refresh="true")  # a second open while it runs: the same reload
        assert again["index"]["refreshing"] is True
        assert loads == [["https://index.test/weights.json"]]
        gate.set()
        await connector._reload
        landed = await plan()
        assert landed["index"]["refreshing"] is False and landed["index"]["cached_unix"] > 1
        assert landed["index"]["error"] is None
        # a local file source reloads the same way, and a failed reload's error stands
        # in the status until the next reload starts
        local = tmp_path / "dev-index.json"
        local.write_text(json.dumps({"version": 1, "models": {}}))
        monkeypatch.setenv("NANOBOT_VOICE_INDEX", str(local))
        await plan(refresh="true")
        await connector._reload
        assert loads[-1] == [str(local)]
        monkeypatch.setattr(w, "refresh_index", lambda *a, **k: (_ for _ in ()).throw(w.WeightsError("offline")))
        await plan(refresh="true")
        await connector._reload
        assert (await plan())["index"]["error"] == "offline"
        monkeypatch.setattr(w, "refresh_index", fake_refresh)
        assert (await plan(refresh="true"))["index"]["error"] is None
        await connector._reload

    _run(scenario())


def test_the_wake_model_row_is_the_tier_switch_and_a_head_fills_the_phrases(store):
    """With a mode on, the Model row leads with Transcript ahead of the index's heads; each
    pick writes wake.engine, a head also fills Phrases with the phrase it hears (the
    package meta once installed, the index's naming before, nothing for a custom key)
    beside the phrases typed there, swapping out only the one the old head heard,
    Local files is told from Transcript by filesOpen, and the Phrases row says when the
    list lacks the picked head's phrase."""
    def fields(section):
        return {f["key"]: f for s in build_form(VoiceConfig.model_validate(section))["sections"] for f in s["fields"]}

    _cache(store, {
        "wake/openwakeword/alexa/onnx": {"files": {"m.onnx": {"url": "https://x/a", "sha256": "0" * 64}}},
        "wake/openwakeword/hey-jarvis/onnx": {"files": {"m.onnx": {"url": "https://x/j", "sha256": "0" * 64}}},
        "wake/openwakeword/hey-mycroft/rknn.rk3588": {"files": {"m.rknn": {"url": "https://x/m", "sha256": "0" * 64}}},
    })
    gate = {"wake": {"mode": "gate", "phrases": ["hey nanobot"]}}
    model = fields(gate)["wake.openwakeword.weights"]
    assert [c["value"] for c in model["choices"]] == ["", "wake/openwakeword/alexa/onnx", "wake/openwakeword/hey-jarvis/onnx"]
    assert model["choices"][0] == {"value": "", "label": "Transcript", "sets": {"wake.engine": "text"}}
    assert model["choices"][2]["sets"] == {
        "wake.engine": "openwakeword", "wake.phrases": ["hey nanobot", "hey jarvis"],
    }
    # switching heads: the phrase the selected head hears goes, what was typed stays
    switching = {"wake": {**gate["wake"], "phrases": ["hey nanobot", "alexa"], "engine": "openwakeword",
                          "openwakeword": {"weights": "wake/openwakeword/alexa/onnx"}}}
    assert fields(switching)["wake.openwakeword.weights"]["choices"][2]["sets"]["wake.phrases"] == [
        "hey nanobot", "hey jarvis",
    ]
    assert model["sets"] == {"wake.engine": "openwakeword"} and model["customOpen"] is False and model["custom"] == "files"
    assert "value" not in model and model["help"].startswith("Transcript matches the phrase in the transcription. A head")
    assert "wake.openwakeword.threshold" not in fields(gate)
    # a head whose phrase Phrases already has fills nothing
    assert fields({"wake": {"mode": "gate", "phrases": ["Alexa"]}})["wake.openwakeword.weights"]["choices"][1]["sets"] == {"wake.engine": "openwakeword"}
    # picked: the engine set with it, the key in force, the threshold shown
    oww = {"wake": {**gate["wake"], "engine": "openwakeword", "openwakeword": {"weights": "wake/openwakeword/hey-jarvis/onnx"}}}
    picked = fields(oww)
    assert picked["wake.openwakeword.weights"]["value"] == "wake/openwakeword/hey-jarvis/onnx"
    assert picked["wake.openwakeword.weights"]["customOpen"] is False and "wake.openwakeword.threshold" in picked
    # on the Phrases row: the Model row's help gives way to a license notice
    told = "Comma separated. The selected head hears `hey jarvis`, so include it."
    assert picked["wake.phrases"]["help"] == told
    oww["wake"]["phrases"] = ["Hey Jarvis", "jarvis"]
    assert fields(oww)["wake.phrases"]["help"] == "Comma separated, for example `hey nanobot`."
    # Custom (a head of your own): the engine without a key, and the backbone is no head
    local = fields({"wake": {**gate["wake"], "engine": "openwakeword"}})
    assert local["wake.openwakeword.weights"]["customOpen"] is True and "value" not in local["wake.openwakeword.weights"]
    assert "wake.openwakeword.modelPath" in local
    assert not any("backbone" in c["value"] for c in local["wake.openwakeword.weights"]["choices"])
    # a key written by hand without the engine is not the tier in force
    assert "value" not in fields({"wake": {**gate["wake"], "openwakeword": {"weights": "wake/openwakeword/alexa/onnx"}}})["wake.openwakeword.weights"]
    # installed: the package meta's phrase wins over the naming
    d = store / "wake/openwakeword/hey-jarvis/onnx"
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"phrase": "Hey Jarvis Bot"}))
    (d / w.MANIFEST).write_text(json.dumps({"key": "wake/openwakeword/hey-jarvis/onnx", "files": {}}))
    assert fields(oww)["wake.phrases"]["help"] == told.replace("hey jarvis", "hey jarvis bot")
    assert fields(gate)["wake.openwakeword.weights"]["choices"][2]["sets"]["wake.phrases"] == [
        "hey nanobot", "hey jarvis bot",
    ]
    oww["wake"]["openwakeword"]["weights"] = "wake/openwakeword/mine/onnx"
    assert fields(oww)["wake.phrases"]["help"].startswith("Comma separated, for example")
    # no head anywhere: the row still switches the tiers
    (store / w.INDEX_CACHE).unlink()
    shutil.rmtree(store / "wake")
    bare = fields(gate)["wake.openwakeword.weights"]
    assert [c["value"] for c in bare["choices"]] == [""] and bare["help"].endswith("The index lists no head, Custom takes one of your own.")


def test_picking_an_engine_also_picks_its_first_model(store):
    """An engine choice carries its first listed model as ``sets`` while its block names
    no model and no file, so a Model row never opens on nothing selected; a block already
    set up keeps its setup, and an engine without a store model carries nothing."""
    def choices(section, path):
        fields = {f["key"]: f for s in build_form(VoiceConfig.model_validate(section))["sections"] for f in s["fields"]}
        return {c["value"]: c.get("sets") for c in fields[path]["choices"]}

    _cache(store, {
        "stt/whisper/base/onnx": {"files": {"e.onnx": {"url": "https://x/o", "sha256": "0" * 64}}},
        "stt/whisper/base/rknn.rv1126b": {"files": {"e.rknn": {"url": "https://x/r", "sha256": "0" * 64}}},
        "stt/whisper/tiny/onnx": {"files": {"e.onnx": {"url": "https://x/t", "sha256": "0" * 64}}},
        "vad/smartturn/v3.2/onnx": {"files": {"m.onnx": {"url": "https://x/s", "sha256": "0" * 64}}},
    })
    stt = choices({"device": "rv1126b"}, "stt.provider")
    assert stt["whisper"] == {"stt.whisper.weights": "stt/whisper/base/rknn.rv1126b"}  # the device build of the first stem
    assert stt["nanobot"] is None and stt["sensevoice"] is None  # nothing listed for it
    assert choices({}, "stt.provider")["whisper"] == {"stt.whisper.weights": "stt/whisper/base/onnx"}
    assert choices({"stt": {"whisper": {"weights": "stt/whisper/tiny/onnx"}}}, "stt.provider")["whisper"] is None
    assert choices({"stt": {"whisper": {"encoderPath": "/e.onnx"}}}, "stt.provider")["whisper"] is None
    assert choices({}, "vad.turn.engine") == {"none": None, "smartturn": {"vad.turn.weights": "vad/smartturn/v3.2/onnx"}}


def test_picking_a_gate_also_picks_its_detector(store):
    """On speech and After wake word run the neural VAD, which the config refuses to do
    without and whose row is advanced: the pick carries Silero with its first model while
    the detector is not neural, the model alone left out once its block is set up, and
    nothing once a neural detector is picked. After wake word also turns the wake mode on
    (the uplink refuses a wake gate without one), the head staying the user's pick in the
    section that then shows; Off there keeps the gate on speech."""
    def choices(section, path="realtime.uplink"):
        from pydantic import ValidationError

        from nanobot_channel_voice.webui_form import lenient_config

        try:  # as the validator does it: a refused section still gets its rows
            cfg = VoiceConfig.model_validate(section)
        except ValidationError:
            cfg = lenient_config(section)
        fields = {f["key"]: f for s in build_form(cfg)["sections"] for f in s["fields"]}
        return {c["value"]: c.get("sets") for c in fields[path]["choices"]}

    cloud = {"backend": "openai"}
    assert choices(cloud) == {  # no index: the engine alone
        "server": None, "vad": {"vad.engine": "silero"}, "wake": {"vad.engine": "silero", "wake.mode": "gate"},
    }
    _cache(store, {"vad/silero/v6/onnx": {"files": {"m.onnx": {"url": "https://x/s", "sha256": "0" * 64}}}})
    picked = {"vad.engine": "silero", "vad.silero.weights": "vad/silero/v6/onnx"}
    assert choices(cloud) == {"server": None, "vad": picked, "wake": {**picked, "wake.mode": "gate"}}
    assert choices({**cloud, "vad": {"silero": {"weights": "vad/silero/mine/onnx"}}})["vad"] == {"vad.engine": "silero"}
    assert choices({**cloud, "vad": {"engine": "firered"}})["vad"] is None
    assert choices({**cloud, "vad": {"engine": "silero"}})["wake"] == {"wake.mode": "gate"}
    strict = {**cloud, "vad": {"engine": "silero"}, "wake": {"mode": "strict", "phrases": ["hey"]}}
    assert choices(strict)["wake"] is None  # a mode on is kept, Strict included
    # ...and leaving the gate takes back the mode it turned on, which nothing has filled:
    # the section would else be refused from a Wake word section the form has folded away
    unfilled = {**cloud, "realtime": {"uplink": "wake"}, "vad": {"engine": "silero"}, "wake": {"mode": "gate"}}
    assert choices(unfilled)["vad"] == {"wake.mode": "off"} and choices(unfilled)["server"] == {"wake.mode": "off"}
    assert choices({**unfilled, "wake": {"mode": "gate", "phrases": ["hey"]}})["vad"] is None  # the user's own
    # Off under After wake word: the gate falls back to On speech; nowhere else
    woken = {**strict, "realtime": {"uplink": "wake"}, "wake": {**strict["wake"], "engine": "openwakeword"}}
    assert choices(woken, "wake.mode") == {"off": {"realtime.uplink": "vad"}, "gate": None, "strict": None}
    assert choices({**strict, "realtime": {"uplink": "vad"}}, "wake.mode")["off"] is None
    assert choices({"wake": {"mode": "gate", "phrases": ["hey"]}}, "wake.mode")["off"] is None  # local
    # what the pick writes is what the gate accepts, where the bare pick is refused
    with pytest.raises(ValueError, match="needs a neural VAD"):
        VoiceConfig.model_validate({**cloud, "realtime": {"uplink": "vad"}})
    section = {**cloud, "realtime": {"uplink": "vad"}, "vad": {"engine": "silero", "silero": {"weights": picked["vad.silero.weights"]}}}
    assert used_weights_keys(section) == {"vad/silero/v6/onnx"}  # validates, and Apply fetches it
