"""Weight store: index merge, fetch (http + file://), prune, path resolution, CLI.

Everything runs against a tmp store via $NANOBOT_VOICE_MODELS_DIR; http
downloads are faked at urllib so no test touches the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request

import pytest

from nanobot_channel_voice import weights as w
from nanobot_channel_voice.cli import main as cli_main


@pytest.fixture()
def store(tmp_path, monkeypatch):
    root = tmp_path / "store"
    monkeypatch.setenv("NANOBOT_VOICE_MODELS_DIR", str(root))
    return root


def _entry_for(*files, langs=("en",), **extra):
    """Index entry linking file:// sources with pinned sha256s."""
    spec = {}
    for path in files:
        spec[path.name] = {
            "url": path.as_uri(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {"files": spec, "langs": list(langs), "license": "MIT", **extra}


def _src(tmp_path, name, blob=b"weights!"):
    d = tmp_path / "served"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_bytes(blob)
    return p


# ---- keys / index -----------------------------------------------------------


def test_key_shape_is_enforced():
    w.validate_key("stt/whisper-base/hef.hailo-10h")
    w.validate_key("tts/matcha/en-US/ljspeech/rknn.rv1126b")
    for bad in ("stt/whisper-base", "a/b", "stt/../etc", "stt//onnx", "/stt/m/p", "stt/m/.p"):
        with pytest.raises(w.WeightsError, match="invalid weights key"):
            w.validate_key(bad)


def test_index_sources_merge_later_wins(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"models": {"stt/m/onnx": {"license": "A"}}}))
    b.write_text(json.dumps({"models": {"stt/m/onnx": {"license": "B"}}}))
    merged = w.load_index([str(a), b.as_uri()])  # plain path AND file:// URL forms
    assert merged["stt/m/onnx"]["license"] == "B"


def test_no_sources_falls_back_to_the_default_urls_and_nothing_bundled(monkeypatch):
    # The wheel ships no index DATA, only URLs to a community-served index, so an
    # argument-less load must read exactly those and produce no entries of its own.
    assert w.DEFAULT_INDEX_SOURCES
    assert all(s.startswith("https://") for s in w.DEFAULT_INDEX_SOURCES)
    seen = []
    monkeypatch.setattr(w, "_read_source", lambda s: seen.append(s) or {"models": {}})
    assert w.load_index() == {}
    assert seen == list(w.DEFAULT_INDEX_SOURCES)
    # An explicit source REPLACES the default rather than adding to it.
    seen.clear()
    w.load_index(["https://example.invalid/i.json"])
    assert seen == ["https://example.invalid/i.json"]


def test_index_with_a_traversal_key_is_rejected(tmp_path):
    evil = tmp_path / "evil.json"
    evil.write_text(json.dumps({"models": {"stt/../../escape": {}}}))
    with pytest.raises(w.WeightsError, match="invalid weights key"):
        w.load_index([str(evil)])


@pytest.mark.parametrize(
    ("models", "match"),
    [
        ([{"stt/m/onnx": {}}], "'models' must be an object"),
        ({"stt/m/onnx": ["files"]}, "must be a JSON object"),
        ({"stt/m/onnx": {"files": ["encoder.onnx"]}}, r"\.files must be an object"),
        ({"stt/m/onnx": {"files": {"encoder.onnx": "https://x.test/e"}}}, r"must be an object"),
    ],
)
def test_malformed_index_shapes_are_errors_not_tracebacks(store, tmp_path, capsys, models, match):
    # An index is external input: a shape error must surface as an actionable
    # line (CLI exit 2), never as a TypeError from whichever consumer trips first.
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"version": 1, "models": models}))
    with pytest.raises(w.WeightsError, match=match):
        w.load_index([str(path)])
    assert cli_main(["--index", str(path), "list"]) == 2
    assert "error:" in capsys.readouterr().err


# ---- fetch ------------------------------------------------------------------


def test_fetch_file_url_links_and_verifies(store, tmp_path):
    src = _src(tmp_path, "encoder.onnx")
    d = w.fetch("stt/m/onnx", _entry_for(src))
    dest = d / "encoder.onnx"
    assert dest.is_symlink() and dest.read_bytes() == b"weights!"
    manifest = json.loads((d / w.MANIFEST).read_text())
    assert manifest["files"]["encoder.onnx"]["linked"] == str(src.resolve())


def test_fetch_file_url_checksum_mismatch_refuses(store, tmp_path):
    src = _src(tmp_path, "encoder.onnx")
    entry = _entry_for(src)
    entry["files"]["encoder.onnx"]["sha256"] = "0" * 64
    with pytest.raises(w.WeightsError, match="sha256 mismatch"):
        w.fetch("stt/m/onnx", entry)
    assert not (w.store_dir("stt/m/onnx") / w.MANIFEST).exists()  # never marked fetched


def test_fetch_http_streams_verifies_and_is_idempotent(store, monkeypatch):
    blob = b"remote-model-bytes" * 1000
    calls = []

    def fake_urlopen(url, timeout=0):
        calls.append(url)
        return io.BytesIO(blob)  # IOBase is already a context manager

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    entry = {
        "files": {"model.onnx": {"url": "https://example.test/model.onnx",
                                 "sha256": hashlib.sha256(blob).hexdigest()}}
    }
    d = w.fetch("vad/firered/onnx", entry)
    assert (d / "model.onnx").read_bytes() == blob
    assert not list(d.glob(".partial-*"))
    w.fetch("vad/firered/onnx", entry)          # already fetched: no second request
    assert len(calls) == 1
    w.fetch("vad/firered/onnx", entry, force=True)
    assert len(calls) == 2


def test_fetch_http_bad_checksum_never_lands(store, monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: io.BytesIO(b"tampered"))
    entry = {"files": {"model.onnx": {"url": "https://x.test/m", "sha256": "0" * 64}}}
    with pytest.raises(w.WeightsError, match="refusing to install"):
        w.fetch("vad/firered/onnx", entry)
    d = w.store_dir("vad/firered/onnx")
    assert not (d / "model.onnx").exists() and not list(d.glob(".partial-*"))


def test_fetch_http_requires_a_pinned_sha256(store):
    entry = {"files": {"model.onnx": {"url": "https://x.test/m"}}}
    with pytest.raises(w.WeightsError, match="must pin a sha256"):
        w.fetch("vad/firered/onnx", entry)


def test_fetch_rejects_unsafe_file_names(store):
    for name in ("../evil", "a/b", ".hidden", ""):
        with pytest.raises(w.WeightsError, match="unsafe file name"):
            w.fetch("stt/m/onnx", {"files": {name: {"url": "file:///x"}}})


# ---- prune / installed ------------------------------------------------------


def test_prune_frees_and_drops_empty_parents(store, tmp_path):
    src = _src(tmp_path, "encoder.onnx", b"x" * 4096)
    w.fetch("stt/m/onnx", _entry_for(src))
    assert w.installed() == {"stt/m/onnx": w.store_dir("stt/m/onnx")}
    freed = w.prune("stt/m/onnx")
    assert freed > 0
    assert w.installed() == {} and not (store / "stt").exists()
    with pytest.raises(w.WeightsError, match="not in the store"):
        w.prune("stt/m/onnx")


def test_disk_usage_counts_links_not_targets(store, tmp_path):
    src = _src(tmp_path, "encoder.onnx", b"x" * (1 << 20))
    d = w.fetch("stt/m/onnx", _entry_for(src))
    assert w.disk_usage(d) < 1 << 16  # the symlink + manifest, not the 1 MiB target


def test_installed_discovers_hierarchical_keys_and_prune_keeps_siblings(store, tmp_path):
    src = _src(tmp_path, "encoder.onnx")
    en = "tts/matcha/en-US/ljspeech/rknn.rv1126b"
    zh = "tts/matcha/zh-CN/baker/rknn.rv1126b"
    w.fetch(en, _entry_for(src))
    w.fetch(zh, _entry_for(src))

    assert set(w.installed()) == {en, zh}
    w.prune(en)
    assert set(w.installed()) == {zh}
    assert (store / "tts" / "matcha" / "zh-CN" / "baker" / "rknn.rv1126b").is_dir()
    assert not (store / "tts" / "matcha" / "en-US").exists()


def test_nesting_keys_are_refused_and_ancestor_prune_rejected(store, tmp_path):
    # Hierarchical keys must never nest: fetch's stale-file sweep (and prune) would
    # otherwise rmtree the inner key's verified weights.
    src = _src(tmp_path, "encoder.onnx")
    child = "tts/matcha/en/ljspeech/onnx"
    w.fetch(child, _entry_for(src))
    for nesting in ("tts/matcha/en", "tts/matcha/en/ljspeech/onnx/sub"):
        with pytest.raises(w.WeightsError, match="nest"):
            w.fetch(nesting, _entry_for(src))
    # Pruning an intermediate dir (a never-fetched ancestor) must not delete children.
    with pytest.raises(w.WeightsError, match="not in the store"):
        w.prune("tts/matcha/en")
    assert set(w.installed()) == {child}


# ---- runtime resolution -----------------------------------------------------


def _fetched_whisper_store(store, tmp_path):
    files = [_src(tmp_path, n) for n in
             ("encoder.onnx", "decoder.onnx", "vocab.json", "mel_filters.txt")]
    w.fetch("stt/whisper-base/onnx", _entry_for(*files))


def test_fill_engine_paths_resolves_by_field_stem(store, tmp_path):
    from nanobot_channel_voice.config import WhisperSttConfig

    _fetched_whisper_store(store, tmp_path)
    block = WhisperSttConfig.model_validate(
        {"weights": "stt/whisper-base/onnx", "vocabPath": "/explicit/vocab.json"}
    )
    filled = w.fill_engine_paths(block)
    d = w.store_dir("stt/whisper-base/onnx")
    assert filled.encoder_path == str(d / "encoder.onnx")
    assert filled.decoder_path == str(d / "decoder.onnx")
    assert filled.mel_filters_path == str(d / "mel_filters.txt")
    assert filled.vocab_path == "/explicit/vocab.json"  # explicit config wins over the store


def test_unfetched_weights_error_names_the_command(store):
    from nanobot_channel_voice.config import WhisperSttConfig

    block = WhisperSttConfig.model_validate({"weights": "stt/whisper-base/onnx"})
    with pytest.raises(w.WeightsError, match="nanobot-voice fetch stt/whisper-base/onnx"):
        w.fill_engine_paths(block)


def test_ambiguous_store_files_are_an_error(store, tmp_path):
    from nanobot_channel_voice.config import FireRedVadConfig

    # Two engine formats of the same field, plus external data: the companion
    # filter must not swallow the real ambiguity between them.
    w.fetch("vad/firered/onnx", _entry_for(_src(tmp_path, "model.onnx"),
                                           _src(tmp_path, "model.onnx.data"),
                                           _src(tmp_path, "model.rknn"),
                                           _src(tmp_path, "cmvn.ark")))
    block = FireRedVadConfig.model_validate({"weights": "vad/firered/onnx"})
    with pytest.raises(w.WeightsError, match="ambiguous model"):
        w.fill_engine_paths(block)


def test_onnx_external_data_is_a_companion_not_an_ambiguity(store, tmp_path):
    from nanobot_channel_voice.config import WhisperSttConfig

    # Both spellings a graph may name its external tensor blob with.
    files = [_src(tmp_path, n) for n in
             ("encoder.onnx", "encoder.onnx.data", "encoder.onnx_data", "decoder.onnx")]
    w.fetch("stt/whisper-base/onnx", _entry_for(*files))
    filled = w.fill_engine_paths(
        WhisperSttConfig.model_validate({"weights": "stt/whisper-base/onnx"})
    )
    d = w.store_dir("stt/whisper-base/onnx")
    assert filled.encoder_path == str(d / "encoder.onnx")
    assert filled.decoder_path == str(d / "decoder.onnx")


def test_a_dotted_variant_still_resolves(store, tmp_path):
    from nanobot_channel_voice.config import FireRedVadConfig

    w.fetch("vad/firered/onnx", _entry_for(_src(tmp_path, "model.int8.onnx"),
                                           _src(tmp_path, "cmvn.ark")))
    filled = w.fill_engine_paths(FireRedVadConfig.model_validate({"weights": "vad/firered/onnx"}))
    assert filled.model_path == str(w.store_dir("vad/firered/onnx") / "model.int8.onnx")


def test_refetch_sweeps_a_file_the_entry_dropped(store, tmp_path):
    from nanobot_channel_voice.config import FireRedVadConfig

    onnx = _src(tmp_path, "model.onnx")
    cmvn = _src(tmp_path, "cmvn.ark")
    w.fetch("vad/firered/onnx", _entry_for(onnx, _src(tmp_path, "model.rknn"), cmvn))
    d = w.fetch("vad/firered/onnx", _entry_for(onnx, cmvn))  # revision without the rknn
    assert sorted(p.name for p in d.iterdir()) == [w.MANIFEST, "cmvn.ark", "model.onnx"]
    # A survivor would wedge model.* resolution for good.
    filled = w.fill_engine_paths(FireRedVadConfig.model_validate({"weights": "vad/firered/onnx"}))
    assert filled.model_path == str(d / "model.onnx")


def test_make_stt_delegates_when_weights_unfetched(store):
    from nanobot_channel_voice.config import SttConfig
    from nanobot_channel_voice.stt import make_stt

    cfg = SttConfig.model_validate(
        {"provider": "whisper", "whisper": {"weights": "stt/whisper-base/onnx"}}
    )
    assert make_stt(cfg) is None  # warn + delegate, never crash the channel


def test_make_vad_falls_back_when_weights_unfetched(store):
    from nanobot_channel_voice.config import VadConfig
    from nanobot_channel_voice.vad import EnergyVad, make_vad

    cfg = VadConfig.model_validate(
        {"engine": "firered", "firered": {"weights": "vad/firered/onnx"}}
    )
    assert isinstance(make_vad(cfg, 16000, 20), EnergyVad)


# ---- CLI --------------------------------------------------------------------


def _write_index(tmp_path, models):
    p = tmp_path / "index.json"
    p.write_text(json.dumps({"version": 1, "models": models}))
    return str(p)


def test_cli_fetch_list_prune_roundtrip(store, tmp_path, capsys):
    src = _src(tmp_path, "encoder.onnx")
    index = _write_index(tmp_path, {"stt/whisper-base/hef.hailo-10h": _entry_for(src)})

    assert cli_main(["--index", index, "fetch", "stt/whisper-base"]) == 0  # unique prefix
    assert (w.store_dir("stt/whisper-base/hef.hailo-10h") / "encoder.onnx").exists()

    assert cli_main(["--index", index, "list", "--lang", "en"]) == 0
    out = capsys.readouterr().out
    assert "stt/whisper-base/hef.hailo-10h" in out and "installed" in out

    assert cli_main(["prune", "stt/whisper-base", "--yes"]) == 0
    assert "freed" in capsys.readouterr().out
    assert w.installed() == {}


def test_cli_accept_notice_needs_yes_off_a_tty(store, tmp_path, capsys):
    src = _src(tmp_path, "encoder.onnx")
    entry = _entry_for(src, accept="CC-BY-NC 4.0: non-commercial use only.")
    index = _write_index(tmp_path, {"tts/mms-eng/onnx": entry})

    assert cli_main(["--index", index, "fetch", "tts/mms-eng/onnx"]) == 2
    assert "--yes" in capsys.readouterr().err
    assert cli_main(["--index", index, "fetch", "tts/mms-eng/onnx", "--yes"]) == 0
    assert "NOTICE" in capsys.readouterr().out


def test_cli_ambiguous_and_unknown_keys(store, tmp_path, capsys):
    src = _src(tmp_path, "encoder.onnx")
    index = _write_index(tmp_path, {
        "stt/whisper-base/onnx": _entry_for(src),
        "stt/whisper-base/hef.hailo-10h": _entry_for(src),
    })
    assert cli_main(["--index", index, "fetch", "stt/whisper-base"]) == 2
    assert "ambiguous" in capsys.readouterr().err
    assert cli_main(["--index", index, "fetch", "stt/nope"]) == 2
    assert "unknown weights key" in capsys.readouterr().err


def test_cli_prune_takes_keys_xor_all(store, capsys):
    assert cli_main(["prune"]) == 2
    assert "either keys or --all" in capsys.readouterr().err


def _write_config(tmp_path, voice_section):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"channels": {"voice": voice_section}}))
    return str(p)


def test_cli_sync_fetches_configured_keys_and_prunes_the_rest(store, tmp_path, capsys):
    src = _src(tmp_path, "encoder.onnx")
    index = _write_index(tmp_path, {
        "stt/whisper-base/onnx": _entry_for(src),
        "tts/mms-deu/onnx": _entry_for(src),
    })
    cfg = _write_config(tmp_path, {
        "stt": {"provider": "whisper", "whisper": {"weights": "stt/whisper-base/onnx"}},
    })

    # An installed key the config does NOT name: sync --prune must drop it.
    assert cli_main(["--index", index, "fetch", "tts/mms-deu/onnx"]) == 0
    assert cli_main(["--index", index, "sync", "--config", cfg, "--prune"]) == 0
    out = capsys.readouterr().out
    assert "stt/whisper-base/onnx" in out and "pruned tts/mms-deu/onnx" in out
    assert set(w.installed()) == {"stt/whisper-base/onnx"}
    # Idempotent: a second sync fetches nothing new and prunes nothing.
    assert cli_main(["--index", index, "sync", "--config", cfg, "--prune"]) == 0
    assert "already fetched" in capsys.readouterr().out


def test_cli_sync_names_configured_keys_missing_from_the_index(store, tmp_path, capsys):
    index = _write_index(tmp_path, {})
    cfg = _write_config(tmp_path, {"vad": {"firered": {"weights": "vad/firered/rknn.rk3588"}}})
    assert cli_main(["--index", index, "sync", "--config", cfg]) == 2
    assert "vad/firered/rknn.rk3588" in capsys.readouterr().err


def test_cli_sync_with_an_empty_config_refuses_to_prune(store, tmp_path, capsys):
    src = _src(tmp_path, "encoder.onnx")
    index = _write_index(tmp_path, {"stt/m/onnx": _entry_for(src)})
    assert cli_main(["--index", index, "fetch", "stt/m/onnx"]) == 0
    cfg = _write_config(tmp_path, {"backend": "local"})
    assert cli_main(["--index", index, "sync", "--config", cfg, "--prune"]) == 2
    assert "refusing to prune" in capsys.readouterr().err
    assert set(w.installed()) == {"stt/m/onnx"}  # untouched
    assert cli_main(["--index", index, "sync", "--config", cfg]) == 0  # without --prune: fine


def test_cli_sync_missing_config_is_an_actionable_error(store, tmp_path, capsys):
    # --index, so the run never falls back to the remote DEFAULT_INDEX_SOURCES: the
    # index load happens BEFORE the config is read, and this asserts on the config error.
    index = _write_index(tmp_path, {})
    assert cli_main(["--index", index, "sync", "--config", str(tmp_path / "nope.json")]) == 2
    assert "cannot read nanobot config" in capsys.readouterr().err


def test_cli_list_survives_an_unreachable_builtin_index(store, tmp_path, monkeypatch, capsys):
    """Offline, `list` must still show the local store: the default index is a URL."""
    def _boom(source):
        raise OSError("offline")

    monkeypatch.setattr(w, "_read_source", _boom)
    (store / "stt" / "m" / "onnx").mkdir(parents=True)
    (store / "stt" / "m" / "onnx" / w.MANIFEST).write_text(json.dumps({"files": {}}))

    assert cli_main(["list"]) == 0                       # degrades, does not fail
    cap = capsys.readouterr()
    assert "stt/m/onnx" in cap.out and "not in index" in cap.out
    assert "warning" in cap.err
    # A source the USER named still hard-errors: they asked for that one specifically.
    assert cli_main(["--index", "https://example.invalid/i.json", "list"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_env_index_default(store, tmp_path, monkeypatch, capsys):
    src = _src(tmp_path, "encoder.onnx")
    monkeypatch.setenv(
        "NANOBOT_VOICE_INDEX", _write_index(tmp_path, {"stt/m/onnx": _entry_for(src)})
    )
    assert cli_main(["list"]) == 0
    assert "stt/m/onnx" in capsys.readouterr().out

