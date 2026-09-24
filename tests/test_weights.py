"""Weight store: index merge, fetch (http + file://), prune, path resolution, CLI.

Everything runs against a tmp store via $NANOBOT_VOICE_MODELS_DIR; http
downloads are faked at urllib so no test touches the network.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import threading
import urllib.request

import pytest

from nanobot_channel_voice import weights as w
from nanobot_channel_voice.cli import main as cli_main


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


def test_no_sources_is_no_index_and_the_built_in_one_is_urls_only(monkeypatch):
    """The wheel ships no index DATA, only the URL of a community-served one, which is
    ``channels.voice.index``'s default: a load reads exactly the sources it is given, none
    being no index at all, and a source given replaces the built-in one."""
    from nanobot_channel_voice.config import VoiceConfig

    assert w.DEFAULT_INDEX_SOURCES
    assert all(s.startswith("https://") for s in w.DEFAULT_INDEX_SOURCES)
    assert VoiceConfig().index == list(w.DEFAULT_INDEX_SOURCES)
    seen = []
    monkeypatch.setattr(w, "_read_source", lambda s, _timeout: seen.append(s) or {"models": {}})
    assert w.load_index([]) == {} and seen == []
    w.load_index(["https://example.invalid/i.json"])
    assert seen == ["https://example.invalid/i.json"]


def test_a_file_url_relative_to_its_index_follows_the_index(monkeypatch):
    """A relative file url resolves against the index that listed it, so a copy of the
    index on a mirror or a disk serves its files from there; an absolute one stays as
    written, which is also why a copy of an index of absolute urls still downloads from
    where it came from."""
    listed = {"models": {"stt/m/onnx": {"files": {
        "encoder.onnx": {"url": "models/stt/m/onnx/encoder.onnx", "sha256": "0" * 64},
        "vocab.txt": {"url": "https://cdn.example/vocab.txt", "sha256": "0" * 64},
    }}}}
    monkeypatch.setattr(w, "_read_source", lambda _s, _t: json.loads(json.dumps(listed)))
    for source in (
        "https://huggingface.co/o/r/resolve/main/weights-index.json",
        "https://hf-mirror.com/o/r/resolve/main/weights-index.json",
        "file:///mnt/usb/weights-index.json",
    ):
        files = w.load_index([source])["stt/m/onnx"]["files"]
        beside = source.rsplit("/", 1)[0]
        assert files["encoder.onnx"]["url"] == f"{beside}/models/stt/m/onnx/encoder.onnx"
        assert files["vocab.txt"]["url"] == "https://cdn.example/vocab.txt"


def test_a_relative_index_on_disk_links_the_files_beside_it(store, tmp_path):
    """A bare path is the file it names: an index copied to a disk with its files next to
    it installs from there, the store linking each file in place."""
    disk = tmp_path / "usb"
    (disk / "models").mkdir(parents=True)
    blob = b"weights!"
    (disk / "models" / "encoder.onnx").write_bytes(blob)
    (disk / "weights-index.json").write_text(json.dumps({"models": {"stt/m/onnx": {"files": {
        "encoder.onnx": {"url": "models/encoder.onnx", "sha256": hashlib.sha256(blob).hexdigest()},
    }}}}))
    entry = w.load_index([str(disk / "weights-index.json")])["stt/m/onnx"]
    assert entry["files"]["encoder.onnx"]["url"] == (disk / "models" / "encoder.onnx").as_uri()
    installed = w.fetch("stt/m/onnx", entry) / "encoder.onnx"
    assert installed.is_symlink() and installed.resolve() == (disk / "models" / "encoder.onnx").resolve()
    # through a link it is the link's neighbours, a path and its file:// URL alike
    link = tmp_path / "linked" / "weights-index.json"
    link.parent.mkdir()
    link.symlink_to(disk / "weights-index.json")
    for named in (str(link), link.as_uri()):
        files = w.load_index([named])["stt/m/onnx"]["files"]
        assert files["encoder.onnx"]["url"] == (link.parent / "models" / "encoder.onnx").as_uri()


def test_an_index_arrives_authenticated(monkeypatch):
    """The index pins every file's sha256, so it is read over https, from a file, or over
    plain http from this machine alone; the files it lists may come over anything."""
    monkeypatch.setattr(w, "_read_source", lambda _s, _t: {"models": {}})
    for fine in (
        "https://a.example/i.json", "file:///srv/i.json", "/srv/i.json", "i.json",
        "http://127.0.0.1:8000/i.json", "http://localhost/i.json", "http://[::1]/i.json",
    ):
        w.load_index([fine])
    for refused, why in (
        ("http://mirror.lan/i.json", "must be https or a file"),
        ("http://10.0.0.2/i.json", "must be https or a file"),
        ("http://localhost.example/i.json", "must be https or a file"),
        ("ftp://a.example/i.json", "not an index scheme"),
    ):
        with pytest.raises(w.WeightsError, match=why) as caught:
            w.load_index([refused])
        assert refused in str(caught.value)  # the refusal names the index it refused


def test_a_redirect_does_not_downgrade_an_index(monkeypatch):
    """urllib follows a redirect from https down to plain http, so where the index lands
    is held to the rule the index it was named as is."""
    class Landed(io.BytesIO):
        url = "http://mirror.lan/weights-index.json"  # where urlopen's answer came from

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: Landed(b'{"models": {}}'))
    named = "https://hub.example/weights-index.json"
    with pytest.raises(w.WeightsError, match=f"'{named}': it redirects to {Landed.url}, and an index must be https"):
        w.load_index([named])
    Landed.url = "https://cdn.example/c/abc/weights-index.json"  # still https: fine
    assert w.load_index([named]) == {}


def test_an_index_a_vendor_wrote_loosely_is_refused_or_survived(store, tmp_path, capsys):
    """Index entries are external input the FORM reads on every validate: a size that is
    not a number is refused at load (an actionable line), and a wake key with no phrase
    segment does not break the backbone derivation the whole load runs through."""
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"models": {
        "stt/m/onnx": {"files": {"e.onnx": {"url": "https://x/e", "sha256": "0" * 64, "size": "7MB"}}},
    }}))
    with pytest.raises(w.WeightsError, match=r"size must be a whole number"):
        w.load_index([str(path)])
    assert cli_main(["--index", str(path), "list"]) == 2
    assert "error:" in capsys.readouterr().err
    # wake/openwakeword/<platform>: a valid key with no stem, so no backbone to derive
    path.write_text(json.dumps({"models": {
        "wake/openwakeword/onnx": {"files": {"m.onnx": {"url": "https://x/m", "sha256": "0" * 64}}},
    }}))
    assert set(w.load_index([str(path)])) == {"wake/openwakeword/onnx"}
    assert w.entry_size({"files": {"a": {"size": None}, "b": {"size": 5}}}) == 5


def test_a_torn_manifest_is_not_an_installed_model(store, tmp_path):
    """A power cut between the write and the flush can leave an empty manifest. Such a key
    must read as not fetched — else the plan says "Models are in place" while the engine
    cannot resolve a single path."""
    src = _src(tmp_path, "encoder.onnx")
    w.fetch("stt/m/onnx", _entry_for(src))
    manifest = w.store_dir("stt/m/onnx") / w.MANIFEST
    assert set(w.installed()) == {"stt/m/onnx"}
    for torn in ("", "null", "[]", '{"key": "stt/m/onnx"}'):
        manifest.write_text(torn)
        assert w.installed() == {}
        from nanobot_channel_voice.config import SttConfig

        with pytest.raises(w.NotFetchedError):
            w.fill_engine_paths(SttConfig.model_validate({"whisper": {"weights": "stt/m/onnx"}}).whisper)


def test_a_store_that_cannot_be_written_says_so(store, tmp_path, monkeypatch):
    """The gateway reloads the index in the background; a store it cannot write must say
    which store, not raise an OSError nobody catches."""
    path = tmp_path / "index.json"
    path.write_text(json.dumps({"models": {}}))
    (tmp_path / "blocked").write_text("not a directory")
    monkeypatch.setenv("NANOBOT_VOICE_MODELS_DIR", str(tmp_path / "blocked"))
    with pytest.raises(w.WeightsError, match="cannot cache the weights index"):
        w.refresh_index([str(path)])


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
        ({"stt/m/onnx": {"files": {"encoder.onnx": {"url": "https://[x.test/e"}}}}, r"\.url must be a URL"),
        ({"stt/m/onnx": {"files": {"encoder.onnx": {"url": 7}}}}, r"\.url must be a URL"),
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


def test_an_update_that_fails_leaves_the_installed_revision_whole(store, monkeypatch):
    """Nothing moves in until every file verifies: a later file failing (a bad pin, a
    cancel) leaves the installed revision intact, never half new."""
    served = {}
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: io.BytesIO(served[url]))

    def revision(tag, meta_pin=None):
        blobs = {"encoder.rknn": b"encoder " + tag, "meta.json": b"meta " + tag}
        served.update({f"https://x.test/{n}": b for n, b in blobs.items()})
        pins = {**blobs, "meta.json": meta_pin or blobs["meta.json"]}
        return {"files": {n: {"url": f"https://x.test/{n}", "sha256": hashlib.sha256(pins[n]).hexdigest()}
                          for n in blobs}}

    key = "tts/m/rknn.rv1126b"
    d = w.fetch(key, revision(b"v1"))
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    with pytest.raises(w.WeightsError, match="meta.json: sha256 mismatch"):
        w.fetch(key, revision(b"v2", meta_pin=b"meta v3"))
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before
    seen = []
    with pytest.raises(w.WeightsError, match="meta.json: download cancelled"):
        w.fetch(key, revision(b"v2"), progress=lambda name, _n: seen.append(name),
                should_stop=lambda: "encoder.rknn" in seen)
    assert {p.name: p.read_bytes() for p in d.iterdir()} == before
    w.fetch(key, revision(b"v2"))
    assert (d / "encoder.rknn").read_bytes() == b"encoder v2" and (d / "meta.json").read_bytes() == b"meta v2"


def test_a_model_keeps_whoever_installed_it(store, tmp_path):
    """A re-check or an update, forced or not, leaves the tag as the install set it: the
    CLI does not take the panel's models, nor the panel's update the user's."""
    src = _src(tmp_path, "encoder.onnx", b"e1")
    w.fetch("stt/panel/onnx", _entry_for(src), managed_by="webui")
    w.fetch("stt/panel/onnx", _entry_for(src))
    w.fetch("stt/panel/onnx", _entry_for(src), force=True)
    assert w.managed_by("stt/panel/onnx") == "webui"
    w.fetch("stt/mine/onnx", _entry_for(src))
    src.write_bytes(b"e2")
    w.fetch("stt/mine/onnx", _entry_for(src), managed_by="webui")
    assert (w.store_dir("stt/mine/onnx") / "encoder.onnx").read_bytes() == b"e2"
    assert w.managed_by("stt/mine/onnx") is None


def test_a_partial_left_under_this_pid_is_never_written_through(store, monkeypatch, tmp_path):
    """Under a pid reused after a reboot, a crashed run's partial may be a staged link:
    the download replaces it rather than writing into its target."""
    src = _src(tmp_path, "model.onnx", b"the user's own copy")
    blob = b"remote-model-bytes"
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: io.BytesIO(blob))
    d = w.store_dir("vad/firered/onnx")
    d.mkdir(parents=True)
    (d / f".partial-{os.getpid()}-model.onnx").symlink_to(src)
    entry = {"files": {"model.onnx": {"url": "https://x.test/m", "sha256": hashlib.sha256(blob).hexdigest()}}}
    w.fetch("vad/firered/onnx", entry)
    assert src.read_bytes() == b"the user's own copy" and (d / "model.onnx").read_bytes() == blob


def test_a_fetch_first_frees_what_a_dead_run_left(store, tmp_path):
    """A dead pid's partials (a crash, a power cut) go before the download: a key with no
    manifest cannot be pruned. A live fetcher's stay."""
    src = _src(tmp_path, "encoder.onnx")
    d = w.store_dir("stt/m/onnx")
    (d / ".partial-4194305-pack.tar.bz2.d" / "espeak-ng-data").mkdir(parents=True)  # above any pid_max
    (d / ".partial-4194305-decoder.onnx").write_bytes(b"x")
    live = d / f".partial-{os.getppid()}-decoder.onnx"
    live.write_bytes(b"y")
    w.fetch("stt/m/onnx", _entry_for(src))
    assert sorted(p.name for p in d.iterdir()) == sorted([w.MANIFEST, live.name, "encoder.onnx"])


def _tar(members: dict) -> bytes:
    """A .tar.bz2 of name -> bytes (a file), None (a directory) or a str (a link to it)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tar:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            if body is None:
                info.type = tarfile.DIRTYPE
            elif isinstance(body, str):
                info.type, info.linkname = tarfile.SYMTYPE, body
            else:
                info.size = len(body)
            tar.addfile(info, io.BytesIO(body) if isinstance(body, bytes) else None)
    return buf.getvalue()


def _packed(served, pack, name="espeak-ng-data.tar.bz2"):
    """An entry of one ``extract`` archive, served at https://x.test/pack."""
    served["https://x.test/pack"] = pack
    sha = hashlib.sha256(pack).hexdigest()
    return {"files": {name: {"url": "https://x.test/pack", "sha256": sha, "extract": True}}}


def test_an_extract_archive_lands_as_the_directory_it_unpacks_to(store, monkeypatch, tmp_path):
    """The directory stays, the archive does not: a refetch downloads nothing, a new
    revision replaces the directory whole, and a local archive unpacks rather than links."""
    served, calls = {}, []
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: calls.append(url) or io.BytesIO(served[url]))
    key = "tts/m/rknn.rv1126b"
    v1 = _packed(served, _tar({"espeak-ng-data": None, "espeak-ng-data/phondata": b"v1", "espeak-ng-data/voices/!v/adam": b"a"}))
    d = w.fetch(key, v1)
    assert sorted(p.name for p in d.iterdir()) == [w.MANIFEST, "espeak-ng-data"]
    assert (d / "espeak-ng-data" / "phondata").read_bytes() == b"v1"
    assert json.loads((d / w.MANIFEST).read_text())["files"]["espeak-ng-data.tar.bz2"]["unpacked"] == "espeak-ng-data"
    w.fetch(key, v1)
    assert len(calls) == 1 and (d / "espeak-ng-data" / "voices").is_dir()
    w.fetch(key, _packed(served, _tar({"espeak-ng-data/phondata": b"v2"})))
    assert (d / "espeak-ng-data" / "phondata").read_bytes() == b"v2"
    assert not (d / "espeak-ng-data" / "voices").exists()  # replaced whole, never merged
    src = tmp_path / "espeak-ng-data.tar.bz2"
    src.write_bytes(_tar({"espeak-ng-data/phondata": b"local"}))
    local = {"files": {src.name: {"url": src.as_uri(), "extract": True}}}
    unpacked = w.fetch("tts/l/onnx", local) / "espeak-ng-data"
    assert not unpacked.is_symlink() and (unpacked / "phondata").read_bytes() == b"local"


def test_an_archive_unpacks_in_one_pass(tmp_path):
    """A pipe reads forward only: listing then extracting would seek back, which in a
    compressed stream means decompressing it twice."""
    pipe = tmp_path / "espeak-ng-data.tar.bz2"
    os.mkfifo(pipe)
    pack = _tar({"espeak-ng-data/phondata": b"v1"})
    writer = threading.Thread(target=pipe.write_bytes, args=(pack,), daemon=True)
    writer.start()
    w._unpack(pipe, *w._tar_parts(pipe.name), tmp_path / "out")
    writer.join(timeout=10)
    assert (tmp_path / "out" / "espeak-ng-data" / "phondata").read_bytes() == b"v1"


def test_an_archive_must_make_one_directory_of_files(store, monkeypatch):
    """A member beside the directory, one escaping it, or a link fails the key and leaves
    nothing; so does extract on a name that is no tar archive."""
    served = {}
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: io.BytesIO(served[url]))
    for members in (
        {"espeak-ng-data/a": b"x", "tokens.txt": b"beside"},
        {"espeak-ng-data/../../escaped": b"x"},
        {"espeak-ng-data/phondata": "/etc/passwd"},
    ):
        with pytest.raises(w.WeightsError, match="espeak-ng-data.tar.bz2: cannot unpack: it holds"):
            w.fetch("tts/m/onnx", _packed(served, _tar(members)))
        assert list(w.store_dir("tts/m/onnx").iterdir()) == []
    with pytest.raises(w.WeightsError, match="marks 'pack.zip' extract, but only a .tar"):
        w.fetch("tts/m/onnx", _packed(served, b"zip", name="pack.zip"))


def test_a_python_without_the_codec_installs_the_archive_packed(store, monkeypatch):
    """Without bz2 (minimal builds) the archive installs packed and a hand-unpacked
    directory survives refetches; once the codec exists, a fetch unpacks it in place
    without downloading."""
    served, calls = {}, []
    entry = _packed(served, _tar({"espeak-ng-data/phondata": b"v1"}))
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=0: calls.append(url) or io.BytesIO(served[url]))
    monkeypatch.setitem(sys.modules, "bz2", None)
    lines = []
    d = w.fetch("tts/m/onnx", entry, log=lines.append)
    assert (d / "espeak-ng-data.tar.bz2").read_bytes() == served["https://x.test/pack"]
    assert any("left packed (bz2 module is not available)" in line for line in lines)
    (d / "espeak-ng-data").mkdir()  # unpacked by hand
    w.fetch("tts/m/onnx", entry)
    assert (d / "espeak-ng-data").is_dir() and (d / "espeak-ng-data.tar.bz2").is_file()
    sys.modules.pop("bz2")  # the codec arrives
    w.fetch("tts/m/onnx", entry)
    assert (d / "espeak-ng-data" / "phondata").read_bytes() == b"v1"
    assert not (d / "espeak-ng-data.tar.bz2").exists() and len(calls) == 1


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


def test_prune_never_follows_a_relocation_symlink(store, tmp_path):
    """installed() follows symlinks because users relocate subtrees that way; prune must
    drop a symlinked LEAF as a link (never rmtree into the target) and stop at a
    symlinked ANCESTOR (the user's relocation, not store scaffolding)."""
    src = _src(tmp_path, "encoder.onnx")
    w.fetch("vad/silero/onnx", _entry_for(src))
    leaf = store / "vad" / "silero" / "onnx"
    target = tmp_path / "elsewhere"
    leaf.rename(target)
    leaf.symlink_to(target)
    assert set(w.installed()) == {"vad/silero/onnx"}
    link_size = leaf.lstat().st_size
    assert w.relocation_target("vad/silero/onnx") == target
    assert w.prune("vad/silero/onnx") == link_size  # the link's bytes, not the target's
    assert not leaf.is_symlink() and not leaf.exists()
    assert (target / w.MANIFEST).is_file()  # untouched
    assert not (store / "vad").exists()     # the store's own scaffolding still goes
    assert w.relocation_target("vad/silero/onnx") is None

    w.fetch("tts/mms/eng/onnx", _entry_for(src))
    ancestor = store / "tts"
    big = tmp_path / "big-disk-tts"
    ancestor.rename(big)
    ancestor.symlink_to(big)
    assert set(w.installed()) == {"tts/mms/eng/onnx"}
    assert w.prune("tts/mms/eng/onnx") > 0
    assert w.installed() == {}
    assert ancestor.is_symlink() and big.is_dir() and not any(big.iterdir())


def test_prune_drops_a_dangling_relocation_link(store, tmp_path, capsys):
    """The relocated target is gone (USB store unplugged, deleted): the link itself is
    the only thing left to prune, and "not in the store" would strand it."""
    src = _src(tmp_path, "encoder.onnx")
    w.fetch("vad/silero/onnx", _entry_for(src))
    leaf = store / "vad" / "silero" / "onnx"
    target = tmp_path / "gone"
    leaf.rename(target)
    leaf.symlink_to(target)
    import shutil

    shutil.rmtree(target)
    assert leaf.is_symlink() and not leaf.exists()
    assert cli_main(["prune", "vad/silero/onnx", "--yes"]) == 0
    assert "dangling link" in capsys.readouterr().out
    assert not leaf.is_symlink()


def test_fetch_refuses_a_dangling_relocation_link(store, tmp_path):
    """mkdir raises FileExistsError on a link whose target is gone, so Apply would fail
    with a bare OSError; the store names the key and prune is the way out."""
    import shutil

    src = _src(tmp_path, "encoder.onnx")
    w.fetch("vad/silero/onnx", _entry_for(src))
    leaf = store / "vad" / "silero" / "onnx"
    target = tmp_path / "gone"
    leaf.rename(target)
    leaf.symlink_to(target)
    shutil.rmtree(target)
    with pytest.raises(w.WeightsError, match="prune it first"):
        w.fetch("vad/silero/onnx", _entry_for(src))


def test_installed_stops_at_a_relocation_that_links_back(store, tmp_path):
    """The walk follows symlinks, which is how a relocated subtree stays visible; a
    target that links back into the store would else walk forever in a gateway thread."""
    src = _src(tmp_path, "encoder.onnx")
    w.fetch("vad/silero/onnx", _entry_for(src))
    leaf = store / "vad" / "silero" / "onnx"
    target = tmp_path / "elsewhere"
    leaf.rename(target)
    leaf.symlink_to(target)
    (target / "loop").symlink_to(store)
    assert set(w.installed()) == {"vad/silero/onnx"}


def test_cli_prune_names_where_a_relocated_key_still_lives(store, tmp_path, capsys):
    src = _src(tmp_path, "encoder.onnx")
    w.fetch("vad/silero/onnx", _entry_for(src))
    leaf = store / "vad" / "silero" / "onnx"
    target = tmp_path / "elsewhere"
    leaf.rename(target)
    leaf.symlink_to(target)
    assert cli_main(["prune", "vad/silero/onnx", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "unlinked vad/silero/onnx" in out and str(target) in out
    assert (target / w.MANIFEST).is_file()


def test_cli_reports_a_filesystem_failure_as_a_clean_error(store, tmp_path, monkeypatch, capsys):
    src = _src(tmp_path, "encoder.onnx")
    w.fetch("vad/silero/onnx", _entry_for(src))

    def locked(path, *a, **k):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(w.shutil, "rmtree", locked)
    assert cli_main(["prune", "vad/silero/onnx", "--yes"]) == 2
    assert "error: [Errno 13] Permission denied" in capsys.readouterr().err


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


def test_a_head_of_your_own_runs_on_the_stores_backbone(store, tmp_path):
    """openWakeWord's feature models are one pair per platform, shared by every head: the
    index gains a derived backbone entry per platform (the head's files minus the head and
    its meta), a custom head resolves them from that package, the device build first, and
    the plan wants that package for such a section."""
    from nanobot_channel_voice.config import VoiceConfig, WakeConfig
    from nanobot_channel_voice.sync import backbone_wanted, plan_sync

    mel, emb, head, meta = (_src(tmp_path, n, n.encode()) for n in ("mel.onnx", "embedding.onnx", "model.onnx", "meta.json"))
    filt, emb_rknn = (_src(tmp_path, n, n.encode()) for n in ("mel_filters.npy", "embedding.rknn"))
    index = w.with_backbones({
        "wake/openwakeword/alexa/onnx": _entry_for(mel, emb, head, meta),
        "wake/openwakeword/hey-jarvis/onnx": _entry_for(mel, emb, head, meta),
        "wake/openwakeword/alexa/rknn.rv1126b": _entry_for(filt, emb_rknn, head, meta),
        "stt/whisper/base/onnx": _entry_for(head),
    })
    assert set(index) - {"stt/whisper/base/onnx"} == {
        "wake/openwakeword/alexa/onnx", "wake/openwakeword/hey-jarvis/onnx", "wake/openwakeword/alexa/rknn.rv1126b",
        "wake/openwakeword/backbone/onnx", "wake/openwakeword/backbone/rknn.rv1126b",
    }
    assert set(index["wake/openwakeword/backbone/onnx"]["files"]) == {"mel.onnx", "embedding.onnx"}
    assert set(index["wake/openwakeword/backbone/rknn.rv1126b"]["files"]) == {"mel_filters.npy", "embedding.rknn"}
    assert w.with_backbones(index) == index  # idempotent, and an index's own entry is kept
    own = {**index, "wake/openwakeword/backbone/onnx": {"files": {"x": {"url": "u"}}}}
    assert w.with_backbones(own)["wake/openwakeword/backbone/onnx"] == {"files": {"x": {"url": "u"}}}

    # the section: a head of your own, no key; the plan wants the platform's backbone
    section = {"wake": {"mode": "gate", "phrases": ["mine"], "engine": "openwakeword", "openwakeword": {"modelPath": "/heads/mine.onnx"}}}

    def wanted(values, models=index):
        return backbone_wanted(VoiceConfig.model_validate(values), models)

    assert wanted(section) == "wake/openwakeword/backbone/onnx"
    assert wanted({"device": "rv1126b", **section}) == "wake/openwakeword/backbone/rknn.rv1126b"
    assert wanted({"wake": {"mode": "off"}}) is None
    # an index without the chip's build: the CPU pair is what it can fetch, and is named
    assert wanted({"device": "rv1126b", **section}, {"wake/openwakeword/backbone/onnx": {}}) == "wake/openwakeword/backbone/onnx"
    # a refused section (a gate without its phrase) plans no backbone
    assert plan_sync({"wake": {"mode": "gate"}}, index, store, managed_by=None).wanted == []
    plan = plan_sync({"device": "rv1126b", **section}, index, store, managed_by=None)
    assert plan.wanted == plan.fetch == ["wake/openwakeword/backbone/rknn.rv1126b"]

    # at start: not fetched names the fetch; fetched fills the pair, the head stays yours
    cfg = VoiceConfig.model_validate(section)
    with pytest.raises(w.WeightsError, match="nanobot-voice fetch wake/openwakeword/backbone/onnx"):
        w.apply_weights(cfg.wake, "openwakeword")
    w.fetch("wake/openwakeword/backbone/onnx", index["wake/openwakeword/backbone/onnx"])
    oww = w.apply_weights(cfg.wake, "openwakeword").openwakeword
    d = w.store_dir("wake/openwakeword/backbone/onnx")
    assert (oww.model_path, oww.mel_path, oww.embedding_path, oww.meta_path) == ("/heads/mine.onnx", str(d / "mel.onnx"), str(d / "embedding.onnx"), None)
    assert oww.mel_filters_path is None
    # a device host prefers its build once fetched, and falls back to the CPU pair until then
    board = VoiceConfig.model_validate({"device": "rv1126b", **section})
    assert w.apply_weights(board.wake, "openwakeword").openwakeword.embedding_path == str(d / "embedding.onnx")
    # the CPU pair in the store does not settle what a board wants: the plan still fetches
    # its build (the runtime prefers whichever is installed, the plan the host's platform)
    assert plan_sync({"device": "rv1126b", **section}, index, store, managed_by=None).fetch == [
        "wake/openwakeword/backbone/rknn.rv1126b"
    ]
    w.fetch("wake/openwakeword/backbone/rknn.rv1126b", index["wake/openwakeword/backbone/rknn.rv1126b"])
    oww = w.apply_weights(board.wake, "openwakeword").openwakeword
    assert oww.embedding_path.endswith("backbone/rknn.rv1126b/embedding.rknn") and oww.mel_filters_path.endswith("mel_filters.npy")
    # a block with the pair set by hand is left alone
    assert w.apply_weights(WakeConfig.model_validate({"openwakeword": {"modelPath": "/m.onnx", "embeddingPath": "/e.onnx"}}), "openwakeword").openwakeword.mel_path is None


def test_the_embedding_takes_the_npu_build_whenever_there_is_one(store, tmp_path):
    """The device's, from the backbone package the plan fetches beside a head in another
    build (whose own pair stands in until then), or with no device the head's own."""
    from nanobot_channel_voice.config import VoiceConfig
    from nanobot_channel_voice.sync import plan_sync

    mel, emb, head, meta = (_src(tmp_path, n, n.encode()) for n in ("mel.onnx", "embedding.onnx", "model.onnx", "meta.json"))
    filt, emb_rknn = (_src(tmp_path, n, n.encode()) for n in ("mel_filters.npy", "embedding.rknn"))
    cpu, chip = "wake/openwakeword/hey-jarvis/onnx", "wake/openwakeword/hey-jarvis/rknn.rv1126b"
    pair, cpu_pair = "wake/openwakeword/backbone/rknn.rv1126b", "wake/openwakeword/backbone/onnx"
    index = w.with_backbones({cpu: _entry_for(mel, emb, head, meta), chip: _entry_for(filt, emb_rknn, head, meta)})

    def section(key, device=None, **oww):
        wake = {"mode": "gate", "phrases": ["hey jarvis"], "engine": "openwakeword", "openwakeword": {"weights": key, **oww}}
        return {"wake": wake, **({"device": device} if device else {})}

    def wanted(values, **kw):
        return plan_sync(values, index, store, managed_by=None, **kw).wanted

    assert wanted(section(cpu, "rv1126b")) == [pair, cpu]
    assert wanted(section(chip, "rv1126b")) == wanted(section(chip)) == [chip]
    assert wanted(section(chip, "rk3588")) == [cpu_pair, chip]  # another chip's build is no option
    assert wanted(section(cpu)) == [cpu]
    assert wanted(section(cpu, "rv1126b", embeddingPath="/e.rknn")) == [cpu]
    assert wanted({**section(cpu, "rv1126b"), "backend": "openai"}, used_only=True) == []

    board = VoiceConfig.model_validate(section(cpu, "rv1126b")).wake
    w.fetch(cpu, index[cpu])
    d, p = w.store_dir(cpu), w.store_dir(pair)
    assert w.apply_weights(board, "openwakeword").openwakeword.embedding_path == str(d / "embedding.onnx")
    w.fetch(pair, index[pair])
    oww = w.apply_weights(board, "openwakeword").openwakeword
    assert (oww.mel_path, oww.mel_filters_path, oww.embedding_path) == (None, str(p / "mel_filters.npy"), str(p / "embedding.rknn"))
    assert (oww.model_path, oww.meta_path) == (str(d / "model.onnx"), str(d / "meta.json"))
    cpu_host = VoiceConfig.model_validate(section(cpu)).wake
    assert w.apply_weights(cpu_host, "openwakeword").openwakeword.embedding_path == str(d / "embedding.onnx")
    assert w.backbone_key("rk 3588", cpu) == cpu  # a block target no key can spell
    w.fetch(chip, index[chip])
    w.fetch(cpu_pair, index[cpu_pair])
    for device in ("rv1126b", None):
        own = w.apply_weights(VoiceConfig.model_validate(section(chip, device)).wake, "openwakeword").openwakeword
        assert own.embedding_path == str(w.store_dir(chip) / "embedding.rknn")


def test_a_mel_frontend_set_by_hand_holds_the_slot(store, tmp_path):
    """Whichever file the package carries: two mel frontends refuse to start."""
    from nanobot_channel_voice.config import VoiceConfig

    files = (_src(tmp_path, n, n.encode()) for n in ("mel.onnx", "embedding.onnx", "model.onnx", "mel_filters.npy", "embedding.rknn"))
    mel, emb, head, filt, emb_rknn = files
    cpu, pair = "wake/openwakeword/hey-jarvis/onnx", "wake/openwakeword/backbone/rknn.rv1126b"
    w.fetch(cpu, _entry_for(mel, emb, head))
    w.fetch(pair, _entry_for(filt, emb_rknn))

    def resolve(device=None, **oww):
        cfg = VoiceConfig.model_validate({"wake": {"openwakeword": {"weights": cpu, **oww}}, **({"device": device} if device else {})})
        return w.apply_weights(cfg.wake, "openwakeword").openwakeword

    mine = resolve("rv1126b", melPath="/m.onnx")
    assert (mine.mel_path, mine.mel_filters_path, mine.embedding_path) == ("/m.onnx", None, str(w.store_dir(pair) / "embedding.rknn"))
    mine = resolve(embeddingPath="/e.rknn", melFiltersPath="/f.npy")
    assert (mine.mel_path, mine.mel_filters_path, mine.model_path) == (None, "/f.npy", str(w.store_dir(cpu) / "model.onnx"))


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


def test_hybrid_wake_package_resolves_without_a_mel_graph(store, tmp_path):
    from nanobot_channel_voice.config import OpenWakeWordConfig

    # The published RV1126B package shape: python-mel filterbank, no mel graph.
    key = "wake/openwakeword/hey-mycroft/rknn.rv1126b"
    w.fetch(key, _entry_for(_src(tmp_path, "embedding.rknn"),
                            _src(tmp_path, "mel_filters.npy"),
                            _src(tmp_path, "meta.json"),
                            _src(tmp_path, "model.onnx")))
    filled = w.fill_engine_paths(OpenWakeWordConfig.model_validate({"weights": key}))
    d = w.store_dir(key)
    assert filled.embedding_path == str(d / "embedding.rknn")
    assert filled.model_path == str(d / "model.onnx")
    assert filled.mel_filters_path == str(d / "mel_filters.npy")
    assert filled.meta_path == str(d / "meta.json")
    assert filled.mel_path is None  # "mel.*" must not swallow mel_filters.npy


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


def _write_index(tmp_path, models, name="index.json"):
    p = tmp_path / name
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


def test_cli_sync_fetches_the_operator_baseline_beneath_the_section(store, tmp_path, capsys, monkeypatch):
    """The layer under channels.voice reaches the structural scan too: a fresh section
    syncs the baseline's models, and a section over it names its own keys as well (sync
    fetches every named key, the baseline's included; Apply plans what runs)."""
    src = _src(tmp_path, "encoder.onnx")
    index = _write_index(tmp_path, {
        "stt/sensevoice/small/onnx": _entry_for(src),
        "stt/whisper-base/onnx": _entry_for(src),
        "vad/silero/v6/onnx": _entry_for(src),
    })
    baseline = tmp_path / "voice-defaults.json"
    baseline.write_text(json.dumps({
        "stt": {"provider": "sensevoice", "sensevoice": {"weights": "stt/sensevoice/small/onnx"}},
        "vad": {"engine": "silero", "silero": {"weights": "vad/silero/v6/onnx"}},
    }))
    monkeypatch.setenv("NANOBOT_VOICE_DEFAULTS", str(baseline))
    cfg = _write_config(tmp_path, {"enabled": True, "importJson": ""})  # as onboarding leaves it
    assert cli_main(["--index", index, "sync", "--config", cfg]) == 0
    assert set(w.installed()) == {"stt/sensevoice/small/onnx", "vad/silero/v6/onnx"}
    cfg = _write_config(tmp_path, {"stt": {"provider": "whisper", "whisper": {"weights": "stt/whisper-base/onnx"}}})
    assert cli_main(["--index", index, "sync", "--config", cfg, "--prune"]) == 0
    assert set(w.installed()) == {"stt/sensevoice/small/onnx", "stt/whisper-base/onnx", "vad/silero/v6/onnx"}
    monkeypatch.setenv("NANOBOT_VOICE_DEFAULTS", str(tmp_path / "gone.json"))
    assert cli_main(["--index", index, "sync", "--config", cfg]) == 2
    assert "NANOBOT_VOICE_DEFAULTS: cannot read" in capsys.readouterr().err


def test_cli_sync_names_configured_keys_missing_from_the_index(store, tmp_path, capsys):
    index = _write_index(tmp_path, {})
    cfg = _write_config(tmp_path, {"vad": {"firered": {"weights": "vad/firered/rknn.rk3588"}}})
    assert cli_main(["--index", index, "sync", "--config", cfg]) == 2
    assert "vad/firered/rknn.rk3588" in capsys.readouterr().err


def _stale(src):
    """An entry pinning a hash the file no longer has (an older index)."""
    entry = _entry_for(src)
    entry["files"][src.name]["sha256"] = "0" * 64
    return entry


def test_cli_sync_goes_past_a_key_that_fails_and_reports_each(store, tmp_path, capsys):
    """A bad pin or an unindexed key fails alone: every key is tried, a per-key report
    ends the run, the exit is non-zero, and nothing is pruned."""
    src = _src(tmp_path, "encoder.onnx")
    index = _write_index(tmp_path, {
        "stt/a/onnx": _entry_for(src),
        "tts/b/rknn.rv1126b": _stale(src),
        "vad/c/onnx": _entry_for(src),
        "wake/e/onnx": _entry_for(src),
    })
    assert cli_main(["--index", index, "fetch", "wake/e/onnx"]) == 0  # no longer configured
    cfg = _write_config(tmp_path, {
        "stt": {"whisper": {"weights": "stt/a/onnx"}},
        "tts": {"matcha": {"weights": "tts/b/rknn.rv1126b"}},
        "vad": {"silero": {"weights": "vad/c/onnx"}},
        "wake": {"openwakeword": {"weights": "wake/d/onnx"}},
    })
    capsys.readouterr()
    assert cli_main(["--index", index, "sync", "--config", cfg, "--prune"]) == 2
    out, err = capsys.readouterr()
    assert set(w.installed()) == {"stt/a/onnx", "vad/c/onnx", "wake/e/onnx"}
    assert "2 ok, 2 failed" in out
    assert "  failed   tts/b/rknn.rv1126b: encoder.onnx: sha256 mismatch" in out
    assert "  failed   wake/d/onnx: not in the index" in out
    assert "  ok       stt/a/onnx" in out and "  ok       vad/c/onnx" in out
    assert "error: 2 of 4 weights failed: tts/b/rknn.rv1126b, wake/d/onnx, nothing pruned" in err


def test_cli_fetch_of_several_keys_goes_past_one_that_fails(store, tmp_path, capsys):
    """Same for fetch; a token naming nothing is a typo, refused before any download."""
    src = _src(tmp_path, "encoder.onnx")
    index = _write_index(tmp_path, {"stt/a/onnx": _entry_for(src), "tts/b/onnx": _stale(src), "vad/c/onnx": _entry_for(src)})
    assert cli_main(["--index", index, "fetch", "stt/a/onnx", "stt/nope"]) == 2
    assert "unknown weights key 'stt/nope'" in capsys.readouterr().err and w.installed() == {}
    assert cli_main(["--index", index, "fetch", "tts/b/onnx", "stt/a/onnx", "vad/c/onnx"]) == 2
    out, err = capsys.readouterr()
    assert set(w.installed()) == {"stt/a/onnx", "vad/c/onnx"}
    assert "2 ok, 1 failed" in out and "  failed   tts/b/onnx: encoder.onnx: sha256 mismatch" in out
    assert "error: 1 of 3 weights failed: tts/b/onnx" in err
    assert cli_main(["--index", index, "fetch", "tts/b/onnx", "tts/b"]) == 2  # one key, named twice
    out, err = capsys.readouterr()
    assert err.startswith("error: 'tts/b/onnx' encoder.onnx: sha256 mismatch") and "failed" not in out


def test_a_renamed_keys_alias_installs_but_nothing_offers_it(store, tmp_path, capsys):
    """An alias installs by its exact name, noting its new key; prefixes and the list mean
    current keys, the alias listed once installed."""
    src = _src(tmp_path, "model.onnx")
    new, old = "vad/silero/v6/onnx", "vad/silero/onnx"
    index = _write_index(tmp_path, {new: _entry_for(src), old: {**_entry_for(src), "deprecated": True, "renamed_to": new}})
    assert cli_main(["--index", index, "list"]) == 0
    out = capsys.readouterr().out
    assert new in out and old not in out
    assert cli_main(["--index", index, "fetch", "vad/silero"]) == 0  # not ambiguous
    assert set(w.installed()) == {new}
    assert cli_main(["--index", index, "fetch", old]) == 0
    assert f"  DEPRECATED: renamed to {new}, name that instead" in capsys.readouterr().out
    assert cli_main(["--index", index, "list"]) == 0
    assert f"renamed to {new}" in capsys.readouterr().out


def test_cli_sync_leaves_an_installed_key_this_index_does_not_carry(store, tmp_path, capsys):
    """A model fetched from another index (or one a vendor has since dropped) is installed
    and configured: sync says it cannot re-verify it, rather than failing on it."""
    src = _src(tmp_path, "encoder.onnx")
    mine = _write_index(tmp_path, {"stt/whisper-base/onnx": _entry_for(src)}, name="mine.json")
    assert cli_main(["--index", mine, "fetch", "stt/whisper-base/onnx"]) == 0
    other = _write_index(tmp_path, {"tts/mms-deu/onnx": _entry_for(src)}, name="other.json")
    cfg = _write_config(tmp_path, {"stt": {"provider": "whisper", "whisper": {"weights": "stt/whisper-base/onnx"}}})
    assert cli_main(["--index", other, "sync", "--config", cfg]) == 0
    assert "stt/whisper-base/onnx: installed, not in this index" in capsys.readouterr().out
    assert set(w.installed()) == {"stt/whisper-base/onnx"}


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
    def _boom(source, _timeout):
        raise OSError("offline")

    monkeypatch.setattr(w, "_read_source", _boom)
    (store / "stt" / "m" / "onnx").mkdir(parents=True)
    (store / "stt" / "m" / "onnx" / w.MANIFEST).write_text(json.dumps({"files": {}}))

    assert cli_main(["list"]) == 0                       # degrades, does not fail
    cap = capsys.readouterr()
    assert "stt/m/onnx" in cap.out and "not in index" in cap.out
    assert "warning" in cap.err
    # A source the USER named still hard-errors: they asked for that one specifically,
    # here or in the config.
    assert cli_main(["--index", "https://example.invalid/i.json", "list"]) == 2
    assert "error:" in capsys.readouterr().err
    cfg = _write_config(tmp_path, {"index": ["https://example.invalid/i.json"]})
    assert cli_main(["list", "--config", cfg]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_reads_the_index_the_config_names(store, tmp_path, monkeypatch, capsys):
    """Without --index, list, fetch and sync read channels.voice.index from the config the
    gateway reads, so the CLI caches the index the panel shows; --index overrides it for
    one run, leaving that cache alone. With no config at all the index is the operator's
    defaults' or the built-in one, except for sync, which needs the config anyway and says
    so before downloading anything."""
    import nanobot.config.loader as loader

    src = _src(tmp_path, "encoder.onnx")
    index = _write_index(tmp_path, {"stt/m/onnx": _entry_for(src)})
    cfg = _write_config(tmp_path, {"index": [index]})
    assert cli_main(["list", "--config", cfg]) == 0
    assert "stt/m/onnx" in capsys.readouterr().out
    assert w.cached_index()[2] == [index]  # the same cache the panel reads, from the same index
    assert cli_main(["fetch", "stt/m/onnx", "--config", cfg]) == 0
    assert set(w.installed()) == {"stt/m/onnx"}
    other = _write_index(tmp_path, {"stt/n/onnx": _entry_for(src)}, name="other.json")
    assert cli_main(["--index", other, "list", "--config", cfg]) == 0
    assert "stt/n/onnx" in capsys.readouterr().out
    assert w.cached_index()[2] == [index]  # not the panel's index: the cache stays the config's
    # no --config: the file the gateway reads
    monkeypatch.setattr(loader, "get_config_path", lambda: tmp_path / "config.json")
    assert cli_main(["list"]) == 0
    assert "stt/m/onnx" in capsys.readouterr().out
    seen = []
    monkeypatch.setattr(w, "_read_source", lambda s, _t: seen.append(s) or {"models": {}})
    nowhere = str(tmp_path / "none.json")
    assert cli_main(["list", "--config", nowhere]) == 0
    assert seen == list(w.DEFAULT_INDEX_SOURCES)
    seen.clear()
    # a board's baseline names its own index: what the gateway resolves for a fresh config
    vendor = "https://vendor.example/weights-index.json"
    monkeypatch.setenv("NANOBOT_VOICE_DEFAULTS", json.dumps({"index": [vendor]}))
    assert cli_main(["list", "--config", nowhere]) == 0
    assert seen == [vendor]
    seen.clear()
    monkeypatch.setenv("NANOBOT_VOICE_DEFAULTS", str(tmp_path / "gone-defaults.json"))
    assert cli_main(["list", "--config", nowhere]) == 2
    assert "NANOBOT_VOICE_DEFAULTS: cannot read" in capsys.readouterr().err and seen == []
    monkeypatch.delenv("NANOBOT_VOICE_DEFAULTS")
    capsys.readouterr()
    assert cli_main(["sync", "--config", nowhere]) == 2
    assert "cannot read nanobot config" in capsys.readouterr().err and seen == []
    # an index the config names wrongly is the error, named with the config
    bad = _write_config(tmp_path, {"index": ["http://mirror.lan/i.json"]})
    assert cli_main(["list", "--config", bad]) == 2
    assert "channels.voice.index entry 'http://mirror.lan/i.json'" in capsys.readouterr().err



def test_apply_weights_resolves_the_bilingual_secondary(store, tmp_path):
    from nanobot_channel_voice.config import TtsConfig

    zh = [_src(tmp_path, n) for n in
          ("acoustic_model.onnx", "vocoder.onnx", "tokens.txt", "lexicon.txt")]
    w.fetch("tts/matcha/zh/onnx", _entry_for(*zh))
    en_dir = tmp_path / "en-src"
    en_dir.mkdir()
    en = [_src(en_dir, n) for n in ("acoustic_model.onnx", "tokens.txt")]
    w.fetch("tts/matcha/en/onnx", _entry_for(*en))

    cfg = TtsConfig.model_validate({
        "provider": "matcha",
        "matcha": {"weights": "tts/matcha/zh/onnx",
                   "secondary": {"weights": "tts/matcha/en/onnx"}},
    })
    filled = w.apply_weights(cfg, "matcha")
    assert filled.matcha.acoustic_model_path.endswith("acoustic_model.onnx")
    assert "tts/matcha/en" in filled.matcha.secondary.acoustic_model_path
    assert cfg.matcha.secondary.acoustic_model_path is None  # source cfg untouched
