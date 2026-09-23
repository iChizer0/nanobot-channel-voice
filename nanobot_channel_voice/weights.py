"""Local model-weight store: fetch, prune, and engine-path resolution.

The plugin never bundles model files. An INDEX (JSON) maps a weights key
(``<kind>/<model-path...>/<platform>``, e.g. ``stt/whisper/base/onnx``) to per-file URLs
plus sha256::

    {"version": 1, "models": {"stt/whisper/base/onnx": {
        "source": "https://... (where these files come from)",
        "license": "MIT", "accept": "non-commercial use only",
        "langs": ["en", "ja", "de"],
        "files": {"encoder.onnx": {"url": "https://...", "sha256": "...", "size": 42000000},
                  "decoder.onnx": {"url": "file:///srv/models/decoder.onnx"}}}}}

``accept`` makes ``fetch`` print the notice and demand confirmation (``--yes`` to
script it); per-file ``size`` (bytes) only feeds ``list``'s estimate. The wheel ships NO
entries and NO weights: they come from the index files ``channels.voice.index`` names
(:data:`DEFAULT_INDEX_SOURCES` unless it names others), or ``--index``. An index is read
over https or from a file, since it pins the hashes. A file ``url`` may be relative to
the index, so a copy of one serves its files from wherever it is put (a mirror, a disk).
``http(s)://`` files stream into the store and MUST pin a sha256; ``file://`` files are
symlinked in place, verified when the index pins one.

File names inside an entry are the resolution contract: an engine block setting
``weights: <key>`` gets its unset ``*_path`` fields filled by stem + any extension
(``encoder_path`` -> ``encoder.<ext>``); explicit paths always win. The network is used
only by the CLI and the WebUI's sync flow (``webui_sync``), never at channel start:
:func:`apply_weights` touches only the local store. The last loaded index is cached in
the store (:data:`INDEX_CACHE`) so the WebUI form can offer keys offline.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import re
import shutil
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

MANIFEST = ".manifest.json"
INDEX_CACHE = ".index.json"

# ``channels.voice.index`` unless it names others. URLs only, never a bundled data file
# (the wheel ships no entries); a configured index replaces it.
DEFAULT_INDEX_SOURCES: tuple[str, ...] = (
    "https://huggingface.co/iChizer0/nanobot-channel-voice-models/resolve/main/weights-index.json",
)

_KEY_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")  # no dot-prefix: kills "..", hidden dirs
_CHUNK = 1 << 20


class WeightsError(RuntimeError):
    """Actionable store/index failure; the message is user-facing."""


class NotFetchedError(WeightsError):
    """A ``weights`` key the store does not hold yet."""

    def __init__(self, key: str) -> None:
        super().__init__(f"weights '{key}' are not fetched; run: nanobot-voice fetch {key}")
        self.key = key


def store_root() -> Path:
    """``$NANOBOT_VOICE_MODELS_DIR``, else ``$XDG_DATA_HOME|~/.local/share/nanobot-voice/models``."""
    env = os.environ.get("NANOBOT_VOICE_MODELS_DIR")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "nanobot-voice" / "models"


def validate_key(key: str) -> str:
    """``<kind>/<model-path...>/<platform>``, >=3 segments: kind first, platform last."""
    parts = key.split("/")
    if len(parts) < 3 or not all(_KEY_SEGMENT.fullmatch(p) for p in parts):
        raise WeightsError(
            f"invalid weights key '{key}': expected <kind>/<model-path>/<platform>, "
            "e.g. stt/whisper/base/onnx or tts/matcha/en-US/ljspeech/rknn.rv1126b"
        )
    return key


def store_dir(key: str, root: Path | None = None) -> Path:
    return (root or store_root()).joinpath(*validate_key(key).split("/"))


# ---- index ------------------------------------------------------------------


def check_index_source(source: str) -> str:
    """``source`` when an index may be read from it. ``ValueError`` says why not: the
    index pins every file's sha256, so it has to arrive authenticated."""
    split = urllib.parse.urlsplit(source)
    if split.scheme not in ("", "http", "https", "file"):
        raise ValueError(f"'{split.scheme}' is not an index scheme (https://, file:// or a path)")
    if split.scheme == "http" and not _loopback(split.hostname):
        raise ValueError(
            "an index must be https or a file: it pins each model's sha256, so over plain "
            "http anything on the way could swap a model and its hash together"
        )
    return source


def _loopback(host: str | None) -> bool:
    """This machine's own address: plain http to it never crosses a network."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


def _index_base(source: str) -> str:
    """What a relative file ``url`` in this index resolves against: the index's location as
    named, not where a redirect lands (a hub answers with a cache or CDN address, where
    the index's siblings are not), a bare path taken as the file it names."""
    if urllib.parse.urlsplit(source).scheme:
        return source
    return Path(source).expanduser().absolute().as_uri()


def _read_source(source: str, timeout: float) -> dict[str, Any]:
    split = urllib.parse.urlsplit(source)
    if split.scheme in ("http", "https"):
        with urllib.request.urlopen(source, timeout=timeout) as resp:  # noqa: S310 - user-given index URL
            # urllib follows a redirect from https down to http: the index must still arrive
            # authenticated.
            try:
                check_index_source(resp.url)
            except ValueError as exc:
                raise ValueError(f"it redirects to {resp.url}, and {exc}") from None
            raw = resp.read()
    elif split.scheme == "file":
        raw = Path(urllib.request.url2pathname(split.path)).read_bytes()
    else:
        raw = Path(source).expanduser().read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("index root must be a JSON object")
    return data


def _validate_entry(source: str, key: str, entry: Any) -> None:
    """Index shape errors must reach the user as an ``error:`` line, not as a
    traceback from whichever consumer trips over them first."""
    where = f"weights index '{source}': '{key}'"
    if not isinstance(entry, dict):
        raise WeightsError(f"{where} must be a JSON object")
    if not isinstance(entry.get("langs") or [], list):
        raise WeightsError(f"{where}.langs must be a list (a string silently breaks --lang)")
    files = entry.get("files") or {}
    if not isinstance(files, dict):
        raise WeightsError(f"{where}.files must be an object of name -> {{url, sha256}}")
    for name, spec in files.items():
        if not isinstance(spec or {}, dict):
            raise WeightsError(f"{where}.files['{name}'] must be an object with a url")
        size = (spec or {}).get("size")
        if size is not None and not isinstance(size, int):
            raise WeightsError(f"{where}.files['{name}'].size must be a whole number of bytes")
        url = (spec or {}).get("url")
        if url is not None and not _parses(url):
            raise WeightsError(f"{where}.files['{name}'].url must be a URL, or a path relative to the index")


def _parses(url: Any) -> bool:
    """Whether a file ``url`` is one the load can resolve against its index."""
    if not isinstance(url, str):
        return False
    try:
        urllib.parse.urlsplit(url)
    except ValueError:  # an unclosed IPv6 bracket, a host NFKC would rewrite
        return False
    return True


def load_index(sources: Sequence[str], *, timeout: float = 30.0) -> dict[str, dict[str, Any]]:
    """Merge the sources (path, ``file://`` or ``https://``) in order, later winning per
    key, each entry's relative file urls resolved against the source that listed it. No
    sources is no index."""
    models: dict[str, dict[str, Any]] = {}
    for source in sources:
        try:
            check_index_source(source)
        except ValueError as exc:
            raise WeightsError(f"weights index '{source}': {exc}") from None
        try:
            data = _read_source(source, timeout)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise WeightsError(f"cannot read weights index '{source}': {exc}") from None
        entries = data.get("models")
        if entries is None:
            entries = {}
        if not isinstance(entries, dict):
            raise WeightsError(
                f"weights index '{source}': 'models' must be an object of key -> entry"
            )
        base = _index_base(source)
        for key, entry in entries.items():
            validate_key(key)
            _validate_entry(source, key, entry)
            for spec in (entry.get("files") or {}).values():
                if spec and spec.get("url"):
                    spec["url"] = urllib.parse.urljoin(base, spec["url"])
        models.update(entries)
    return with_backbones(models)


def refresh_index(
    sources: Sequence[str], root: Path | None = None, *, timeout: float = 30.0,
) -> dict[str, dict[str, Any]]:
    """:func:`load_index`, then cache the result in the store for offline readers."""
    models = load_index(sources, timeout=timeout)
    base = root or store_root()
    try:
        base.mkdir(parents=True, exist_ok=True)
        _write_json(base / INDEX_CACHE, {
            "fetched_unix": int(time.time()),
            "sources": list(sources),
            "models": models,
        })
    except OSError as exc:
        raise WeightsError(f"cannot cache the weights index under {base}: {exc}") from None
    return models


def _write_json(path: Path, payload: Any) -> None:
    """Replace atomically: a reader (or a crash) never meets a half-written file. The temp
    name carries the pid, since the CLI and the gateway write one store."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def cached_index(
    root: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], int, list[str]] | None:
    """The last cached index as (models, fetched_unix, sources), or None when nothing
    is cached."""
    try:
        data = json.loads(((root or store_root()) / INDEX_CACHE).read_text("utf-8"))
        models = data["models"]
        if not isinstance(models, dict) or not all(isinstance(e, dict) for e in models.values()):
            return None
        sources = data.get("sources") or []
        return with_backbones(models), int(data.get("fetched_unix") or 0), [str(s) for s in sources]
    except (OSError, ValueError, KeyError, TypeError):
        return None


# openWakeWord: every phrase head ships the same feature models (mel + embedding, one
# pair per platform). A head of your own needs just that pair, so the index gains one
# ``backbone`` entry per platform, derived from any head's files minus the head itself —
# unless the index carries its own.
WAKE_PREFIX = "wake/openwakeword/"
BACKBONE_STEM = "backbone"
_HEAD_FILES = frozenset({"model.onnx", "meta.json"})


def with_backbones(models: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``models`` plus a derived ``wake/openwakeword/backbone/<platform>`` per platform
    that has a head and no backbone entry of its own."""
    out = dict(models)
    for key in sorted(models):
        rest = key[len(WAKE_PREFIX):] if key.startswith(WAKE_PREFIX) else ""
        stem, platform = rest.rsplit("/", 1) if "/" in rest else ("", "")
        backbone = f"{WAKE_PREFIX}{BACKBONE_STEM}/{platform}"
        if not stem or stem == BACKBONE_STEM or backbone in out:
            continue
        files = {n: f for n, f in (models[key].get("files") or {}).items() if n not in _HEAD_FILES}
        if files:
            out[backbone] = {"files": files, "derived_from": key}
    return out


def entry_size(entry: dict[str, Any]) -> int:
    """Bytes the index declares for an entry (0 when it declares none, or declares one a
    third-party index wrote as something other than a number)."""
    sizes = ((f or {}).get("size") for f in (entry.get("files") or {}).values())
    return sum(s for s in sizes if isinstance(s, int))


def key_platform(key: str) -> str:
    return key.rsplit("/", 1)[-1]


def host_platforms(device: str | None) -> tuple[str, ...]:
    """Platform suffixes a section can run: ``onnx`` always (the CPU path), plus the
    RKNN build for the SoC its ``device`` names (spelled as keys spell it)."""
    device = (device or "").strip().lower()
    return ("onnx", f"rknn.{device}") if device else ("onnx",)


# ---- fetch / prune ----------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_files(d: Path) -> dict[str, Any] | None:
    """The manifest's recorded files, None when there is no readable manifest: a truncated
    one (a power cut between write and flush) must not read as a fetched model."""
    try:
        payload = json.loads((d / MANIFEST).read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), dict):
        return None
    return payload["files"]


def fetch(
    key: str,
    entry: dict[str, Any],
    *,
    force: bool = False,
    root: Path | None = None,
    log: Callable[[str], None] = lambda _line: None,
    managed_by: str | None = None,
    progress: Callable[[str, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Verify-and-install one index entry into the store; idempotent. Files already
    present with the index's sha256 (per the manifest) are kept; ``force`` refetches
    everything. Downloads stream to ``.partial-*``, verify, then atomically replace: a
    partial never lands on the final name. ``managed_by`` is recorded in the manifest
    (what installed the key, so an automatic cleanup removes only its own); ``progress``
    gets (file name, bytes so far) per chunk and ``should_stop`` is polled between
    chunks: True abandons the download as a :class:`WeightsError`."""
    d = store_dir(key, root)
    # nested keys would let the stale-file sweep rmtree the inner installation
    for other in installed(root):
        if other != key and (other.startswith(key + "/") or key.startswith(other + "/")):
            raise WeightsError(
                f"key '{key}' would nest with installed '{other}'; prune one first"
            )
    files: dict[str, Any] = entry.get("files") or {}
    if not files:
        raise WeightsError(f"index entry '{key}' lists no files")
    for name in files:
        if not name or "/" in name or "\\" in name or name.startswith("."):
            raise WeightsError(f"index entry '{key}' has an unsafe file name '{name}'")
    if dangling(key, root):  # mkdir raises FileExistsError on a link whose target is gone
        raise WeightsError(f"'{key}' is a relocated leaf whose target is gone; prune it first")
    d.mkdir(parents=True, exist_ok=True)
    have = ({} if force else _manifest_files(d)) or {}
    recorded: dict[str, Any] = {}
    for name, spec in files.items():
        url = str((spec or {}).get("url") or "")
        want = (spec or {}).get("sha256")
        dest = d / name
        prior = have.get(name) or {}
        if dest.exists() and want and prior.get("sha256") == want:
            recorded[name] = prior
            log(f"  {name}: already fetched")
            continue
        scheme = urllib.parse.urlsplit(url).scheme
        if scheme == "file":
            src = Path(urllib.request.url2pathname(urllib.parse.urlsplit(url).path)).resolve()
            if not src.is_file():
                raise WeightsError(f"'{key}' {name}: source file not found: {src}")
            digest = _sha256_file(src)
            if want and digest != want:
                raise WeightsError(
                    f"'{key}' {name}: sha256 mismatch (index {want[:12]}..., file {digest[:12]}...)"
                )
            dest.unlink(missing_ok=True)
            dest.symlink_to(src)  # link, not copy: the source stays the one copy on disk
            recorded[name] = {"sha256": digest, "linked": str(src)}
            log(f"  {name}: linked -> {src}")
        elif scheme in ("http", "https"):
            if not want:
                raise WeightsError(
                    f"'{key}' {name}: remote files must pin a sha256 in the index"
                )
            part = d / f".partial-{os.getpid()}-{name}"
            digester = hashlib.sha256()
            total = 0
            try:
                try:
                    with urllib.request.urlopen(url, timeout=60) as resp, part.open("wb") as out:  # noqa: S310
                        while chunk := resp.read(_CHUNK):
                            if should_stop is not None and should_stop():
                                raise WeightsError(f"'{key}' {name}: download cancelled")
                            digester.update(chunk)
                            out.write(chunk)
                            total += len(chunk)
                            if progress is not None:
                                progress(name, total)
                # A truncated chunked body raises IncompleteRead: HTTPException, NOT OSError.
                except (OSError, http.client.HTTPException) as exc:
                    raise WeightsError(f"'{key}' {name}: download failed: {exc}") from None
                digest = digester.hexdigest()
                if digest != want:
                    raise WeightsError(
                        f"'{key}' {name}: sha256 mismatch after download "
                        f"(index {want[:12]}..., got {digest[:12]}...); refusing to install"
                    )
                os.replace(part, dest)
            finally:  # covers Ctrl-C too; a no-op once the replace has happened
                part.unlink(missing_ok=True)
            recorded[name] = {"sha256": digest}
            log(f"  {name}: fetched {total / 1e6:.1f} MB")
        else:
            raise WeightsError(
                f"'{key}' {name}: unsupported url '{url or '<missing>'}' (need http(s):// or file://)"
            )
    payload: dict[str, Any] = {"key": key, "fetched_unix": int(time.time()), "files": recorded}
    if managed_by:
        payload["managed_by"] = managed_by
    _write_json(d / MANIFEST, payload)
    # A file dropped by a later revision of the entry would poison <stem>.* resolution
    # forever. Sweep AFTER the manifest write: a fetch that raised deletes nothing.
    for p in d.iterdir():
        # Another fetcher's partial (or its manifest temp) is its business, not stale.
        if p.name == MANIFEST or p.name in recorded or p.name.startswith((".partial-", MANIFEST + ".")):
            continue
        try:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            else:
                p.unlink()
        except OSError:
            log(f"  {p.name}: stale, could not remove")
    return d


def installed(root: Path | None = None) -> dict[str, Path]:
    """Fetched keys -> store dirs (anything holding a manifest, index or not). Keys are
    hierarchical, so manifests are discovered at arbitrary depth; validation keeps stray
    directories from being exposed as installed keys."""
    base = root or store_root()
    if not base.is_dir():
        return {}
    found: dict[str, Path] = {}
    seen: set[tuple[int, int]] = set()
    # os.walk, not rglob: ** skips symlinked dirs, and users relocate subtrees that way.
    # Following them needs the loop guard a relocation back into the store would else hit.
    for dirpath, dirnames, filenames in os.walk(base, followlinks=True):
        try:
            node = os.stat(dirpath)
        except OSError:
            dirnames[:] = []
            continue
        if (node.st_dev, node.st_ino) in seen:
            dirnames[:] = []
            continue
        seen.add((node.st_dev, node.st_ino))
        if MANIFEST not in filenames:
            continue
        rel = Path(dirpath).relative_to(base).as_posix()
        try:
            validate_key(rel)
        except WeightsError:
            continue
        if _manifest_files(Path(dirpath)) is None:
            continue  # torn write: the plan refetches it instead of calling it in place
        found[rel] = Path(dirpath)
    return dict(sorted(found.items()))


def managed_by(key: str, root: Path | None = None) -> str | None:
    """What installed a fetched key (its manifest's ``managed_by``), else None."""
    try:
        data = json.loads((store_dir(key, root) / MANIFEST).read_text("utf-8"))
    except (OSError, ValueError, WeightsError):
        return None
    value = data.get("managed_by") if isinstance(data, dict) else None
    return value if isinstance(value, str) and value else None


def disk_usage(d: Path) -> int:
    """Bytes under ``d``; symlinks count as the link itself, never the target."""
    total = 0
    for p in d.rglob("*"):
        try:
            total += p.lstat().st_size
        except OSError:
            pass
    return total


def relocation_target(key: str, root: Path | None = None) -> Path | None:
    """Where a relocated (symlinked) leaf points, else None."""
    d = store_dir(key, root or store_root())
    return Path(os.readlink(d)) if d.is_symlink() else None


def dangling(key: str, root: Path | None = None) -> bool:
    """A relocated leaf whose target is gone: absent from ``installed()``, prunable.
    False for anything but a full key (a prefix resolves against ``installed()``)."""
    try:
        d = store_dir(key, root or store_root())
    except WeightsError:
        return False
    return d.is_symlink() and not d.exists()


def prune(key: str, root: Path | None = None) -> int:
    """Remove one fetched key from the store; returns the bytes freed (a relocated leaf
    frees only its link)."""
    base = root or store_root()
    d = store_dir(key, base)
    if d.is_symlink() and not d.exists():
        d.unlink()  # a relocation whose target is gone (store unplugged, deleted)
        return 0
    # manifest required: a bare ancestor dir would prune other keys' children
    if not (d / MANIFEST).is_file():
        raise WeightsError(f"'{key}' is not in the store ({base})")
    if d.is_symlink():
        # A relocated leaf (installed() follows links): drop the link, never its target.
        freed = d.lstat().st_size
        d.unlink()
    else:
        freed = disk_usage(d)
        shutil.rmtree(d)
    # Remove every now-empty ancestor, never the store root, stopping at a sibling or at
    # a symlink: that is the user's relocation of a subtree, not store scaffolding.
    parent = d.parent
    while (
        parent != base
        and not parent.is_symlink()
        and parent.is_dir()
        and not any(parent.iterdir())
    ):
        parent.rmdir()
        parent = parent.parent
    return freed


# ---- runtime resolution -----------------------------------------------------


def backbone_key_for(platform: str) -> str:
    """The backbone package built for one platform suffix."""
    return f"{WAKE_PREFIX}{BACKBONE_STEM}/{platform}"


def backbone_key(device: str | None, root: Path | None = None) -> str:
    """The openWakeWord backbone package this host RUNS: the device build when the store
    has it, else the CPU one (the head itself stays ONNX either way). What the store
    should hold is the plan's question (:func:`sync.backbone_wanted`)."""
    keys = [backbone_key_for(p) for p in host_platforms(device)]
    have = installed(root)
    return next((k for k in reversed(keys) if k in have), keys[0])


def fill_engine_paths(block: Any) -> Any:
    """Copy of an engine block with unset ``*_path`` fields resolved from the store dir
    named by ``block.weights``; explicit paths always win."""
    key = getattr(block, "weights", None)
    if not key:
        return block
    d = store_dir(key)
    known = _manifest_files(d)
    if known is None:
        raise NotFetchedError(key)
    updates: dict[str, str] = {}
    for name in type(block).model_fields:
        if not name.endswith("_path") or getattr(block, name) is not None:
            continue
        # Manifest-recorded names only: a hand-dropped file must not shadow the entry.
        matches = sorted(p for p in d.glob(name[:-5] + ".*") if p.name in known)
        # ONNX external data (encoder.onnx.data / encoder.onnx_data) is a companion,
        # not a variant.
        matches = [
            m for m in matches
            if not any(
                o is not m
                and m.name.startswith(o.name)
                and m.name[len(o.name):len(o.name) + 1] in (".", "_")
                for o in matches
            )
        ]
        if len(matches) > 1:
            raise WeightsError(
                f"weights '{key}': ambiguous {name[:-5]}.* "
                f"({', '.join(m.name for m in matches)})"
            )
        if matches:
            updates[name] = str(matches[0])
    return block.model_copy(update=updates) if updates else block


def apply_weights(cfg: Any, block_name: str) -> Any:
    """``cfg`` with the named engine block store-resolved (a bilingual ``secondary``
    sub-block resolves too); a no-op when nothing names a ``weights`` key. An openWakeWord
    block with a head of its own and no key resolves its feature models from the store's
    backbone package for its platform. Local filesystem only."""
    block = getattr(cfg, block_name, None)
    if block is None:
        return cfg
    if block_name == "openwakeword" and not block.weights and block.model_path and not block.embedding_path:
        block = block.model_copy(update={"weights": backbone_key(block.resolved_target)})
    filled = fill_engine_paths(block) if getattr(block, "weights", None) else block
    second = getattr(filled, "secondary", None)
    if second is not None and getattr(second, "weights", None):
        filled = filled.model_copy(update={"secondary": fill_engine_paths(second)})
    return cfg if filled is block else cfg.model_copy(update={block_name: filled})
