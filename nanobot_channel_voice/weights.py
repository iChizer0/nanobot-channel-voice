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
entries and NO weights: they come from vendor/user index files (``--index`` /
``$NANOBOT_VOICE_INDEX``, else :data:`DEFAULT_INDEX_SOURCES`). ``http(s)://`` sources
stream into the store and MUST pin a sha256; ``file://`` sources are symlinked in place,
verified when the index pins one.

File names inside an entry are the resolution contract: an engine block setting
``weights: <key>`` gets its unset ``*_path`` fields filled by stem + any extension
(``encoder_path`` -> ``encoder.<ext>``); explicit paths always win. The network belongs
to the CLI alone; :func:`apply_weights` touches only the local store.
"""

from __future__ import annotations

import hashlib
import http.client
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

# Consulted when neither --index nor $NANOBOT_VOICE_INDEX names a source. URLs only,
# never a bundled data file (the wheel ships no entries); an explicit source replaces it.
DEFAULT_INDEX_SOURCES: tuple[str, ...] = (
    "https://huggingface.co/iChizer0/nanobot-channel-voice-models/resolve/main/weights-index.json",
)

_KEY_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")  # no dot-prefix: kills "..", hidden dirs
_CHUNK = 1 << 20


class WeightsError(RuntimeError):
    """Actionable store/index failure; the message is user-facing."""


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


def _read_source(source: str) -> dict[str, Any]:
    split = urllib.parse.urlsplit(source)
    if split.scheme in ("http", "https"):
        with urllib.request.urlopen(source, timeout=30) as resp:  # noqa: S310 - user-given index URL
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


def load_index(sources: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """Merge the sources (path, ``file://`` or ``http(s)://``) in order, later winning
    per key; empty means :data:`DEFAULT_INDEX_SOURCES`."""
    models: dict[str, dict[str, Any]] = {}
    for source in sources or DEFAULT_INDEX_SOURCES:
        try:
            data = _read_source(source)
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise WeightsError(f"cannot read weights index '{source}': {exc}") from None
        entries = data.get("models")
        if entries is None:
            entries = {}
        if not isinstance(entries, dict):
            raise WeightsError(
                f"weights index '{source}': 'models' must be an object of key -> entry"
            )
        for key, entry in entries.items():
            validate_key(key)
            _validate_entry(source, key, entry)
        models.update(entries)
    return models


# ---- fetch / prune ----------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_files(d: Path) -> dict[str, Any]:
    try:
        return json.loads((d / MANIFEST).read_text("utf-8")).get("files") or {}
    except (OSError, ValueError):
        return {}


def fetch(
    key: str,
    entry: dict[str, Any],
    *,
    force: bool = False,
    root: Path | None = None,
    log: Callable[[str], None] = lambda _line: None,
) -> Path:
    """Verify-and-install one index entry into the store; idempotent. Files already
    present with the index's sha256 (per the manifest) are kept; ``force`` refetches
    everything. Downloads stream to ``.partial-*``, verify, then atomically replace: a
    partial never lands on the final name."""
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
    d.mkdir(parents=True, exist_ok=True)
    have = {} if force else _manifest_files(d)
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
            part = d / f".partial-{name}"
            digester = hashlib.sha256()
            total = 0
            try:
                try:
                    with urllib.request.urlopen(url, timeout=60) as resp, part.open("wb") as out:  # noqa: S310
                        while chunk := resp.read(_CHUNK):
                            digester.update(chunk)
                            out.write(chunk)
                            total += len(chunk)
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
    payload = {"key": key, "fetched_unix": int(time.time()), "files": recorded}
    (d / MANIFEST).write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
    # A file dropped by a later revision of the entry would poison <stem>.* resolution
    # forever. Sweep AFTER the manifest write: a fetch that raised deletes nothing.
    for p in d.iterdir():
        if p.name == MANIFEST or p.name in recorded:
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
    # os.walk, not rglob: ** skips symlinked dirs, and users relocate subtrees that way
    for dirpath, _dirnames, filenames in os.walk(base, followlinks=True):
        if MANIFEST not in filenames:
            continue
        rel = Path(dirpath).relative_to(base).as_posix()
        try:
            validate_key(rel)
        except WeightsError:
            continue
        found[rel] = Path(dirpath)
    return dict(sorted(found.items()))


def disk_usage(d: Path) -> int:
    """Bytes under ``d``; symlinks count as the link itself, never the target."""
    total = 0
    for p in d.rglob("*"):
        try:
            total += p.lstat().st_size
        except OSError:
            pass
    return total


def prune(key: str, root: Path | None = None) -> int:
    """Remove one fetched key from the store; returns the bytes freed."""
    base = root or store_root()
    d = store_dir(key, base)
    # manifest required: a bare ancestor dir would prune other keys' children
    if not (d / MANIFEST).is_file():
        raise WeightsError(f"'{key}' is not in the store ({base})")
    freed = disk_usage(d)
    shutil.rmtree(d)
    # Remove every now-empty ancestor, never the store root, stopping at a sibling.
    parent = d.parent
    while parent != base and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()
        parent = parent.parent
    return freed


# ---- runtime resolution -----------------------------------------------------


def fill_engine_paths(block: Any) -> Any:
    """Copy of an engine block with unset ``*_path`` fields resolved from the store dir
    named by ``block.weights``; explicit paths always win."""
    key = getattr(block, "weights", None)
    if not key:
        return block
    d = store_dir(key)
    if not (d / MANIFEST).is_file():
        raise WeightsError(
            f"weights '{key}' are not fetched (store: {d}); run: nanobot-voice fetch {key}"
        )
    known = _manifest_files(d)
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
    sub-block resolves too); a no-op when nothing names a ``weights`` key. Local
    filesystem only."""
    block = getattr(cfg, block_name, None)
    if block is None:
        return cfg
    filled = fill_engine_paths(block) if getattr(block, "weights", None) else block
    second = getattr(filled, "secondary", None)
    if second is not None and getattr(second, "weights", None):
        filled = filled.model_copy(update={"secondary": fill_engine_paths(second)})
    return cfg if filled is block else cfg.model_copy(update={block_name: filled})
