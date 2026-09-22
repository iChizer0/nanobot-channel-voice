"""``nanobot-voice``: manage on-device model weights, export the effective config.

Thin argparse front-end over :mod:`.weights`. ``config`` is the export counterpart of
the WebUI "Import Json" box: the effective ``channels.voice`` section as canonical JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic.alias_generators import to_snake

from nanobot_channel_voice import weights as w


def _resolve_key(token: str, candidates: dict[str, Any], what: str) -> str:
    """Exact key, or a unique SEGMENT prefix of one. Segment-aware, or ``tts/mms/en``
    would string-match a sibling family like ``tts/mms/eng/...``."""
    if token in candidates:
        return token
    prefix = token.rstrip("/") + "/"
    hits = sorted(k for k in candidates if k.startswith(prefix))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise w.WeightsError(f"unknown weights key '{token}': {what}")
    raise w.WeightsError(f"'{token}' is ambiguous: {', '.join(hits)}")


def _confirm(prompt: str, yes: bool, why: str) -> None:
    if yes:
        return
    if not sys.stdin.isatty():
        raise w.WeightsError(f"{why}; pass --yes to confirm")
    if input(f"{prompt} [y/N] ").strip().lower() not in ("y", "yes"):
        raise w.WeightsError("aborted")


def _fetch_one(key: str, entry: dict[str, Any], *, force: bool, yes: bool, root: Path) -> None:
    line = key
    if entry.get("license"):
        line += f"  [{entry['license']}]"
    if entry.get("source"):
        line += f"  ({entry['source']})"
    print(line)
    if entry.get("accept"):  # e.g. a non-commercial license notice
        print(f"  NOTICE: {entry['accept']}")
        _confirm("  Accept and download?", yes, f"'{key}' requires accepting its notice")
    d = w.fetch(key, entry, force=force, root=root, log=print)
    print(f"  -> {d}")


def _fetch(args: argparse.Namespace, index: dict[str, Any], root: Path) -> int:
    for token in args.keys:
        key = _resolve_key(token, index, "try: nanobot-voice list")
        _fetch_one(key, index[key], force=args.force, yes=args.yes, root=root)
    return 0


def _sync(args: argparse.Namespace, index: dict[str, Any], root: Path) -> int:
    from nanobot_channel_voice.sync import plan_sync, voice_section

    path = Path(args.config).expanduser() if args.config else Path.home() / ".nanobot" / "config.json"
    plan = plan_sync(voice_section(path), index, root, managed_by=None, prune=args.prune)
    for key in plan.wanted:
        w.validate_key(key)
    if plan.unknown:
        raise w.WeightsError(
            f"configured weights not in the index: {', '.join(plan.unknown)} "
            "(pass the right --index / $NANOBOT_VOICE_INDEX)"
        )
    # The CLI fetches through its own loop: notices prompt here, and every named key is
    # re-verified against its manifest (fetch is idempotent; --force refetches). A key the
    # store holds but this index does not name stays as it is: it was fetched from another.
    for key in plan.wanted:
        entry = index.get(key)
        if entry is None:
            print(f"{key}: installed, not in this index (not re-verified)")
            continue
        _fetch_one(key, entry, force=args.force, yes=args.yes, root=root)
    if not plan.wanted:
        print(f"{path}: channels.voice configures no weights keys")
    if args.prune:
        if not plan.wanted:  # an empty config must not silently empty the store
            raise w.WeightsError(
                "config names no weights; refusing to prune everything "
                "(use: nanobot-voice prune --all)"
            )
        freed = 0
        for key in plan.prune:
            freed += w.prune(key, root)
            print(f"pruned {key} (not in config)")
        print(f"freed {_fmt_mb(freed)}")
    return 0


def _config(args: argparse.Namespace) -> int:
    """Print the EFFECTIVE ``channels.voice`` section: the file's section run through the
    plugin schema (spelling twins folded, a pending importJson merge applied), canonical
    camelCase. Doubles as a linter: an invalid section prints the schema error."""
    from pydantic import ValidationError

    from nanobot_channel_voice.config import VoiceConfig

    path = Path(args.config).expanduser() if args.config else Path.home() / ".nanobot" / "config.json"
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise w.WeightsError(f"cannot read nanobot config {path}: {exc}") from None
    section = (data.get("channels") or {}).get("voice") or {}
    try:
        cfg = VoiceConfig.model_validate(section)
    except ValidationError as exc:
        raise w.WeightsError(f"channels.voice does not validate: {exc}") from None
    dumped = cfg.model_dump(
        mode="json",
        by_alias=True,
        exclude_unset=not args.full,  # default: only what is configured (round-trips small)
        # Not the transport, nor `enabled`: the WebUI toggle owns it and the paste box drops it.
        exclude={"import_json", "enabled"},
    )
    if not args.secrets:
        omitted = _scrub_secrets(dumped)
        if omitted:
            print(
                f"note: {omitted} secret field(s) omitted; --secrets includes them",
                file=sys.stderr,
            )
    _fold_home(dumped)
    print(json.dumps(dumped, indent=2, ensure_ascii=False))
    return 0


def _is_path_field(name: str) -> bool:
    """The schema's rule (config._VoiceBase): a ``*Path``/``*Dir`` leaf."""
    return to_snake(name).rsplit("_", 1)[-1] in ("path", "dir")


def _fold_home(node: Any) -> None:
    """Paths back to ``~`` in place: the schema expands them, and an export that names
    the home directory neither pastes across machines nor keeps the user name out."""
    home = os.path.expanduser("~")
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and _is_path_field(key):
                if value == home or value.startswith(home + os.sep):
                    node[key] = "~" + value[len(home):]
            else:
                _fold_home(value)
    elif isinstance(node, list):
        for value in node:
            _fold_home(value)


def _scrub_secrets(node: Any) -> int:
    """Drop every ``apiKey`` in place, returning how many held a value. Dropping, not
    masking: a mask pasted back would overwrite the real key. test_cli_config pins that
    every credential-shaped schema field has the ``api_key`` leaf."""
    omitted = 0
    if isinstance(node, dict):
        if node.pop("apiKey", None):
            omitted += 1
        for value in node.values():
            omitted += _scrub_secrets(value)
    return omitted


def _fmt_mb(n: int) -> str:
    return f"{n / 1e6:,.1f} MB"


def _list(args: argparse.Namespace, index: dict[str, Any], root: Path) -> int:
    have = w.installed(root)
    keys = sorted(set(index) | set(have))
    shown = 0
    for key in keys:
        entry = index.get(key)
        langs = (entry or {}).get("langs") or []
        if args.lang and args.lang not in langs:
            continue
        if key in have:
            status = f"installed {_fmt_mb(w.disk_usage(have[key]))}"
        else:
            size = sum(int((f or {}).get("size") or 0) for f in (entry or {}).get("files", {}).values())
            status = f"available {_fmt_mb(size)}" if size else "available"
        notes = [s for s in ((entry or {}).get("license"), " ".join(langs) or None) if s]
        if entry is None:
            notes.append("not in index")
        print(f"{key:<44} [{status}]" + (f"  {' | '.join(notes)}" if notes else ""))
        shown += 1
    if not shown:
        where = f" for --lang {args.lang}" if args.lang else ""
        print(f"no weights{where}; add an index with --index or $NANOBOT_VOICE_INDEX")
    return 0


def _prune(args: argparse.Namespace, root: Path) -> int:
    have = w.installed(root)
    if args.all == bool(args.keys):
        raise w.WeightsError("prune takes either keys or --all")
    if args.all:
        if not have:
            print(f"store is empty ({root})")
            return 0
        _confirm(
            f"Remove ALL {len(have)} fetched weights under {root}?",
            args.yes,
            f"--all removes {len(have)} fetched weights",
        )
        keys = sorted(have)
    else:
        keys = [
            t if w.dangling(t, root) else _resolve_key(t, have, "nothing fetched under that name")
            for t in args.keys
        ]
    freed = 0
    for key in keys:
        target = w.relocation_target(key, root)
        freed += w.prune(key, root)
        if target is None:
            print(f"removed {key}")
        elif target.exists():  # only the link went: say where the data still is
            print(f"unlinked {key} (relocated; its files stay at {target})")
        else:
            print(f"removed the dangling link {key} (its target {target} is gone)")
    print(f"freed {_fmt_mb(freed)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nanobot-voice",
        description="Fetch, list, sync, and prune on-device model weights for nanobot-channel-voice.",
    )
    parser.add_argument(
        "--index",
        action="append",
        metavar="PATH_OR_URL",
        help="weights index (JSON path, file:// or http(s):// URL); repeatable, later wins "
        "per key; default: $NANOBOT_VOICE_INDEX, else a built-in community index URL",
    )
    parser.add_argument(
        "--models-dir",
        metavar="DIR",
        help="store directory (default: $NANOBOT_VOICE_MODELS_DIR or "
        "~/.local/share/nanobot-voice/models; the channel resolves the same way, so "
        "non-default locations need the env var set for the runtime too)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download+verify weights into the store")
    p.add_argument(
        "keys", nargs="+", metavar="KEY",
        help="index key or unique prefix (<kind>/<model-path>/<platform>)",
    )
    p.add_argument("--force", action="store_true", help="refetch even if already installed")
    p.add_argument("-y", "--yes", action="store_true", help="accept license notices non-interactively")

    p = sub.add_parser("list", help="show index entries and installed weights")
    p.add_argument("--lang", metavar="XX", help="only entries listing this language code")

    p = sub.add_parser(
        "sync",
        help="fetch every weights key configured in nanobot's config "
        "(channels.voice.**.weights); --prune removes fetched weights the config no longer names",
    )
    p.add_argument("--config", metavar="FILE", help="nanobot config (default: ~/.nanobot/config.json)")
    p.add_argument("--prune", action="store_true", help="also remove installed keys not in the config")
    p.add_argument("--force", action="store_true", help="refetch even if already installed")
    p.add_argument("-y", "--yes", action="store_true", help="accept license notices non-interactively")

    p = sub.add_parser("prune", help="remove fetched weights from the store")
    p.add_argument(
        "keys", nargs="*", metavar="KEY",
        help="fetched key or unique prefix (<kind>/<model-path>/<platform>)",
    )
    p.add_argument("--all", action="store_true", help="remove everything in the store")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask for confirmation")

    p = sub.add_parser(
        "config",
        help="print the effective channels.voice section as canonical JSON, "
        "paste-ready for the WebUI Import Json box (secrets omitted unless --secrets)",
    )
    p.add_argument("--config", metavar="FILE", help="nanobot config (default: ~/.nanobot/config.json)")
    p.add_argument("--full", action="store_true", help="include every default, not just configured keys")
    p.add_argument("--secrets", action="store_true", help="keep apiKey values (full local backup)")

    args = parser.parse_args(argv)
    root = Path(args.models_dir).expanduser() if args.models_dir else w.store_root()
    env_index = os.environ.get("NANOBOT_VOICE_INDEX")
    sources = args.index if args.index else ([env_index] if env_index else [])
    try:
        if args.cmd == "config":
            return _config(args)  # needs no weights index
        if args.cmd == "prune":
            return _prune(args, root)
        try:
            index = w.refresh_index(sources, root)  # also caches it for the WebUI form
        except w.WeightsError:
            # Offline: `list` degrades to the installed keys. A source the user NAMED
            # stays a hard error, and fetch/sync need an index.
            if sources or args.cmd != "list":
                raise
            print("warning: no reachable weights index; listing the local store only",
                  file=sys.stderr)
            index = {}
        if args.cmd == "fetch":
            return _fetch(args, index, root)
        if args.cmd == "sync":
            return _sync(args, index, root)
        return _list(args, index, root)
    except (w.WeightsError, OSError) as exc:  # OSError: a store path that won't go
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
