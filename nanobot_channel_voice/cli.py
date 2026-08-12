"""``nanobot-voice``: manage on-device model weights, export the effective config.

Thin argparse front-end over :mod:`.weights`. ``config`` is the export
counterpart of the WebUI "Import Json" box: the effective ``channels.voice``
section as paste-ready canonical JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from nanobot_channel_voice import weights as w


def _resolve_key(token: str, candidates: dict[str, Any], what: str) -> str:
    """Exact key, or a unique prefix of one (``stt/whisper-base`` picks the sole
    platform variant; several -> error listing them)."""
    if token in candidates:
        return token
    hits = sorted(k for k in candidates if k.startswith(token))
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


def _config_weights_keys(node: Any) -> set[str]:
    """Every ``"weights": "<key>"`` string anywhere under a config node: the
    scan is structural, so new engine blocks need no CLI change."""
    found: set[str] = set()
    if isinstance(node, dict):
        for name, value in node.items():
            if name == "weights" and isinstance(value, str) and value:
                found.add(value)
            else:
                found |= _config_weights_keys(value)
    elif isinstance(node, list):
        for value in node:
            found |= _config_weights_keys(value)
    return found


def _sync(args: argparse.Namespace, index: dict[str, Any], root: Path) -> int:
    path = Path(args.config).expanduser() if args.config else Path.home() / ".nanobot" / "config.json"
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise w.WeightsError(f"cannot read nanobot config {path}: {exc}") from None
    section = (data.get("channels") or {}).get("voice") or {}
    wanted = sorted(_config_weights_keys(section))
    for key in wanted:
        w.validate_key(key)
    unknown = [k for k in wanted if k not in index]
    if unknown:
        raise w.WeightsError(
            f"configured weights not in the index: {', '.join(unknown)} "
            "(pass the right --index / $NANOBOT_VOICE_INDEX)"
        )
    for key in wanted:
        _fetch_one(key, index[key], force=args.force, yes=args.yes, root=root)
    if not wanted:
        print(f"{path}: channels.voice configures no weights keys")
    if args.prune:
        if not wanted:  # an empty config must not silently empty the store
            raise w.WeightsError(
                "config names no weights; refusing to prune everything "
                "(use: nanobot-voice prune --all)"
            )
        freed = 0
        for key in sorted(set(w.installed(root)) - set(wanted)):
            freed += w.prune(key, root)
            print(f"pruned {key} (not in config)")
        print(f"freed {_fmt_mb(freed)}")
    return 0


def _config(args: argparse.Namespace) -> int:
    """Print the EFFECTIVE ``channels.voice`` section: the file's section run through
    the plugin schema (spelling twins folded, a pending importJson merge applied),
    canonical camelCase keys — paste-ready for the WebUI "Import Json" box on another
    install. Doubles as a linter: an invalid section prints the schema error instead."""
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
        exclude={"import_json"},      # the transport itself never exports
    )
    if not args.secrets:
        omitted = _scrub_secrets(dumped)
        if omitted:
            print(
                f"note: {omitted} secret field(s) omitted; --secrets includes them",
                file=sys.stderr,
            )
    print(json.dumps(dumped, indent=2, ensure_ascii=False))
    return 0


def _scrub_secrets(node: Any) -> int:
    """Drop every ``apiKey`` in place, returning how many held a value. Dropping (not
    masking) keeps the output import-safe: a mask pasted back would overwrite the real
    key. Completeness is pinned by test_cli_config: every credential-shaped schema
    field has the ``api_key`` leaf."""
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
        keys = [_resolve_key(t, have, "nothing fetched under that name") for t in args.keys]
    freed = 0
    for key in keys:
        freed += w.prune(key, root)
        print(f"removed {key}")
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
    p.add_argument("keys", nargs="+", metavar="KEY", help="index key or unique prefix")
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
    p.add_argument("keys", nargs="*", metavar="KEY", help="fetched key or unique prefix")
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
            index = w.load_index(sources)
        except w.WeightsError:
            # The default index is remote, so an offline box must still SEE its own
            # store: `list` degrades to the installed keys. A source the user NAMED
            # stays a hard error, and fetch/sync cannot do their job without an index.
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
    except w.WeightsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
