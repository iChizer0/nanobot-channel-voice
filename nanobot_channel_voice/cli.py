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
    if entry.get("deprecated"):  # still installs: configs may name it
        new = w.renamed_to(entry)
        print(f"  DEPRECATED: renamed to {new}, name that instead" if new else "  DEPRECATED")
    if entry.get("accept"):  # e.g. a non-commercial license notice
        print(f"  NOTICE: {entry['accept']}")
        _confirm("  Accept and download?", yes, f"'{key}' requires accepting its notice")
    d = w.fetch(key, entry, force=force, root=root, log=print)
    print(f"  -> {d}")


def _fetch_all(
    keys: list[str], index: dict[str, Any], *, force: bool, yes: bool, root: Path,
) -> dict[str, str]:
    """Fetch every key, returning the failures by key: one failure never stops the rest."""
    failed: dict[str, str] = {}
    for key in keys:
        try:
            _fetch_one(key, index[key], force=force, yes=yes, root=root)
        except (w.WeightsError, OSError) as exc:
            failed[key] = str(exc)
    return failed


def _report(keys: list[str], failed: dict[str, str], skipped: dict[str, str]) -> None:
    """Per-key outcomes, printed last so a long run ends on what failed."""
    ok = len(keys) - len(failed) - len(skipped)
    print(f"{ok} ok, {len(failed)} failed" + (f", {len(skipped)} skipped" if skipped else ""))
    for key in keys:
        if key in failed:
            why = failed[key].removeprefix(f"'{key}' ")  # fetch's messages lead with the key
            print(f"  failed   {key}: {why}")
        elif key in skipped:
            print(f"  skipped  {key}: {skipped[key]}")
        else:
            print(f"  ok       {key}")


def _fetch(args: argparse.Namespace, index: dict[str, Any], root: Path) -> int:
    # Resolved before any download, so a typo fails fast. Prefixes match current keys
    # only: an alias answers to its exact name.
    current = {k: e for k, e in index.items() if not e.get("deprecated")}
    keys = list(dict.fromkeys(
        t if t in index else _resolve_key(t, current, "try: nanobot-voice list") for t in args.keys
    ))
    failed = _fetch_all(keys, index, force=args.force, yes=args.yes, root=root)
    if len(keys) == 1 and failed:
        raise w.WeightsError(failed[keys[0]])
    if len(keys) > 1:
        _report(keys, failed, {})
    if failed:
        raise w.WeightsError(f"{len(failed)} of {len(keys)} weights failed: {', '.join(failed)}")
    return 0


def _config_path(args: argparse.Namespace) -> Path:
    """``--config``, else the file the gateway reads by default."""
    from nanobot.config.loader import get_config_path

    return Path(args.config).expanduser() if args.config else get_config_path()


def _configured_index(path: Path, *, required: bool) -> list[str]:
    """``channels.voice.index`` as the gateway resolves it from the config at ``path``. With
    no config file at all (a machine nanobot is not set up on) the operator's defaults
    decide alone, unless the command needs the config anyway: then its absence is the
    error, before any download."""
    from nanobot_channel_voice.config import layer_defaults, section_index
    from nanobot_channel_voice.sync import voice_section

    try:
        section = voice_section(path) if required or path.exists() else layer_defaults({})
    except ValueError as exc:  # the defaults, unreadable (voice_section names its own)
        raise w.WeightsError(str(exc)) from None
    try:
        return section_index(section)
    except ValueError as exc:
        raise w.WeightsError(f"{path}: channels.voice.{exc}") from None


def _sync(args: argparse.Namespace, index: dict[str, Any], root: Path) -> int:
    from nanobot_channel_voice.sync import plan_sync, voice_section

    path = _config_path(args)
    plan = plan_sync(voice_section(path), index, root, managed_by=None, prune=args.prune)
    for key in plan.wanted:
        w.validate_key(key)
    if not plan.wanted:
        print(f"{path}: channels.voice configures no weights keys")
        if args.prune:  # an empty config must not silently empty the store
            raise w.WeightsError(
                "config names no weights; refusing to prune everything "
                "(use: nanobot-voice prune --all)"
            )
        return 0
    # Its own loop, not run_sync: notices prompt here and every named key is re-verified
    # (fetch is idempotent). An installed key this index lacks came from another: skipped.
    skipped = {
        k: "installed, not in this index (not re-verified)"
        for k in plan.wanted if k not in index and k not in plan.unknown
    }
    unindexed = "not in the index (set the right channels.voice.index, or pass --index)"
    failed = dict.fromkeys(plan.unknown, unindexed)
    listed = [k for k in plan.wanted if k in index]
    failed |= _fetch_all(listed, index, force=args.force, yes=args.yes, root=root)
    _report(plan.wanted, failed, skipped)
    if failed:
        unpruned = ", nothing pruned" if args.prune and plan.prune else ""
        which = ", ".join(k for k in plan.wanted if k in failed)
        raise w.WeightsError(f"{len(failed)} of {len(plan.wanted)} weights failed: {which}{unpruned}")
    if args.prune:
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

    path = _config_path(args)
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
    keys = sorted({k for k, e in index.items() if not e.get("deprecated")} | set(have))
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
        elif entry.get("deprecated"):
            notes.append(f"renamed to {new}" if (new := w.renamed_to(entry)) else "deprecated")
        print(f"{key:<44} [{status}]" + (f"  {' | '.join(notes)}" if notes else ""))
        shown += 1
    if not shown:
        where = f" for --lang {args.lang}" if args.lang else ""
        print(f"no weights{where}; name an index in channels.voice.index, or pass --index")
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
        help="weights index (JSON path, file:// or https:// URL); repeatable, later wins "
        "per key; default: the config's channels.voice.index, else the built-in community index",
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
    p.add_argument("--config", metavar="FILE", help="nanobot config (default: ~/.nanobot/config.json)")

    p = sub.add_parser("list", help="show index entries and installed weights")
    p.add_argument("--lang", metavar="XX", help="only entries listing this language code")
    p.add_argument("--config", metavar="FILE", help="nanobot config (default: ~/.nanobot/config.json)")

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
    try:
        if args.cmd == "config":
            return _config(args)  # needs no weights index
        if args.cmd == "prune":
            return _prune(args, root)
        sources = args.index or _configured_index(_config_path(args), required=args.cmd == "sync")
        try:
            # The configured index is cached for the panel as well; a one-run --index is not
            # the panel's, so it leaves the cache alone.
            index = w.load_index(sources) if args.index else w.refresh_index(sources, root)
        except w.WeightsError:
            # Offline: `list` degrades to the installed keys while the index is the built-in
            # one. One the user NAMED (here, in the config or its defaults) stays a hard
            # error, and fetch/sync need an index.
            if args.cmd != "list" or args.index or sources != list(w.DEFAULT_INDEX_SOURCES):
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
