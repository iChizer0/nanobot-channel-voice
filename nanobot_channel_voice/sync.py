"""Bring the weights store in line with the config: what to fetch, what to remove, then do it.

Shared by ``nanobot-voice sync`` and the WebUI's Apply flow. The plan is pure (index +
store + the section), so a caller can show it before running; the run reports progress
per chunk and stops between chunks when asked. Automatic cleanup removes only keys a
``managed_by`` tag says the same flow installed: a hand-fetched model is the user's.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nanobot_channel_voice import weights as w


def config_weights_keys(node: Any) -> set[str]:
    """Every ``"weights": "<key>"`` string anywhere under a config node: structural, so
    new engine blocks need no change here."""
    found: set[str] = set()
    if isinstance(node, dict):
        for name, value in node.items():
            if name == "weights" and isinstance(value, str) and value:
                found.add(value)
            else:
                found |= config_weights_keys(value)
    elif isinstance(node, list):
        for value in node:
            found |= config_weights_keys(value)
    return found


def _resolved(section: dict[str, Any]) -> Any | None:
    """The section as the schema sees it, None when it refuses."""
    from pydantic import ValidationError

    from nanobot_channel_voice.config import VoiceConfig

    try:
        return VoiceConfig.model_validate(section)
    except ValidationError:
        return None


def used_weights_keys(cfg: Any) -> set[str]:
    """The ``weights`` keys the section's resolved setup runs: the selected engines' blocks
    (a bilingual voice's ``secondary`` too), the wake head under a mode on, and under a
    cloud backend only the gate's detectors and the served speech-to-text. A block an
    engine pick left behind names nothing here."""
    local = cfg.backend == "local"
    blocks: list[Any] = []
    if local or cfg.realtime.uplink != "server":
        blocks += [getattr(cfg.vad, cfg.vad.engine, None), cfg.vad.turn if cfg.vad.turn.engine != "none" else None]
    if local or cfg.stt.serve.enabled:
        blocks.append(getattr(cfg.stt, cfg.stt.provider, None))
    if local and cfg.tts.enabled:
        voice = getattr(cfg.tts, cfg.tts.provider, None)
        blocks += [voice, getattr(voice, "secondary", None)]
    if cfg.wake.mode != "off" and cfg.wake.engine == "openwakeword" and (local or cfg.realtime.uplink == "wake"):
        blocks.append(cfg.wake.openwakeword)
    return {block.weights for block in blocks if getattr(block, "weights", None)}


def backbone_wanted(cfg: Any, index: dict[str, dict[str, Any]]) -> str | None:
    """The openWakeWord backbone a head of your own needs: named by no ``weights`` key, so
    the structural scan misses it. The plan wants the build this host runs — the chip's
    when the index carries it, else the CPU one — not whichever build happens to be
    installed, which is the runtime's question (:func:`weights.backbone_key`)."""
    oww = cfg.wake.openwakeword
    if cfg.wake.mode == "off" or cfg.wake.engine != "openwakeword" or oww.weights or not oww.model_path:
        return None
    keys = [w.backbone_key_for(platform) for platform in w.host_platforms(oww.resolved_target)]
    return next((k for k in reversed(keys) if k in index), keys[0])


def voice_section(path: Path) -> dict[str, Any]:
    """The file's ``channels.voice`` as the channel resolves it: a not-yet-consumed WebUI
    paste merged in (it holds the whole section as ONE nested value the structural scan
    cannot see) and the operator's defaults laid beneath, the baseline's models included."""
    from nanobot_channel_voice.config import resolve_section

    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise w.WeightsError(f"cannot read nanobot config {path}: {exc}") from None
    section = dict((data.get("channels") or {}).get("voice") or {})
    try:
        return resolve_section(section)[0]
    except ValueError as exc:
        raise w.WeightsError(f"{path}: {exc}") from None


@dataclass
class SyncPlan:
    wanted: list[str]               # keys the section names
    fetch: list[str]                # wanted, not installed, in the index
    unknown: list[str]              # wanted, not installed, not in the index
    prune: list[str]                # installed keys the section no longer names
    sizes: dict[str, int] = field(default_factory=dict)  # per fetch key (index-declared) and prune key (on disk)
    free_bytes: int = 0
    notices: dict[str, str] = field(default_factory=dict)  # fetch keys whose entry demands acceptance

    @property
    def fetch_bytes(self) -> int:
        return sum(self.sizes[k] for k in self.fetch)

    @property
    def prune_bytes(self) -> int:
        return sum(self.sizes[k] for k in self.prune)

    @property
    def empty(self) -> bool:
        return not self.fetch and not self.prune


def plan_sync(
    section: dict[str, Any],
    index: dict[str, dict[str, Any]],
    root: Path,
    *,
    managed_by: str | None,
    prune: bool = True,
    used_only: bool = False,
) -> SyncPlan:
    """``managed_by`` limits removal to keys that tag installed; None removes every
    installed key the section does not name (the CLI's ``--prune``). ``used_only`` wants
    what the resolved setup runs (the panel's Apply), else every key the section names
    (the CLI's ``sync``). The one place a section is resolved: one the schema refuses
    plans every key it names, backbone excluded."""
    cfg = _resolved(section)
    named = used_weights_keys(cfg) if used_only and cfg is not None else config_weights_keys(section)
    backbone = backbone_wanted(cfg, index) if cfg is not None else None
    wanted = sorted(named | ({backbone} if backbone else set()))
    have = w.installed(root)
    fetch = [k for k in wanted if k not in have and k in index]
    unknown = [k for k in wanted if k not in have and k not in index]
    prune_keys = [
        k for k in have
        if prune and k not in wanted
        and (managed_by is None or w.managed_by(k, root) == managed_by)
    ]
    return SyncPlan(
        wanted=wanted,
        fetch=fetch,
        unknown=unknown,
        prune=prune_keys,
        sizes={**{k: w.entry_size(index[k]) for k in fetch}, **{k: w.disk_usage(have[k]) for k in prune_keys}},
        free_bytes=_free_bytes(root),
        notices={k: str(index[k]["accept"]) for k in fetch if index[k].get("accept")},
    )


def _free_bytes(root: Path) -> int:
    """Free space on the volume that holds (or will hold) the store; 0 when unknown."""
    probe = root
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return 0


def _key_progress(
    progress: Callable[[str, str, int], None], key: str, base: int, files: dict[str, int],
) -> Callable[[str, int], None]:
    """The per-file byte counts :func:`weights.fetch` reports (kept in ``files``), as one
    rising number: the files this key has finished plus the one in flight, over the keys
    before it."""

    def report(name: str, so_far: int) -> None:
        files[name] = so_far
        progress(key, name, base + sum(files.values()))

    return report


def run_sync(
    plan: SyncPlan,
    index: dict[str, dict[str, Any]],
    root: Path,
    *,
    managed_by: str | None,
    log: Callable[[str], None] = lambda _line: None,
    progress: Callable[[str, str, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[int, int]:
    """Fetch, then remove; returns (bytes fetched, bytes freed). ``progress`` gets
    (key, file name, bytes fetched so far by the whole run), which only rises;
    ``should_stop`` is honoured between chunks, between keys and before the removals.
    A model that fails does not keep the others off the device, and a run with any
    failure removes nothing: a failed or stopped run leaves the store as it was, plus
    whole models."""
    def check_stop() -> None:
        if should_stop is not None and should_stop():
            raise w.WeightsError("sync cancelled")

    done = 0
    failed: list[str] = []
    for key in plan.fetch:
        check_stop()
        log(key)
        files: dict[str, int] = {}
        try:
            w.fetch(
                key, index[key], root=root, log=log, managed_by=managed_by,
                progress=_key_progress(progress, key, done, files) if progress else None,
                should_stop=should_stop,
            )
        except (w.WeightsError, OSError) as exc:
            if should_stop is not None and should_stop():
                raise  # a cancel ends the run, naming the download it cut off
            failed.append(str(exc))
        # What it downloaded when that ran past the size declared (an index older than the
        # file, or one declaring none), so the next key's count starts where this one's ended.
        done += max(plan.sizes[key], sum(files.values()))
    if failed:
        landed = len(plan.fetch) - len(failed)
        raise w.WeightsError(
            "; ".join(failed) + (f" ({landed} of {len(plan.fetch)} fetched)" if landed else "")
        )
    check_stop()
    freed = 0
    for key in plan.prune:
        freed += w.prune(key, root)
        log(f"removed {key}")
    return done, freed
