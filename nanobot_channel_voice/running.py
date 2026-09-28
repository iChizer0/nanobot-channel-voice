"""What the running channel started without, for the WebUI's Resolved setup.

The validator runs in the gateway process but reads the config and the store, so a model
fetched since the start, or one that did not load, has it describe a stack the channel is
not running. A local start publishes its fallbacks here; the validator compares.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanobot_channel_voice.engines import Fallback


@dataclass(frozen=True)
class Started:
    """A local start: its config, and per slot it started without, what the store said
    then (a ``Fallback``, or None when the engine should have loaded)."""

    config: Any
    fell_back: dict[str, Fallback | None]


_current: tuple[object, Started] | None = None


def publish(owner: object, started: Started) -> None:
    global _current
    _current = (owner, started)


def withdraw(owner: object) -> None:
    """Only the owner's record: a stopped channel's teardown can land after its restart."""
    global _current
    if _current is not None and _current[0] is owner:
        _current = None


def current() -> Started | None:
    entry = _current  # one read: the validator runs in a worker thread
    return None if entry is None else entry[1]


def preflights(cfg: Any) -> dict[str, Fallback | None]:
    """Per slot the local pipeline runs (vad, turn, stt, tts, wake, in that order), why its
    selected engine would not load at start, None when it would. Slots not in use are absent."""
    from nanobot_channel_voice import stt, tts, vad, wake
    from nanobot_channel_voice.engines import preflight

    why = {
        "vad": preflight(cfg.vad, cfg.vad.engine, vad.ENGINES, prefix="vad."),
        "turn": preflight(cfg.vad, cfg.vad.turn.engine, vad.TURN_ENGINES, prefix="vad.", block="turn"),
        "stt": preflight(cfg.stt, cfg.stt.provider, stt.ENGINES, prefix="stt."),
    }
    if cfg.tts.enabled:
        why["tts"] = preflight(cfg.tts, cfg.tts.provider, tts.ENGINES, prefix="tts.")
    if cfg.wake.mode != "off":
        why["wake"] = preflight(cfg.wake, cfg.wake.engine, wake.ENGINES, prefix="wake.")
    return why


def fell_back(cfg: Any, *, vad: Any, turn: Any, stt: Any, tts: Any, wake: Any) -> dict[str, Fallback | None]:
    """The slots whose selected engine did not load, against the built pipeline, each with
    its preflight (why the store says it would not, None when it says it would)."""
    from nanobot_channel_voice import stt as stt_mod
    from nanobot_channel_voice import vad as vad_mod
    from nanobot_channel_voice import wake as wake_mod
    from nanobot_channel_voice.tts.system import SystemTtsAdapter

    missed = {
        "vad": cfg.vad.engine in vad_mod.ENGINES and isinstance(vad, vad_mod.EnergyVad),
        # A consult window at or past the hangover builds none by config, not by failure.
        "turn": cfg.vad.turn.engine in vad_mod.TURN_ENGINES and turn is None
        and cfg.vad.turn.consult_ms < cfg.vad.hangover_ms,
        "stt": cfg.stt.provider in stt_mod.ENGINES and stt is None,
        "tts": cfg.tts.provider != "system" and isinstance(tts, SystemTtsAdapter),
        "wake": cfg.wake.mode != "off" and cfg.wake.engine in wake_mod.ENGINES and wake is None,
    }
    if not any(missed.values()):
        return {}
    why = preflights(cfg)
    return {slot: why.get(slot) for slot, missing in missed.items() if missing}
