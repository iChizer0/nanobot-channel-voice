"""Channel-package manifest for nanobot >= 0.3.5.

Core discovers channels as subpackages of ``nanobot.channels`` carrying a dependency-free
``manifest.py`` (see ``nanobot.channels.plugin``), not via an entry-point group. This shim
ships FROM the nanobot-channel-voice wheel into that namespace, leaving core untouched.
The implementation stays in ``nanobot_channel_voice``; the contract requires the runtime
target to live inside ``nanobot.channels.voice``, which ``runtime.py`` satisfies by
re-export.
"""

import base64
import dataclasses
from dataclasses import dataclass
from importlib.resources import files
from importlib.util import find_spec
from typing import Any

from nanobot.channels._manifest import field
from nanobot.channels.contracts import (
    ChannelSetupSpec,
    ChannelValidationContext,
)
from nanobot.channels.plugin import ChannelPlugin


def _validate(values: dict[str, Any], _context: ChannelValidationContext) -> dict[str, Any]:
    """Run the plugin schema over the merged section (the saved one under the panel's
    pending patch), then explain what it resolved to: the check rows, the identity line
    and the ``form`` spec are the panel's Resolved setup. Core shows at most 6 rows, a
    "pass" row as its label alone and ``manual_review`` never, and renders a message as
    plain text, so a row names options by their form labels and quotes what is typed.
    Non-schema checks never "fail": that would block saving the edit itself. Lazy
    imports keep the manifest importable with no plugin deps."""
    from nanobot.channels.validation import check, status_from_checks

    checks: list[dict[str, Any]] = []
    try:
        from nanobot_channel_voice.config import VoiceConfig, resolve_openai_key

        cfg = VoiceConfig.model_validate(values)
    except Exception as exc:  # noqa: BLE001 - pydantic detail becomes the check message
        detail = _schema_detail(exc)
        checks.append(check("schema", "Voice configuration", "fail", detail))
        payload = status_from_checks("voice", checks, [])  # voice declares no required fields
        payload["message"] = detail
        payload.update(_lenient_form(values))
        return payload
    from nanobot_channel_voice.webui_form import Store, build_form, choice_label

    # No pass row for the schema: the identity line names the backend, and the WebUI
    # renders the first six rows only, which the cloud path can fill.
    backend = choice_label("backend", cfg.backend)
    rt = cfg.realtime
    store = Store.read(cfg.device)
    if cfg.backend == "local":
        checks.append(_pipeline_check(cfg, check, store))
        # What the section SET, not what the schema holds: no hand-kept field list to drift.
        if "realtime" in _touched(cfg):
            checks.append(
                check(
                    "realtime_unused", "Cloud settings", "skipped",
                    "The provider settings are kept for the cloud backends, Local does "
                    "not use them.",
                )
            )
        # "skipped", not "fail": the channel starts keyless (local/system TTS need none).
        # Both OpenAI dialects run _build_openai, which raises without a key or a base URL.
        if (
            cfg.tts.enabled
            and cfg.tts.provider in ("openai", "openai_compat")
            and not resolve_openai_key(cfg.tts.api_key)
            and not cfg.tts.api_base
        ):
            checks.append(
                check(
                    "tts_key", "Text-to-speech key", "skipped",
                    f"{choice_label('tts.provider', cfg.tts.provider)} text-to-speech has no "
                    "API key. Fill it in under Text-to-speech, export OPENAI_API_KEY in the "
                    "gateway environment, or point the API base URL at a keyless local server.",
                )
            )
    else:
        # Mirrors _build_cloud()'s first statement: every dialect loads the same transport.
        if find_spec("websockets") is None:
            checks.append(
                check(
                    "realtime_module", "Provider transport", "skipped",
                    "The cloud backends speak over websockets, which is not installed: "
                    "pip install 'nanobot-channel-voice[realtime]'.",
                )
            )
        # Mirrors start(): the OpenAI dialects share the OPENAI_API_KEY fallback, gemini
        # reads the Google SDK variables and never OPENAI_API_KEY.
        if cfg.backend == "gemini":
            from nanobot_channel_voice.backend.gemini_live import resolve_gemini_key

            resolve_key, env_var = resolve_gemini_key, "GEMINI_API_KEY or GOOGLE_API_KEY"
        else:
            resolve_key, env_var = resolve_openai_key, "OPENAI_API_KEY"
        # One slot, three states: env-fallback on a non-OpenAI dialect warns (start() WILL
        # send that key and the provider rejects it), no key nudges, a real key is silent.
        if (
            not rt.api_key
            and cfg.backend not in ("openai", "gemini")
            and resolve_openai_key(None)
        ):
            checks.append(
                check(
                    "realtime_key", "Provider key", "skipped",
                    f"{backend} would send the exported OPENAI_API_KEY, which it rejects. "
                    "Fill in the API key under Provider.",
                )
            )
        elif not resolve_key(rt.api_key):
            # The env-export alternative is offered ONLY where that key would work.
            hint = (
                f" or export {env_var} in the gateway environment."
                if cfg.backend in ("openai", "gemini")
                else "."
            )
            checks.append(
                check(
                    "realtime_key", "Provider key", "skipped",
                    f"{backend} needs an API key. Fill it in under Provider{hint}",
                )
            )
        # Mirrors start(): an open mic on the cloud path needs a real canceller (there is
        # no transcript to filter the reply out of), or the shell's gate. The software one
        # is only there with its binding, which start() imports inside the same branch.
        webrtc_aec = cfg.aec == "webrtc" and find_spec("livekit") is not None
        if rt.barge_in == "aec" and not (rt.aec_available or cfg.full_duplex or webrtc_aec):
            remedy = (
                "The WebRTC canceller needs its binding: pip install "
                "'nanobot-channel-voice[aec]'."
                if cfg.aec == "webrtc"
                else "Under Provider in Advanced, set Echo cancellation to WebRTC or "
                "Hardware, or Barge-in to Gated."
            )
            checks.append(
                check("realtime_aec", "Open mic", "skipped", f"Open mic barge-in needs echo cancellation. {remedy}")
            )
        # Mirrors the start()-time fail-fast: azure is the one profile with no default
        # endpoint (the URL names your resource).
        if cfg.backend == "azure" and not rt.base_url:
            checks.append(
                check(
                    "realtime_endpoint", "Provider endpoint", "skipped",
                    "Azure OpenAI has no default endpoint. Fill in the Endpoint under "
                    "Provider with your resource URL.",
                )
            )
        # Blocks the cloud path DOES read (channel._build_gate / _start_stt_server): the
        # gated uplink runs vad.* and, under uplink="wake", wake.*; stt.serve loads stt.*.
        gated = rt.uplink != "server"
        if (gate := _gate_check(cfg, check, store)) is not None:
            checks.append(gate)
        cloud_blocks = {"realtime", "bargeIn"}  # bargeIn is read below, phrase by phrase
        if gated:
            cloud_blocks.add("vad")
            if rt.uplink == "wake":
                cloud_blocks.add("wake")
        if cfg.stt.serve.enabled:
            cloud_blocks.add("stt")
        # The rest is local-only and the row NAMES what the section set: what a block
        # leaves at its default is not a setting, and the form shows none of them here.
        touched = _touched(cfg)
        unused = [name for name in _LOCAL_BLOCKS if name in touched and name not in cloud_blocks]
        if cfg.context:  # an emptied context is the absence of one, not a setting
            unused.append("context")
        # bargeIn is only PARTLY local: the realtime backend reads its stop/ack phrases,
        # the gate its suspicion-duck onset.
        cloud_read = {"stopPhrases", "ackPhrases"} | ({"duckStartFrames"} if gated else set())
        if _touched(cfg.barge_in) - cloud_read:
            unused.append("bargeIn")
        if unused:
            checks.append(
                check(
                    "local_unused", "On-device settings", "skipped",
                    f"{backend} is end-to-end speech-to-speech, so the on-device settings "
                    f"({', '.join(unused)}) are not used.",
                )
            )

    # The one row that is always "skipped": devices open at start, so the summary line
    # reads "not verified" — this row is why. It also says where the names live, the
    # rows being hardware facts behind Advanced.
    checks.append(
        check(
            "audio_devices", "Audio devices", "skipped",
            f"The microphone '{cfg.audio.capture_device}' and speaker "
            f"'{cfg.audio.playback_device}' are opened when the channel starts. Change them "
            "under Audio in Advanced.",
        )
    )
    payload = status_from_checks("voice", checks, [], identity=_identity(cfg))
    payload["form"] = build_form(cfg, store)
    return payload


# Blocks the local pipeline runs and a speech-to-speech provider does not, in the order a
# row should read them. The names are the config's own, as the form spells them.
_LOCAL_BLOCKS = ("vad", "stt", "tts", "chunker", "prologue", "perf", "wake", "goal", "earcons")


def _touched(model: Any) -> set[str]:
    """The block's keys this section SET, a nested block counting only where a leaf under
    it is set. Pydantic's own equality would also weigh the device each block carries
    (a private attribute), so every board config would read as "set"."""
    dumped = model.model_dump(exclude_defaults=True, by_alias=True)
    return {k for k, v in dumped.items() if not (isinstance(v, dict) and not _set_leaves(v))}


def _set_leaves(block: dict[str, Any]) -> bool:
    return any(not (isinstance(v, dict) and not _set_leaves(v)) for v in block.values())


def _lenient_form(values: dict[str, Any]) -> dict[str, Any]:
    """A refused section still gets a form, shaped from what does validate of it, so the
    edit that broke it (or the key an older config carries) can be fixed in place: the
    panel lays the pending values over the fields. Empty only when shaping itself fails."""
    from nanobot_channel_voice.webui_form import build_form, lenient_config

    try:
        return {"form": build_form(lenient_config(values))}
    except Exception:  # noqa: BLE001 - a row explains the refusal; the shape is best effort
        return {}


def _identity(cfg: Any) -> dict[str, str]:
    """The summary line's "who": for a chat channel the bot account, for voice the
    backend and what it listens and speaks with."""
    from nanobot_channel_voice.webui_form import choice_label

    backend = choice_label("backend", cfg.backend)
    if cfg.backend == "local":
        return {"name": backend, "workspace": ", ".join(_engines(cfg))}
    if cfg.backend == "gemini":
        from nanobot_channel_voice.backend.gemini_live import DEFAULT_MODEL

        model = cfg.realtime.model or DEFAULT_MODEL
    else:
        from nanobot_channel_voice.backend.profiles import resolve_profile

        model = cfg.realtime.model or resolve_profile(cfg.backend).default_model
    return {"name": backend, "workspace": model}


def _engines(cfg: Any) -> list[str]:
    """The local trio by label, the identity line's summary of the pipeline."""
    from nanobot_channel_voice.webui_form import choice_label

    engines = [choice_label("vad.engine", cfg.vad.engine), choice_label("stt.provider", cfg.stt.provider)]
    engines.append(choice_label("tts.provider", cfg.tts.provider) if cfg.tts.enabled else "no TTS")
    return engines


def _schema_detail(exc: Exception) -> str:
    """A refusal as one sentence per error: the key it names and pydantic's reason,
    without the type/input/URL trailer of ``str(ValidationError)``."""
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        return " ".join(str(exc).split())[:300]
    parts = []
    for error in errors():
        message = error["msg"].removeprefix("Value error, ").removeprefix("Assertion failed, ")
        location = ".".join(str(part) for part in error["loc"])
        parts.append(f"{location}: {message}" if location and location not in message else message)
    return ". ".join(parts)[:300]


def _pipeline_check(cfg: Any, check: Any, store: Any) -> dict[str, Any]:
    """One row for the resolved local engines: "pass" names the trio, "warn" says which
    selected engine would fall back at start, why, and what stands in until then. The
    remedy is the panel's own: Apply for a model the index lists, the section's Model
    row for a block without one, the pip extra for a missing module."""
    from nanobot_channel_voice import stt, tts, vad, wake
    from nanobot_channel_voice.config import transcription_gap
    from nanobot_channel_voice.engines import preflight
    from nanobot_channel_voice.webui_form import choice_label

    # (engine, its section as the panel shows it, what stands in, why it would) per slot;
    # a stand-in of None is a slot the channel does not start without.
    listening = "Listening in Advanced"
    stt_slot = (
        choice_label("stt.provider", cfg.stt.provider), "Speech-to-text",
        # Serving borrows this engine, and _start_stt_server raises where the pipeline
        # alone would have fallen back to Internal.
        None if cfg.stt.serve.enabled else "Internal transcribes",
        preflight(cfg.stt, cfg.stt.provider, stt.ENGINES, prefix="stt."),
    )
    turn_slot = (
        choice_label("vad.turn.engine", cfg.vad.turn.engine), listening,
        "turns end on silence alone",
        preflight(cfg.vad, cfg.vad.turn.engine, vad.TURN_ENGINES, prefix="vad.", block="turn"),
    )
    slots = [
        (
            choice_label("vad.engine", cfg.vad.engine), listening, "Energy listens",
            preflight(cfg.vad, cfg.vad.engine, vad.ENGINES, prefix="vad."),
        ),
        turn_slot,
        stt_slot,
        (
            choice_label("tts.provider", cfg.tts.provider), "Text-to-speech", "System speaks",
            preflight(cfg.tts, cfg.tts.provider, tts.ENGINES, prefix="tts.")
            if cfg.tts.enabled
            else None,
        ),
        (
            choice_label("wake.engine", cfg.wake.engine), "Wake word",
            "Transcript matches the wake word",
            preflight(cfg.wake, cfg.wake.engine, wake.ENGINES, prefix="wake.")
            if cfg.wake.mode != "off"
            else None,
        ),
    ]
    degraded = [slot for slot in slots if slot[3] is not None]
    sentences = _why_sentences([(engine, section, why) for engine, section, _, why in degraded], store)
    if stand_ins := [stand_in for _, _, stand_in, _ in degraded if stand_in]:
        sentences.append(f"Until then {_join(stand_ins)}.")
    fatal = [engine for engine, _, stand_in, _ in degraded if not stand_in]
    if fatal:
        sentences.append(
            f"Serve transcription is on, so the channel does not start without {_join(fatal)}."
        )
    # The turn slot's other fallback, the one preflight cannot see. Not said of a model
    # that would not load either: it is not consulted for the louder reason.
    if turn_slot[3] is None and (idle := _turn_idle(cfg)) is not None:
        sentences.append(idle)
    # Delegated STT with nothing behind it decodes every utterance to "", which the
    # pipeline cannot tell from silence: the channel would start and hear nothing.
    if cfg.stt.provider == "nanobot" and (gap := transcription_gap()) is not None:
        sentences.append(
            f"Internal delegates to nanobot's transcription, but {gap}, so every utterance "
            "would be heard as silence."
        )
    if sentences:
        return check("pipeline", "Local pipeline", "skipped" if fatal else "warn", " ".join(sentences))
    return check("pipeline", "Local pipeline", "pass", f"{', '.join(_engines(cfg))}.")


def _why_sentences(degraded: list[tuple[str, str, Any]], store: Any) -> list[str]:
    """Why each (engine, its section, ``Fallback``) would not load, as the panel's own
    remedies: Apply for a model the index lists, the section's Model row for a block
    without one, the pip extra for a missing module."""
    sentences = []
    unfetched = [(engine, why.key) for engine, _, why in degraded if why.key]
    if unfetched:
        one = len(unfetched) == 1
        index = store.index or {}
        listed = all(key in index for _, key in unfetched)
        # A notice in the index holds Apply until it is accepted under Models, so the
        # remedy says so: the row would otherwise promise a download the button refuses.
        noticed = [engine for engine, key in unfetched if (index.get(key) or {}).get("accept")]
        remedy = "."
        if listed:
            remedy = f", Apply downloads {'it' if one else 'them'}"
            if noticed and one:
                remedy += " once its notice under Models is accepted"
            elif noticed:
                plural = len(noticed) > 1
                remedy += (
                    f" once the {_join(noticed)} notice{'s' if plural else ''} under Models "
                    f"{'are' if plural else 'is'} accepted"
                )
            remedy += "."
        sentences.append(
            f"The {_join([engine for engine, _ in unfetched])} model{'' if one else 's'} "
            f"{'is' if one else 'are'} not fetched yet{remedy}"
        )
    for engine, section, why in degraded:
        if why.unset:
            sentences.append(f"{engine} has no model, pick one under {section}.")
        elif why.module:
            sentences.append(
                f"{engine} needs the {why.extra} extra, pip install "
                f"'nanobot-channel-voice[{why.extra}]'."
                if why.extra
                else f"{engine} needs the module '{why.module}'."
            )
        elif why.error:
            sentences.append(f"{engine} cannot load its model ({why.error}).")
    return sentences


def _turn_idle(cfg: Any) -> str | None:
    """``make_turn_analyzer``'s static fallback that no preflight sees: a consult window at
    or past the hangover loads the turn model and never asks it. The window is no panel
    row, so it is named as the config key it is."""
    from nanobot_channel_voice import vad
    from nanobot_channel_voice.webui_form import choice_label

    turn = cfg.vad.turn
    if turn.engine not in vad.TURN_ENGINES or turn.consult_ms < cfg.vad.hangover_ms:
        return None
    return (
        f"{choice_label('vad.turn.engine', turn.engine)} is never consulted: "
        f"`vad.turn.consultMs` ({turn.consult_ms} ms) is at or past End of speech "
        f"({cfg.vad.hangover_ms} ms), so turns end on silence alone."
    )


def _gate_check(cfg: Any, check: Any, store: Any) -> dict[str, Any] | None:
    """The pipeline row's cloud counterpart, for the on-device engines the cloud path
    loads: the gate's, and the speech-to-text serving borrows. Mirrors ``_build_gate``
    and ``_start_stt_server``: the channel does not start on a detector that did not load
    (a fallback would upload on any noise, or never) nor on a served engine that did not,
    Smart Turn alone degrades. Nothing while everything loads."""
    from nanobot_channel_voice import stt, vad, wake
    from nanobot_channel_voice.engines import preflight
    from nanobot_channel_voice.webui_form import choice_label

    listening = "Listening in Advanced"
    gated = cfg.realtime.uplink != "server"
    needed = []
    if gated:
        needed.append((
            choice_label("vad.engine", cfg.vad.engine), listening,
            preflight(cfg.vad, cfg.vad.engine, vad.ENGINES, prefix="vad."),
        ))
        if cfg.realtime.uplink == "wake":
            needed.append((
                choice_label("wake.engine", cfg.wake.engine), "Wake word",
                preflight(cfg.wake, cfg.wake.engine, wake.ENGINES, prefix="wake."),
            ))
    if cfg.stt.serve.enabled:
        needed.append((
            choice_label("stt.provider", cfg.stt.provider), "Speech-to-text",
            preflight(cfg.stt, cfg.stt.provider, stt.ENGINES, prefix="stt."),
        ))
    turn = (
        choice_label("vad.turn.engine", cfg.vad.turn.engine), listening,
        preflight(cfg.vad, cfg.vad.turn.engine, vad.TURN_ENGINES, prefix="vad.", block="turn"),
    ) if gated else (None, None, None)
    fatal = [slot for slot in needed if slot[2] is not None]
    degraded = fatal + ([turn] if turn[2] is not None else [])
    idle = _turn_idle(cfg) if gated and turn[2] is None else None
    if not degraded and idle is None:
        return None
    sentences = _why_sentences(degraded, store)
    if fatal:
        sentences.append(f"The channel does not start without {'it' if len(fatal) == 1 else 'them'}.")
    if turn[2] is not None:
        sentences.append("Until then turns end on silence alone.")
    if idle is not None:
        sentences.append(idle)
    # Named after what asks for those engines: the gate, or serving alone.
    label = choice_label("realtime.uplink", cfg.realtime.uplink) if gated else "Serve transcription"
    return check("gate", label, "skipped" if fatal else "warn", " ".join(sentences))


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


@dataclass(frozen=True)
class _PasteBoxSpec(ChannelSetupSpec):
    """A setup spec whose one box renders as the primary field without being required.

    The WebUI places a field up front only when its payload says ``required``, and core
    derives that from ``required=`` — which the browser also enforces before enabling
    ("Required to complete setup."). The paste is optional by design (a bare section
    enables with pure defaults, and start() deletes the blob once expanded, so a
    re-enable would demand a fresh paste), hence the flag is set on the payload alone."""

    def to_public_dict(self, channel_name: str) -> dict[str, Any]:
        payload = super().to_public_dict(channel_name)
        for public_field in payload["fields"]:
            public_field["required"] = True
        return payload


SETUP_SPEC = _PasteBoxSpec(
    # ONE field: the channels.voice section pasted as JSON (bare or file-wrapped). The
    # schema lints it, it deep-merges (paste wins, partial pastes patch), and start()
    # expands it into real config.json keys, deleting the blob. json kind: a textarea, a
    # JSON-object check at save time, and the pending paste shown — echoed, keys and all,
    # to the authenticated session until start() applies it (taken knowingly). Export is
    # `nanobot-voice config`. No other fields, deliberately: core labels by LAST dotted
    # segment (stt/tts "Provider" would collide), materializes every declared field's
    # default into the section on toggle/save, and echoes every non-secret value.
    # allowFrom stays undeclared so the schema default ["*"] governs is_allowed: core's []
    # list filler would mean deny-everyone, unrecoverable without a pairing flow.
    fields={
        "importJson": field("json"),
    },
    # No `required=`: nothing gates enabling (the validator never reports missing fields
    # either), and feature.configured then simply tracks enabled instead of "paste pending".
    validator=_validate,
)

# The Settings tile: a channel installed from a wheel has no icon compiled into core's
# WebUI, so it rides the manifest as an inline image (no hosting, no fetch). Declared only
# where core has the field; an official core without it shows the "VO" initials.
_LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="#5b5bd6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="9" y="2" width="6" height="12" rx="3"/>'
    '<path d="M5 11a7 7 0 0 0 14 0"/>'
    '<path d="M12 18v4M8 22h8"/>'
    "</svg>"
)
_LOGO = {
    "logo_url": "data:image/svg+xml;base64,"
    + base64.b64encode(_LOGO_SVG.encode("ascii")).decode("ascii")
} if any(f.name == "logo_url" for f in dataclasses.fields(ChannelPlugin)) else {}

# The voice panel is compiled into core's WebUI from nanobot/channels/voice/webui/ and
# activates only when the manifest names it, so it is named only where that core shipped
# it next to this shim; elsewhere core renders its generic pane (the paste box).
_PANEL = "webui/index.tsx"
_WEBUI = (
    {"webui": _PANEL}
    if files("nanobot.channels").joinpath("voice", *_PANEL.split("/")).is_file()
    else {}
)

PLUGIN = ChannelPlugin(
    name="voice",
    display_name="Voice",
    runtime="nanobot.channels.voice.runtime:VoiceChannel",
    # The panel's Apply: core's start/poll/cancel routes, fetching the models the section
    # names before core (re)starts the channel.
    connector="nanobot.channels.voice.connect:VoiceSyncStore",
    setup=SETUP_SPEC,
    # Read by core's ensure_enabled_channel_dependencies: a no-op here (the manifest
    # only exists once the dist is installed), declared so a copy still names it.
    dependencies=("nanobot-channel-voice",),
    **_LOGO,
    **_WEBUI,
)
