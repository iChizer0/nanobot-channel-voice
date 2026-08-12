"""Channel-package manifest for nanobot >= 0.3.0.

nanobot v0.3.0 discovers channels as subpackages of ``nanobot.channels`` carrying
a dependency-free ``manifest.py`` (see ``nanobot.channels.plugin``), not via an
entry-point group. This shim ships FROM the nanobot-channel-voice wheel into that
namespace: core is untouched; pip installs these three files alongside nanobot's
own channel packages and removes them with this dist. The implementation stays in
``nanobot_channel_voice``; the contract requires the runtime target to live inside
``nanobot.channels.voice``, which ``runtime.py`` satisfies by re-export.
"""

from typing import Any

from nanobot.channels._manifest import field
from nanobot.channels.contracts import (
    ChannelSetupSpec,
    ChannelValidationContext,
)
from nanobot.channels.plugin import ChannelPlugin


def _validate(values: dict[str, Any], _context: ChannelValidationContext) -> dict[str, Any]:
    """WebUI setup validation: run the plugin's OWN pydantic schema over the merged
    section, then explain what it resolved to. The form is one write-only Import
    Json box, so these rows are the WebUI's ONLY read surface: they must name the
    chosen backend, what it uses and ignores, and where the keys live. Core
    renders at most 6 checks; each branch stays within that. Lazy imports: the
    manifest must remain importable with no plugin dependencies loaded.
    Non-schema checks never "fail": every knob has a working default and a fail
    would block saving the paste itself.
    """
    from nanobot.channels.validation import check, status_from_checks

    checks: list[dict[str, Any]] = []
    try:
        from nanobot_channel_voice.config import VoiceConfig, resolve_openai_key

        cfg = VoiceConfig.model_validate(values)
    except Exception as exc:  # noqa: BLE001 - pydantic detail becomes the check message
        detail = " ".join(str(exc).split())[:300]
        checks.append(check("schema", "Voice configuration", "fail", detail))
        return status_from_checks("voice", checks, [])  # voice declares no required fields
    checks.append(
        check(
            "schema", "Voice configuration", "pass",
            f"Parses against the plugin schema; backend='{cfg.backend}'.",
        )
    )
    if cfg.import_json:
        # The paste already passed the schema above (it is merged before validation);
        # this row says what happens to it, since the box itself is write-only.
        checks.append(
            check(
                "import", "Config import", "pass",
                "importJson parses; it is merged over channels.voice (paste wins) and "
                "expanded into the real config.json keys when the channel starts.",
            )
        )

    rt = cfg.realtime
    if cfg.backend == "local":
        from nanobot_channel_voice.config import RealtimeConfig

        checks.append(_pipeline_check(cfg, check))
        # The realtime.* keys configure the cloud dialect family only; ANY
        # non-default value under backend='local' is almost certainly a
        # misplaced edit (default-compare, so no hand-kept field list to drift).
        if rt != RealtimeConfig():
            checks.append(
                check(
                    "realtime_unused", "Cloud settings", "skipped",
                    "backend='local' ignores the configured realtime.* keys: they "
                    "configure the cloud backends only.",
                )
            )
        # The tts_key note stays "skipped": the channel starts without a key
        # (local TTS servers and system TTS need none), and it names the
        # config-file remedies a paste or hand edit would apply.
        if (
            cfg.tts.enabled
            and cfg.tts.provider == "openai"
            and not resolve_openai_key(cfg.tts.api_key)
            and not cfg.tts.api_base
        ):
            checks.append(
                check(
                    "tts_key", "TTS API key", "skipped",
                    "tts.provider='openai' has no key: set tts.apiKey in config.json, export "
                    "OPENAI_API_KEY, or point tts.apiBase at a keyless local server.",
                )
            )
    else:
        # Same slot, three states: no key anywhere -> nudge to set one; only the env
        # fallback on a non-OpenAI backend -> start() WILL send the OpenAI key to a
        # provider that rejects it (resolve_openai_key's documented sharp edge), say
        # so; a real realtime.apiKey -> silent. At most one row, whatever the state.
        if not rt.api_key and cfg.backend != "openai" and resolve_openai_key(None):
            checks.append(
                check(
                    "realtime_key", "Realtime API key", "skipped",
                    f"backend='{cfg.backend}' will fall back to the exported "
                    "OPENAI_API_KEY, which non-OpenAI endpoints reject: set "
                    "realtime.apiKey.",
                )
            )
        elif not resolve_openai_key(rt.api_key):
            # The env-export alternative is offered ONLY where that key would work.
            hint = (
                "set realtime.apiKey (or export OPENAI_API_KEY in the gateway "
                "environment) before starting."
                if cfg.backend == "openai"
                else "set realtime.apiKey before starting."
            )
            checks.append(
                check(
                    "realtime_key", "Realtime API key", "skipped",
                    f"backend='{cfg.backend}': {hint}",
                )
            )
        # Mirror of the start()-time fail-fast: azure is the one profile with no
        # default endpoint (the URL names your resource).
        if cfg.backend == "azure" and not rt.base_url:
            checks.append(
                check(
                    "realtime_endpoint", "Realtime endpoint", "skipped",
                    "backend='azure' has no default endpoint: set realtime.baseUrl to your "
                    "resource URL (wss://<resource>.openai.azure.com/...) before starting.",
                )
            )
        from nanobot_channel_voice.config import (
            BargeInConfig,
            ChunkerConfig,
            PerfConfig,
            PrologueConfig,
            SttConfig,
            TtsConfig,
            VadConfig,
        )

        # Every local-only block, default-compared, and the row NAMES the touched
        # ones: with no form fields this is the only place a dead knob is visible.
        unused = [
            name
            for name, value, default in (
                ("vad", cfg.vad, VadConfig),
                ("stt", cfg.stt, SttConfig),
                ("tts", cfg.tts, TtsConfig),
                ("chunker", cfg.chunker, ChunkerConfig),
                ("prologue", cfg.prologue, PrologueConfig),
                ("perf", cfg.perf, PerfConfig),
                ("bargeIn", cfg.barge_in, BargeInConfig),
            )
            if value != default()
        ] + (["context"] if cfg.context else [])
        if unused:
            checks.append(
                check(
                    "local_unused", "Local pipeline", "skipped",
                    f"backend='{cfg.backend}' is end-to-end speech-to-speech: the "
                    f"configured local-only blocks ({', '.join(unused)}) are not used.",
                )
            )

    checks.append(
        check(
            "manual_review", "Audio devices", "skipped",
            f"audio.captureDevice='{cfg.audio.capture_device}', audio.playbackDevice="
            f"'{cfg.audio.playback_device}': probed when the channel starts. "
            "`nanobot-voice config` prints the whole section back, paste-ready.",
        )
    )
    return status_from_checks("voice", checks, [])


def _pipeline_check(cfg: Any, check: Any) -> dict[str, Any]:
    """One check that shows the resolved local engine trio and its config-file
    home: the form carries no engine fields (import-only surface), so this row
    is where the selection becomes visible. Downgraded to "warn" when a selected
    engine would silently fall back at start."""
    from nanobot_channel_voice import stt, tts, vad
    from nanobot_channel_voice.engines import preflight

    parts = [f"vad.engine='{cfg.vad.engine}'", f"stt.provider='{cfg.stt.provider}'"]
    parts.append(f"tts.provider='{cfg.tts.provider}'" if cfg.tts.enabled else "tts disabled")
    degraded = [
        f"{kind} '{engine}' would fall back ({reason})"
        for kind, engine, reason in (
            ("vad", cfg.vad.engine, preflight(cfg.vad, cfg.vad.engine, vad.ENGINES, prefix="vad.")),
            (
                "vad.turn",
                cfg.vad.turn.engine,
                preflight(cfg.vad, cfg.vad.turn.engine, vad.TURN_ENGINES, prefix="vad."),
            ),
            (
                "stt",
                cfg.stt.provider,
                preflight(cfg.stt, cfg.stt.provider, stt.ENGINES, prefix="stt."),
            ),
            (
                "tts",
                cfg.tts.provider,
                preflight(cfg.tts, cfg.tts.provider, tts.ENGINES, prefix="tts.")
                if cfg.tts.enabled
                else None,
            ),
        )
        if reason
    ]
    message = (
        f"{', '.join(parts)}: engines and their models are configured under "
        "channels.voice.{vad,stt,tts} in config.json."
    )
    if degraded:
        return check("pipeline", "Local pipeline", "warn", f"{message} {'; '.join(degraded)}.")
    return check("pipeline", "Local pipeline", "pass", message)


SETUP_SPEC = ChannelSetupSpec(
    # The WebUI surface is ONE field: the WHOLE channels.voice section pasted as JSON
    # (bare or still file-wrapped). The plugin schema lints it, deep-merges it over the
    # section (paste wins, partial pastes patch), and start() expands it into real
    # config.json keys, deleting the blob. secret kind though not a credential: core
    # never echoes a secret back to a browser (a paste may CONTAIN keys) and an
    # untouched empty box saves nothing. Write-only; the read/export surface is
    # `nanobot-voice config` (apiKeys scrubbed unless --secrets) plus _validate's
    # rows, which name the resolved backend, engines and devices.
    #
    # Individual form fields are deliberately NOT declared: core labels a field by
    # its LAST dotted segment (colliding leaves like stt/tts "Provider" would render
    # indistinguishably), core's toggle/save materializes every declared field's
    # default into the section as a camelCase sibling ('' for unset strings), and
    # every non-secret value is echoed to each browser session. One paste box keeps
    # the section hand-owned and the surface write-only.
    #
    # allowFrom is NOT declared either, on purpose: undeclared means the toggle
    # materializes nothing for it, and VoiceChannel always parses the section into
    # VoiceConfig, whose ["*"] default governs BaseChannel.is_allowed. (Declared
    # without a default it would materialize core's list filler [] = deny-everyone,
    # which voice, having no pairing flow, cannot recover from.)
    fields={
        "importJson": field("secret"),
    },
    # Every knob has a working default and _validate is authoritative; a "backend
    # is required" check would only make a valid bare section read as needs_setup.
    required=(),
    validator=_validate,
)

PLUGIN = ChannelPlugin(
    name="voice",
    display_name="Voice",
    runtime="nanobot.channels.voice.runtime:VoiceChannel",
    setup=SETUP_SPEC,
    # Read by core's ensure_enabled_channel_dependencies: a no-op here (the manifest
    # only exists once the dist is installed), declared so a copy still names it.
    dependencies=("nanobot-channel-voice",),
)
