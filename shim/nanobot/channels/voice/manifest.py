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

from nanobot.channels._manifest import field, required
from nanobot.channels.contracts import (
    ChannelSetupSpec,
    ChannelValidationContext,
)
from nanobot.channels.plugin import ChannelPlugin


def _validate(values: dict[str, Any], _context: ChannelValidationContext) -> dict[str, Any]:
    """Run the plugin schema over the merged section, then explain what it resolved
    to: the form is one write-only paste box, so these rows are the WebUI's only
    read surface. Core renders at most 6 checks; each branch stays within that.
    Lazy imports keep the manifest importable with no plugin deps. Non-schema
    checks never "fail": a fail would block saving the paste itself."""
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
        # Merged before validation, so it already passed the schema; the row says
        # what happens next, naming the export counterpart while someone pastes.
        checks.append(
            check(
                "import", "Config import", "pass",
                "importJson parses; it is merged over channels.voice (paste wins) and "
                "expanded into the real config.json keys when the channel starts. "
                "`nanobot-voice config` prints the section back, paste-ready.",
            )
        )

    rt = cfg.realtime
    if cfg.backend == "local":
        from nanobot_channel_voice.config import RealtimeConfig

        checks.append(_pipeline_check(cfg, check))
        # Default-compare (no hand-kept field list to drift): any non-default
        # realtime.* under backend='local' is almost certainly a misplaced edit.
        if rt != RealtimeConfig():
            checks.append(
                check(
                    "realtime_unused", "Cloud settings", "skipped",
                    "backend='local' ignores the configured realtime.* keys: they "
                    "configure the cloud backends only.",
                )
            )
        # "skipped", not "fail": the channel starts keyless (local/system TTS
        # need none); the message names the config-file remedies.
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
        # One slot, three states: env-fallback-on-non-OpenAI warns (start() WILL
        # send that key and the provider rejects it), no-key nudges, a real key
        # is silent.
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
            f"'{cfg.audio.playback_device}': probed when the channel starts.",
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
    # The whole surface is ONE field: the channels.voice section pasted as JSON (bare
    # or file-wrapped). The schema lints it, deep-merges it (paste wins, partial
    # pastes patch), and start() expands it into real config.json keys, deleting the
    # blob. secret kind: core never echoes secrets to a browser, and a paste may
    # contain keys. Export is `nanobot-voice config` plus _validate's rows.
    # No other fields, deliberately: core labels by LAST dotted segment (stt/tts
    # "Provider" would collide), materializes every declared field's default into
    # the section on toggle/save, and echoes every non-secret value to the browser.
    # allowFrom stays undeclared too: nothing materializes it, so the schema default
    # ["*"] governs is_allowed (declared without a default, core's [] list filler
    # would mean deny-everyone, unrecoverable without a pairing flow).
    fields={
        "importJson": field("secret"),
    },
    # Shapes the RENDERER only: one required field is the whole primary form, and
    # the "Advanced" section (which repeats every optional field — the same box
    # twice under required=()) disappears. It does NOT gate: with a custom
    # validator core adds no required-field checks, enable never consults it, and
    # _validate always reports missing=[], so a bare section still enables with
    # pure defaults (side effect: feature.configured now means "paste pending").
    required=(required("importJson"),),
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
