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
    section, then explain the CHOSEN backend: the renderer shows one static field
    list for both modes, so this is the only surface that can say which fields
    apply and what still lives in config.json. Core renders at most 6 checks;
    each branch stays within that. Lazy imports: the manifest must remain
    importable with no plugin dependencies loaded. Non-schema checks never
    "fail": every knob has a working default and a fail would block saving the
    form values themselves.
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
        check("schema", "Voice configuration", "pass", "Parses against the plugin schema.")
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
        checks.append(_pipeline_check(cfg, check))
        # The realtime.* fields configure the cloud dialect family only; a value
        # there under backend='local' is almost certainly a misplaced edit.
        if any((rt.api_key, rt.model, rt.voice, rt.base_url, rt.persona)):
            checks.append(
                check(
                    "realtime_unused", "Cloud settings", "skipped",
                    "backend='local' ignores the realtime.* fields (Api Key / Model / "
                    "Voice / Base Url): they configure the cloud backends only.",
                )
            )
        # The tts_key note stays "skipped": the channel starts without a key
        # (local TTS servers and system TTS need none) and it names its
        # config-file remedies because those keys are not WebUI form fields.
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
        if not resolve_openai_key(rt.api_key):
            checks.append(
                check(
                    "realtime_key", "Realtime API key", "skipped",
                    f"backend='{cfg.backend}': set realtime.apiKey (or export OPENAI_API_KEY "
                    "in the gateway environment) before starting.",
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
        from nanobot_channel_voice.config import SttConfig, TtsConfig, VadConfig

        if cfg.vad != VadConfig() or cfg.stt != SttConfig() or cfg.tts != TtsConfig():
            checks.append(
                check(
                    "local_unused", "Local pipeline", "skipped",
                    f"backend='{cfg.backend}' is end-to-end speech-to-speech: the local "
                    "vad/stt/tts engine blocks are configured but not used.",
                )
            )

    checks.append(
        check(
            "manual_review", "Audio devices", "skipped",
            "ALSA capture/playback devices are probed when the channel starts.",
        )
    )
    return status_from_checks("voice", checks, [])


def _pipeline_check(cfg: Any, check: Any) -> dict[str, Any]:
    """One check that shows the resolved local engine trio and its config-file
    home: the WebUI form carries only vad.engine (the stt/tts provider keys
    collide into two "Provider" labels under core's last-segment labelling, and
    their non-default values need file-side companions regardless). Downgraded
    to "warn" when a selected engine would silently fall back at start."""
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
    # Dotted keys resolve and persist through core's nested field helpers; names are
    # the camelCase wire forms the plugin's config accepts. Core's generic WebUI
    # renderer labels a field by its LAST dotted segment only (core's own channels
    # keep leaves unique via flat prefixed names: imapHost, smtpPassword), so every
    # key here must carry a unique, self-describing leaf; it also renders enum
    # choices SORTED alphabetically and value-derived ("Openai", "Xai"), accepted.
    # With no required fields the renderer shows the FIRST field as the whole
    # primary form and the rest under a collapsed "Advanced" section, so dict
    # order below is display order: backend first, then the cloud credential
    # block it selects, then the local audio/pipeline knobs. stt.provider /
    # tts.provider / tts.apiKey / stt.serve.* stay config-file surface: their
    # leaves collide into "Provider"/"Api Key"/"Enabled" labels, and their
    # non-default values need config-file companions (extras, model paths, serve
    # hardening) anyway; _validate's checks point at the file keys instead.
    fields={
        "backend": field(
            "enum",
            choices=("local", "openai", "xai", "azure", "qwen", "glm", "stepfun"),
            default="local",
        ),
        # The cloud dialect family's credential + overrides (flagged as ignored by
        # _validate under backend="local"; empty strings mean "profile default").
        "realtime.apiKey": field("secret"),
        "realtime.model": field(),
        "realtime.voice": field(),
        "realtime.baseUrl": field(),  # Azure resource URL / self-hosted / regional endpoint
        "audio.captureDevice": field(default="default"),
        "audio.playbackDevice": field(default="default"),
        "audio.sampleRate": field("int", default=16000),
        # The one local engine choice with a unique leaf ("Engine"). energy/webrtc
        # are completable from the browser; firered needs model paths in
        # config.json: the pipeline check warns when the selection would degrade.
        "vad.engine": field("enum", choices=("energy", "webrtc", "firered"), default="energy"),
        "aec": field("enum", choices=("auto", "soft", "webrtc", "hardware"), default="auto"),
        # Privacy: user speech stays OUT of gateway logs unless opted in
        # (OTel analog: telemetry.captureContent).
        "logTranscripts": field("bool", default=False),
        # Full-surface escape hatch: the WHOLE channels.voice JSON object pasted (bare or
        # file-wrapped); the plugin schema lints it, deep-merges it over the section
        # (paste wins), and start() expands it into real config.json keys, deleting the
        # blob. secret kind though not a credential: core never echoes it to a browser (a
        # paste may CONTAIN keys) and an untouched empty box saves nothing. Write-only;
        # read config.json for current state.
        "importJson": field("secret"),
        # Never rendered (core drops writable=False fields from the form), but the
        # default MUST mirror the schema's ["*"]: core's enable/disable toggle
        # materializes setup defaults into the saved section, and the generic
        # fallback for lists is []: deny-everyone under BaseChannel.is_allowed,
        # which voice (no pairing flow) cannot recover from.
        "allowFrom": field("list", writable=False, default=["*"]),
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
