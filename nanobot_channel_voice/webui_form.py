"""The WebUI's dynamic form: which knobs a section exposes, given the section.

Core's setup contract is a fixed field list, which cannot express a form whose fields depend
on the chosen backend or engine. So the manifest declares one transport field (the JSON
paste) and the validator returns this spec next to its checks; a panel renders it and
writes edits back as a paste patch. Kinds, choices and what may be null come from the
pydantic schema, so a field listed here can never drift from what the config accepts.
"""

from __future__ import annotations

import functools
import json
import types
import typing
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import TypeAdapter, ValidationError
from pydantic.alias_generators import to_camel, to_snake

from nanobot_channel_voice import weights as w
from nanobot_channel_voice.config import VoiceConfig, layer_defaults, merge_import, split_paste

FormKind = Literal["string", "secret", "int", "float", "bool", "enum", "list", "json", "weights"]

# The one place labels and help live; anything not listed is labelled from its last path
# segment. Keys are camelCase config paths. Help is one or two plain sentences that name
# options as the pills show them and wrap what a user would type (device names, env vars,
# commands) in backticks for the panel. "Empty" is the cleared input, which the panel
# writes as null where the schema allows it (``optional``).
_COPY: dict[str, tuple[str, str | None]] = {
    "backend": (
        "Backend",
        "Local runs speech-to-text and text-to-speech on this machine. The others stream "
        "audio to a realtime speech-to-speech provider.",
    ),
    "device": ("Device", None),  # help follows the index, see _device_help
    "allowFrom": ("Allowed senders", "Sender ids the channel answers, `*` for any. The microphone's sender id is `local`."),
    "audio.captureDevice": ("Microphone", "An ALSA device name such as `default` or `plughw:1,0`."),
    "audio.playbackDevice": ("Speaker", "An ALSA device name such as `default` or `plughw:1,0`."),
    "audio.backend": (
        "Audio backend",
        "ALSA runs `arecord` and `aplay`, ALSA in-process needs the `pyalsa` extra. None runs "
        "without audio.",
    ),
    "aec": (
        "Echo cancellation",
        "Auto mutes the microphone while a reply plays. Open mic keeps it open and filters "
        "out the reply's own words. WebRTC cancels the echo and needs the `aec` extra. "
        "Hardware trusts the device.",
    ),
    "stt.provider": ("Engine", "Internal uses nanobot's transcription provider. The others run on this device."),
    "stt.whisper.language": ("Language", "A language code such as `en` or `zh`."),
    "stt.sensevoice.language": ("Language", "A language code such as `zh` or `en`, or `auto` to detect it."),
    "stt.serve.enabled": (
        "Serve transcription",
        "Offer the on-device speech-to-text as an OpenAI-compatible endpoint, to nanobot's own "
        "transcription provider or any other client.",
    ),
    "stt.serve.host": ("Host", "`127.0.0.1` keeps the endpoint on this machine, `0.0.0.0` opens it to the network."),
    "stt.serve.port": ("Port", None),
    "stt.serve.apiKey": ("API key", "Optional. Callers then send it as a bearer token."),
    "tts.enabled": ("Speak replies", None),
    "tts.matcha.speed": ("Speed", "1 is the voice's own pace. Higher speaks faster."),
    "tts.mms.speakingRate": ("Speed", "1 is the voice's own pace. Higher speaks faster."),
    "tts.supertonic.speed": ("Speed", "1 is the voice's own pace. Higher speaks faster."),
    "tts.audioFormat": (
        "Audio format",
        "PCM streams the reply as it comes, so it can pause for you and carry the cues. WAV "
        "plays each piece whole.",
    ),
    "bargeIn.mode": (
        "Barge-in",
        "What a reply does while you speak over it. Duck turns it down and keeps going, Pause "
        "stops it and resumes if it was not an interruption.",
    ),
    "duckDb": ("Duck level", "Reply volume while you speak. 0 keeps it, -12 is about a quarter as loud."),
    "bargeIn.minWords": (
        "Interrupt after",
        "Words heard before the reply is cut. Fewer cut sooner, a misheard cough included.",
    ),
    "bargeIn.heardMarker": ("Heard marker", "Tell the agent how much of a cut reply you heard, so it picks up from there."),
    "prologue.enabled": ("Thinking filler", "A short phrase while the agent is still working."),
    "prologue.afterMs": (
        "First filler after",
        "The least wait before the first filler. It stretches with the usual first-reply time, "
        "so a filler marks a slow turn.",
    ),
    "prologue.intervalMs": ("Then every", "Spacing of the fillers that follow."),
    "prologue.phrases": (
        "Filler phrases",
        "Comma separated, spoken in order with the last repeating. Empty uses the built-ins for "
        "the reply voice.",
    ),
    "agentTimeoutS": (
        "Agent timeout",
        "Silence from the agent before the stall phrase warns, and twice that before the turn "
        "is given up. Empty turns the watch off, the stall notice with it.",
    ),
    "stallNoticeS": (
        "Stall notice after",
        "Time with nothing spoken on a live turn before the stall phrase, then twice as long "
        "before the next. Empty speaks it only when the agent goes silent for the whole timeout.",
    ),
    "stallPhrase": ("Stall phrase", "Spoken when a reply keeps you waiting, by the reply voice, so write it in its language."),
    "timeoutPhrase": ("Timeout phrase", "Spoken when the turn is given up, by the reply voice too."),
    "earcons.captured": ("Turn receipt", "A short rising tone when what you said is taken."),
    "earcons.attention": ("Attention cue", "A short falling tone when the wake window closes."),
    "earcons.gainDb": ("Gain", "Level of the cues, 0 as they come. The built-in tones peak around -15 dB."),
    "earcons.path": ("Receipt clip", "Optional. The path of a WAV to play instead of the built-in tone, cut to 600 ms."),
    "earcons.attentionPath": ("Attention clip", "Optional. The path of a WAV to play instead of the built-in tone, cut to 600 ms."),
    "wake.attention": (
        "Attention",
        "Conversation keeps listening for follow-ups after every turn. Sentence takes one turn "
        "per wake, unless the reply ends in a question.",
    ),
    "wake.windowS": (
        "Window",
        "How long after a wake, or a turn under Conversation, no phrase is needed. 0 asks for "
        "it every time.",
    ),
    "wake.aliases": ("Aliases", "Comma separated. Ways the transcript spells the phrase, such as `hey nano bot`."),
    "wake.ack.enabled": ("Acknowledge", "Say a short word, such as I'm here, when the phrase alone is spoken."),
    "wake.ack.phrases": ("Acknowledgements", "Comma separated, rotated. Empty uses the built-ins for the reply voice."),
    "vad.silero.threshold": (
        "Threshold",
        "Speech probability at or above which a frame counts as speech. Lower hears quieter "
        "speech, noise included.",
    ),
    "vad.firered.threshold": (
        "Threshold",
        "Speech probability at or above which a frame counts as speech. Lower hears quieter "
        "speech, noise included.",
    ),
    "realtime.persona": (
        "Persona",
        "Empty uses the built-in, a helpful, concise voice assistant keeping replies short. The "
        "provider speaks as this rather than as nanobot's own persona, and the tool rules are "
        "added for you, so say nothing about them here.",
    ),
    "realtime.idleParkS": ("Park after", "Idle time before the connection is parked, the next speech reopens it. 0 keeps it open."),
    "realtime.reasoningEffort": ("Reasoning", "High lets the model think before it answers, at a slower first word."),
    "logTranscripts": ("Log transcripts", "Include what you said in the gateway logs. Off logs word counts only."),
    "tts.provider": (
        "Engine",
        "OpenAI-compatible works with any `/audio/speech` server. System uses espeak-ng or "
        "say. The others run on this device.",
    ),
    "tts.model": ("Model", None),
    "tts.voice": ("Voice", None),
    "tts.apiBase": ("API base URL", "An OpenAI-compatible endpoint. Local servers without a key work too."),
    "tts.apiKey": ("API key", "Empty uses `OPENAI_API_KEY` from the gateway environment."),
    "vad.engine": ("Voice activity detector", "Energy reacts to any sound. FireRed and Silero are more accurate and need a model."),
    "vad.hangoverMs": ("End of speech", "Silence after speech that ends the turn."),
    "vad.turn.engine": (
        "End of turn",
        "None ends the turn when the silence runs out. Smart Turn can end it sooner by "
        "judging whether you sound finished, and needs a model.",
    ),
    "wake.mode": ("Mode", "Gate listens only after the wake phrase. Strict also ignores interruptions without it."),
    "wake.phrases": ("Phrases", "Comma separated, for example `hey nanobot`."),
    "wake.openwakeword.modelPath": (
        "Head model",
        "The path of a phrase head you trained with openWakeWord, an `.onnx` file. Its feature "
        "models are shared by every head, Apply fetches them.",
    ),
    "wake.openwakeword.threshold": ("Threshold", "Score at or above which the head counts as a hit. Lower hears more, false wakes included."),
    "realtime.apiKey": (
        "API key",
        "Empty uses `OPENAI_API_KEY` from the gateway environment, or `GEMINI_API_KEY` for Gemini.",
    ),
    "realtime.model": ("Model", "Empty uses the provider's default."),
    "realtime.baseUrl": ("Endpoint", "Required for Azure OpenAI, where the URL names your resource."),
    "realtime.voice": ("Voice", None),
    "realtime.uplink": (
        "Send audio",
        "Continuous streams the microphone all the time. On speech sends only speech, "
        "detected on this device. After wake word sends it once the phrase set under Wake "
        "word is heard.",
    ),
    "realtime.bargeIn": (
        "Barge-in",
        "Open mic keeps listening while a reply plays and needs echo cancellation. Gated "
        "pauses the microphone until the reply ends.",
    ),
    "realtime.toolMode": (
        "Tools",
        "Direct lets the provider call tools itself. Supervisor runs them through nanobot "
        "and speaks the finished answer.",
    ),
}
# Option labels, per path: the same value reads differently in different fields
# (`aec=soft` is an open microphone, `wake.engine=text` matches the transcript).
_CHOICES: dict[str, dict[str, str]] = {
    "backend": {
        "local": "Local", "openai": "OpenAI", "xai": "xAI", "azure": "Azure OpenAI",
        "qwen": "Qwen", "glm": "GLM", "stepfun": "StepFun", "gemini": "Gemini",
    },
    "audio.backend": {"alsa": "ALSA", "pyalsa": "ALSA in-process", "null": "None"},
    "aec": {"auto": "Auto", "soft": "Open mic", "webrtc": "WebRTC", "hardware": "Hardware"},
    "stt.provider": {"nanobot": "Internal", "whisper": "Whisper", "sensevoice": "SenseVoice", "zipformer": "Zipformer"},
    "tts.provider": {
        "openai": "OpenAI", "openai_compat": "OpenAI-compatible", "system": "System",
        "mms": "MMS", "supertonic": "Supertonic", "matcha": "Matcha",
    },
    "vad.engine": {"energy": "Energy", "webrtc": "WebRTC", "firered": "FireRed", "silero": "Silero"},
    "vad.turn.engine": {"none": "None", "smartturn": "Smart Turn"},
    "wake.mode": {"off": "Off", "gate": "Gate", "strict": "Strict"},
    "wake.engine": {"text": "Transcript", "openwakeword": "openWakeWord"},  # prose only, the Model row switches the tier
    "realtime.uplink": {"server": "Continuous", "vad": "On speech", "wake": "After wake word"},
    "realtime.bargeIn": {"aec": "Open mic", "gated": "Gated"},
    "realtime.reasoningEffort": {"none": "None", "high": "High"},
    "bargeIn.mode": {"duck": "Duck", "pause": "Pause"},
    "tts.audioFormat": {"wav": "WAV", "pcm": "PCM"},
    "wake.attention": {"conversation": "Conversation", "sentence": "Sentence"},
    "realtime.toolMode": {"direct": "Direct", "supervisor": "Supervisor"},
}
# The unit a number is in, shown at the input's end; the label stays the quantity.
_UNITS = {
    "vad.hangoverMs": "ms", "prologue.afterMs": "ms", "prologue.intervalMs": "ms",
    "stallNoticeS": "s", "agentTimeoutS": "s", "wake.windowS": "s", "realtime.idleParkS": "s",
    "duckDb": "dB", "earcons.gainDb": "dB", "bargeIn.minWords": "words",
}
# Rows behind the panel's Advanced toggle, in place in their sections: what a board image
# sets once, the internals the identity and pipeline rows already summarise, the tuning
# numbers and the phrase lists. A set value stays in force while its row is hidden.
_ADVANCED = frozenset({
    "device", "audio.captureDevice", "audio.playbackDevice", "aec", "tts.audioFormat",
    "bargeIn.mode", "realtime.bargeIn",
    "vad.engine", "vad.firered.weights", "vad.silero.weights", "vad.hangoverMs",
    "vad.turn.engine", "vad.turn.weights",
    "tts.matcha.speed", "tts.mms.speakingRate", "tts.supertonic.speed",
    "vad.firered.threshold", "vad.silero.threshold", "wake.openwakeword.threshold", "duckDb",
    "bargeIn.minWords", "prologue.afterMs", "prologue.intervalMs", "stallNoticeS",
    "agentTimeoutS", "wake.windowS", "earcons.gainDb",
    "audio.backend", "stt.serve.enabled", "stt.serve.host", "stt.serve.port", "stt.serve.apiKey",
    "bargeIn.heardMarker", "wake.attention", "wake.aliases", "earcons.path",
    "earcons.attentionPath", "logTranscripts", "realtime.idleParkS", "realtime.toolMode",
    "realtime.reasoningEffort",
    "prologue.phrases", "stallPhrase", "timeoutPhrase", "wake.ack.phrases",
    "allowFrom",
})
# A secret row is not a saved secret while it is pending: the patch rides the manifest's
# one json field, which the WebUI echoes back whole.
_SECRET_PENDING = "A key typed here rides the pending edits, readable under Config import, until the channel (re)starts."
_WEIGHTS_HELP = "Apply downloads the selected model. Custom takes any store key."
_WEIGHTS_HELP_NO_MODEL = "The index has no model for this engine. Custom takes any store key."
_WEIGHTS_HELP_NO_INDEX = "No model index is cached. Enter a store key fetched with `nanobot-voice fetch <key>`."
_DEVICE_HELP = "The chip the on-device models are built for, empty for CPU builds only."
_DEVICE_HELP_NO_INDEX = " No model index is cached, `rv1126b` is one such name."
_DEVICE_HELP_NO_CHIPS = " The index has no chip builds."
_WAKE_HELP = (
    "Transcript matches the phrase in the transcription. A head also hears its phrase in the "
    "audio and fills Phrases with it, Custom takes a head of your own."
)
_WAKE_HELP_NO_HEAD = (
    "Transcript matches the phrase in the transcription. The index lists no head, Custom takes "
    "one of your own."
)
# A cloud session has no STT, so no transcript tier: the heads and Custom alone.
_WAKE_HELP_CLOUD = "A head hears its phrase in the audio and fills Phrases with it, Custom takes a head of your own."
_WAKE_HELP_CLOUD_NO_HEAD = "The index lists no head, Custom takes one of your own."
_TEXT_TIER = {"value": "", "label": "Transcript", "sets": {"wake.engine": "text"}}
_ACOUSTIC_TIER = {"wake.engine": "openwakeword"}
# A section another section's switch turns off stays configurable under Advanced, and
# its note names that switch. The served STT under a cloud provider is Advanced outright.
_NOTE_NO_TTS = "Not in use until Speak replies is on."
# The cloud path has no transcript to filter, so an open mic there needs a real canceller:
# the row offers the two that are one, and the check row speaks while neither is picked.
_AEC_HELP_CLOUD = (
    "Open mic barge-in needs the reply's echo cancelled. WebRTC does it in software and needs "
    "the `aec` extra, Hardware trusts the device."
)
_AEC_CLOUD = ("webrtc", "hardware")
_NOTE_GATE = "Not in use until Send audio is On speech or After wake word."
_NOTE_WAKE_GATE = "Not in use until Send audio is After wake word."
_NOTE_SERVE = "The provider transcribes for itself. Serve transcription runs an engine on this device for other clients."
_DEVICE_HELP_UNUSED = " Not in use until an on-device detector or a served engine runs."


def choice_label(path: str, value: str) -> str:
    """The option's label, for prose that names it (check rows, the identity line)."""
    return _CHOICES.get(path, {}).get(value, _humanize(value))


def build_form(cfg: VoiceConfig, store: Store | None = None) -> dict[str, Any]:
    """Sections for the section as it resolves now; the panel re-asks after every edit.
    A section the setup does not use (another section's switch turns it off) is still
    there, advanced, its note naming the switch. ``store`` is the weights store as the
    caller read it, else read here."""
    dumped = cfg.model_dump(by_alias=True)
    if store is None:
        store = Store.read(cfg.device)

    def section(
        section_id: str, label: str, paths: list[str], *, note: str | None = None, advanced: bool = False,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {"id": section_id, "label": label, "fields": [_field(dumped, p, store) for p in paths]}
        if note or advanced:
            out["advanced"] = True
            out["note"] = note
        return out

    sections = [section("general", "General", ["backend", "device"])]
    if not _on_device(cfg):
        sections[0]["fields"][1]["help"] += _DEVICE_HELP_UNUSED
    # Audio is the hardware facts alone, every row advanced: the panel shows the section
    # under Advanced only.
    audio = ["audio.captureDevice", "audio.playbackDevice", "audio.backend"]
    if cfg.backend == "local":
        sections.append(section("audio", "Audio", audio))
        stt = ["stt.provider"]
        if cfg.stt.provider != "nanobot":
            stt.append(f"stt.{cfg.stt.provider}.weights")
            if cfg.stt.provider in ("whisper", "sensevoice"):
                stt.append(f"stt.{cfg.stt.provider}.language")
        stt.append("stt.serve.enabled")
        if cfg.stt.serve.enabled:
            stt += ["stt.serve.host", "stt.serve.port", "stt.serve.apiKey"]
        sections.append(section("stt", "Speech-to-text", stt))
        tts = ["tts.enabled"]
        if cfg.tts.enabled:
            tts.append("tts.provider")
            if cfg.tts.provider in ("openai", "openai_compat"):
                tts += ["tts.model", "tts.voice", "tts.audioFormat", "tts.apiBase", "tts.apiKey"]
            elif cfg.tts.provider != "system":
                tts.append(f"tts.{cfg.tts.provider}.weights")
                tts.append(_SPEED[cfg.tts.provider])
        sections.append(section("tts", "Text-to-speech", tts))
        spoken = None if cfg.tts.enabled else _NOTE_NO_TTS
        sections.append(section("interruptions", "Interruptions", _interruption_paths(cfg), note=spoken))
        sections.append(section("waiting", "Waiting", _waiting_paths(cfg), note=spoken))
        sections.append(section("vad", "Listening", _listening_paths(cfg)))
        sections.append(section("wake", "Wake word", _wake_paths(cfg) + _wake_extras(cfg, local=True)))
        cues = ["earcons.captured", "earcons.attention", "earcons.gainDb"]
        if cfg.earcons.captured:
            cues.append("earcons.path")
        if cfg.earcons.attention:
            cues.append("earcons.attentionPath")
        sections.append(section("cues", "Cues", cues))
    else:
        provider = ["realtime.apiKey", "realtime.model"]
        if cfg.backend == "azure" or cfg.realtime.base_url:
            provider.append("realtime.baseUrl")
        provider += ["realtime.voice", "realtime.persona", "realtime.uplink"]
        if cfg.realtime.uplink != "server":
            provider.append("realtime.idleParkS")
        provider.append("realtime.bargeIn")
        if cfg.realtime.barge_in == "aec":
            provider.append("aec")  # the canceller an open mic needs, right under its switch
        provider.append("realtime.toolMode")
        if cfg.backend == "xai":
            provider.append("realtime.reasoningEffort")
        sections.append(section("provider", "Provider", provider))
        for field in sections[-1]["fields"]:
            if field["key"] == "aec":
                field["help"] = _AEC_HELP_CLOUD
                field["choices"] = [c for c in field["choices"] if c["value"] in _AEC_CLOUD]
        sections.append(section("audio", "Audio", audio))
        # The provider transcribes; the on-device STT serves other clients, if anything.
        stt = ["stt.serve.enabled"]
        if cfg.stt.serve.enabled:
            stt.append("stt.provider")
            if cfg.stt.provider != "nanobot":
                stt.append(f"stt.{cfg.stt.provider}.weights")
                if cfg.stt.provider in ("whisper", "sensevoice"):
                    stt.append(f"stt.{cfg.stt.provider}.language")
            stt += ["stt.serve.host", "stt.serve.port", "stt.serve.apiKey"]
        sections.append(section("stt", "Speech-to-text", stt, note=_NOTE_SERVE, advanced=True))
        # The gate runs the local detectors; only then are their knobs live.
        gated = cfg.realtime.uplink != "server"
        sections.append(section(
            "vad", "Listening", _listening_paths(cfg), note=None if gated else _NOTE_GATE,
        ))
        sections.append(section(
            "wake", "Wake word", _wake_paths(cfg) + _wake_extras(cfg, local=False),
            note=None if cfg.realtime.uplink == "wake" else _NOTE_WAKE_GATE,
        ))
    sections.append(section("access", "Access", ["allowFrom", "logTranscripts"]))
    return {"sections": sections}


def lenient_config(values: dict[str, Any]) -> VoiceConfig:
    """A config for SHAPING the form when the section is refused: the pending paste merged
    as VoiceConfig merges it, then every field validated on its own, down through the
    blocks, a refused one keeping what does validate of it and the cross-field validators
    skipped. The fields then follow the choices made while the schema row says what is
    refused (a gate without phrases still shows the Model row that fills them)."""
    section, raw = split_paste(values)
    if raw is not None:
        try:
            section = merge_import(section, raw)
        except ValueError:
            pass  # an unusable paste shapes nothing; the row says so
    try:
        section = layer_defaults(section)  # the field-by-field fallback sees the baseline too
    except ValueError:
        pass  # unreadable defaults: the row says so, the section shapes alone
    return _shape(VoiceConfig, section)


def _shape(model: Any, data: Any) -> Any:
    """``model`` from ``data`` as far as it validates: whole, else field by field, a
    nested block shaped the same way and a refused leaf left at its default."""
    try:
        return _adapter(model).validate_python(data)
    except ValidationError:
        pass
    fields: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        for key in (to_camel(name), name):
            if key not in data:
                continue
            sub = _model_of(info.annotation)
            if sub is not None and isinstance(data[key], dict):
                fields[name] = _shape(sub, data[key])
            else:
                try:
                    fields[name] = _adapter(model, name).validate_python(data[key])
                except ValidationError:
                    pass
            break
    return model.model_construct(**fields)


@functools.cache
def _adapter(model: Any, name: str | None = None) -> TypeAdapter[Any]:
    """A validator for ``model``, or for one of its fields with the field's constraints."""
    if name is None:
        return TypeAdapter(model)
    info = model.model_fields[name]
    return TypeAdapter(typing.Annotated[(info.annotation, *info.metadata)] if info.metadata else info.annotation)


def _on_device(cfg: VoiceConfig) -> bool:
    """Whether the section runs on-device models: the local backend, or a cloud backend
    whose uplink gate runs the local detectors or that serves transcription."""
    return cfg.backend == "local" or cfg.realtime.uplink != "server" or cfg.stt.serve.enabled


@dataclass(frozen=True)
class Store:
    """The weights store as one form sees it, read once: the cached index (None when
    nothing is cached), the installed keys, and the platforms the section runs."""

    index: dict[str, dict[str, Any]] | None
    installed: frozenset[str]
    platforms: tuple[str, ...]

    @classmethod
    def read(cls, device: str | None) -> Store:
        cached = w.cached_index()
        return cls(cached[0] if cached else None, frozenset(w.installed()), w.host_platforms(device))


# The pace knob of each on-device voice, named as the engine names it.
_SPEED = {"matcha": "tts.matcha.speed", "mms": "tts.mms.speakingRate", "supertonic": "tts.supertonic.speed"}


def _streams(cfg: VoiceConfig) -> bool:
    """Whether the voice streams PCM (the on-device voices, an OpenAI-style one as PCM):
    playback can then pause and the heard marker can be measured; a WAV voice ducks."""
    if cfg.tts.provider in ("openai", "openai_compat"):
        return cfg.tts.audio_format == "pcm"
    return cfg.tts.provider != "system"


def _interruption_paths(cfg: VoiceConfig) -> list[str]:
    """The Interruptions section: its switch is ``aec``, whether the mic stays open while
    a reply plays (Auto mutes it, so nothing below applies and the rows follow the
    switch). Every row is advanced, the section an audio-path detail, so the panel shows
    it under Advanced only. Pause and the heard marker need a streaming voice, and the
    duck level matters while ducking is what happens."""
    if not cfg.open_mic:
        return ["aec"]
    streams = _streams(cfg)
    paths = ["aec"] + (["bargeIn.mode"] if streams else [])
    if not streams or cfg.barge_in.mode == "duck":
        paths.append("duckDb")
    paths.append("bargeIn.minWords")
    if streams:
        paths.append("bargeIn.heardMarker")
    return paths


def _waiting_paths(cfg: VoiceConfig) -> list[str]:
    """What is said while a reply is awaited: the filler with its timing and phrases once
    on, and the agent timeout, which arms the whole watch: the stall notice and both
    phrases follow it (the stall phrase is also the timeout's warning)."""
    paths = ["prologue.enabled"]
    if cfg.prologue.enabled:
        paths += ["prologue.afterMs", "prologue.intervalMs", "prologue.phrases"]
    paths.append("agentTimeoutS")
    if cfg.agent_timeout_s:
        paths += ["stallNoticeS", "stallPhrase", "timeoutPhrase"]
    return paths


def _vad_paths(cfg: VoiceConfig) -> list[str]:
    paths = ["vad.engine"]
    if cfg.vad.engine in ("firered", "silero"):
        paths += [f"vad.{cfg.vad.engine}.weights", f"vad.{cfg.vad.engine}.threshold"]
    return paths


def _listening_paths(cfg: VoiceConfig) -> list[str]:
    """The detector, when a turn ends and what decides it. A gated uplink endpoints with
    the same three, so the section reads alike on both paths."""
    return _vad_paths(cfg) + ["vad.hangoverMs"] + _turn_paths(cfg)


def _wake_extras(cfg: VoiceConfig, *, local: bool) -> list[str]:
    """What one wake buys, once a mode is on: the attention policy and window everywhere,
    the spoken ack and the transcript aliases where a local STT and TTS exist."""
    if cfg.wake.mode == "off":
        return []
    paths = ["wake.attention", "wake.windowS"]
    if local:
        paths += ["wake.aliases", "wake.ack.enabled"]
        if cfg.wake.ack.enabled:
            paths.append("wake.ack.phrases")
    return paths


def _turn_paths(cfg: VoiceConfig) -> list[str]:
    paths = ["vad.turn.engine"]
    if cfg.vad.turn.engine == "smartturn":
        paths.append("vad.turn.weights")
    return paths


def _wake_paths(cfg: VoiceConfig) -> list[str]:
    """The wake section by state. Off: the Mode row alone (a refused gate keeps its mode,
    the block being shaped field by field, so the Model row that fills the phrases still
    shows). On: the Model row leads and carries the tiers — Transcript, a head, Custom —
    every pick setting ``wake.engine``; Custom is a head of your own, so its path input
    follows; then Phrases, and the threshold once a head listens."""
    if cfg.wake.mode == "off":
        return ["wake.mode"]
    paths = ["wake.mode", "wake.openwakeword.weights"]
    oww = cfg.wake.openwakeword
    acoustic = cfg.wake.engine == "openwakeword"
    # A backbone key names no head, so the row that does still shows.
    if acoustic and (not _is_head(oww.weights or "") or oww.model_path):
        paths.append("wake.openwakeword.modelPath")
    paths.append("wake.phrases")
    if acoustic:
        paths.append("wake.openwakeword.threshold")
    return paths


def _field(dumped: dict[str, Any], path: str, store: Store | None = None) -> dict[str, Any]:
    kind, choices, optional, signed = _schema(path)
    label, help_text = _COPY.get(path, (_humanize(path.rsplit(".", 1)[-1]), None))
    value = _lookup(dumped, path)
    field: dict[str, Any] = {"key": path, "kind": kind, "label": label}
    if path in _UNITS:
        field["unit"] = _UNITS[path]
    if path in _ADVANCED:
        field["advanced"] = True
    if optional:
        field["optional"] = True  # an emptied input is null, its own value
    if signed:
        field["signed"] = True  # the numeric keypad has no minus key
    if path == "device":
        assert store is not None
        help_text = _device_help(value, store)
    if path.endswith(".weights"):
        assert store is not None
        prefix = _key_prefix(dumped, path)
        models = _weights_choices(prefix, store)
        field.update(kind="weights", label="Model", choices=models)
        help_text = (
            _WEIGHTS_HELP if models
            else _no_build_help(prefix, store) if store.index is not None
            else _WEIGHTS_HELP_NO_INDEX
        )
        if prefix == w.WAKE_PREFIX:
            help_text = _wake_field(field, dumped, store, models, help_text)
            if _lookup(dumped, "wake.engine") != "openwakeword":
                value = None  # a key without the engine is not the tier in force
    if path == "wake.phrases" and _lookup(dumped, "wake.engine") == "openwakeword":
        assert store is not None
        help_text = _phrase_help(_lookup(dumped, "wake.openwakeword.weights"), value or [], store) or help_text
    if kind == "secret":
        help_text = f"{help_text} {_SECRET_PENDING}" if help_text else _SECRET_PENDING
    if help_text:
        field["help"] = help_text
    if choices:
        labels = _CHOICES.get(path, {})
        field["choices"] = [{"value": c, "label": labels.get(c, _humanize(c))} for c in choices]
        for choice in field["choices"]:
            sets = _choice_sets(path, choice["value"], dumped, store)
            if sets:
                choice["sets"] = sets
    if kind == "secret":
        field["configured"] = bool(value)
    elif value not in (None, [], {}):
        field["value"] = value
    return field


def _choice_sets(
    path: str, choice: str, dumped: dict[str, Any], store: Store | None,
) -> dict[str, Any] | None:
    """The keys a choice writes with itself: an engine's first model, a gate's engines,
    the uplink a wake mode implies."""
    if path == "realtime.uplink":
        return _gate_sets(choice, dumped, store)
    if path == "wake.mode":
        return _wake_mode_sets(choice, dumped)
    return _engine_sets(path, choice, dumped, store)


# The detector a gated uplink runs while the config names none: the gate refuses Energy
# and WebRTC (every false onset is a billed upload), and its row sits under Advanced.
_GATE_DETECTOR = "silero"


def _gate_sets(choice: str, dumped: dict[str, Any], store: Store | None) -> dict[str, Any] | None:
    """Picking On speech or After wake word picks the neural detector the gate runs,
    Silero with its first model, unless one is set: the pick is then never refused for
    the detector, whose row is advanced. After wake word also turns the wake mode on,
    since the uplink refuses a wake gate without one; the head, the phrase, is the
    user's pick in the Wake word section that then shows. Leaving it takes back a mode
    that is still waiting for its phrase, which would else refuse the section from a
    section the form has folded away."""
    sets: dict[str, Any] = {}
    # A mode this pick turned on and nothing has filled goes with it; one the user filled
    # is theirs to keep. (This row is the cloud path's alone, so no local mode is touched.)
    if choice != "wake" and _lookup(dumped, "wake.mode") != "off" and not _lookup(dumped, "wake.phrases"):
        sets["wake.mode"] = "off"
    if choice == "server":
        return sets or None
    if _lookup(dumped, "vad.engine") not in ("silero", "firered"):
        sets |= {"vad.engine": _GATE_DETECTOR, **(_engine_sets("vad.engine", _GATE_DETECTOR, dumped, store) or {})}
    if choice == "wake" and _lookup(dumped, "wake.mode") == "off":
        sets["wake.mode"] = "gate"
    return sets or None


def _wake_mode_sets(choice: str, dumped: dict[str, Any]) -> dict[str, Any] | None:
    """Off under a cloud provider's After wake word keeps the gate on speech, the detector
    it set up still there: the uplink refuses a wake gate without a mode, and the section
    folds away as not in use."""
    if choice == "off" and dumped.get("backend") != "local" and _lookup(dumped, "realtime.uplink") == "wake":
        return {"realtime.uplink": "vad"}
    return None


# Engine rows whose choices own a weights block: ``<row>`` -> (block path, store prefix)
# by choice, ``{c}`` being the choice. Choices outside the map (nanobot, energy, system...)
# run no store model.
_ENGINE_BLOCKS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "stt.provider": (("whisper", "sensevoice", "zipformer"), "stt.{c}", "stt/{c}/"),
    "tts.provider": (("matcha", "mms", "supertonic"), "tts.{c}", "tts/{c}/"),
    "vad.engine": (("firered", "silero"), "vad.{c}", "vad/{c}/"),
    "vad.turn.engine": (("smartturn",), "vad.turn", "vad/{c}/"),
}


def _engine_sets(
    path: str, choice: str, dumped: dict[str, Any], store: Store | None,
) -> dict[str, Any] | None:
    """Picking an engine also picks its first listed model, while its block names no
    model and no file: the row then never opens on nothing selected, and the config says
    what runs. A block already set up keeps its setup."""
    engines, block_path, prefix = _ENGINE_BLOCKS.get(path, ((), "", ""))
    if choice not in engines or store is None:
        return None
    block = _lookup(dumped, block_path.format(c=choice)) or {}
    if block.get("weights") or any(k.endswith("Path") and v for k, v in block.items()):
        return None
    models = _weights_choices(prefix.format(c=choice), store)
    return {f"{block_path.format(c=choice)}.weights": models[0]["value"]} if models else None


def _key_prefix(dumped: dict[str, Any], path: str) -> str:
    """Store keys for a weights field start with ``<kind>/<engine>/``. The engine is the
    path's block (``stt.whisper.weights`` -> ``stt/whisper/``), except ``vad.turn``, the
    one block whose engine is its own choice (``vad/smartturn/``)."""
    kind, block = path.split(".")[:2]
    engine = _lookup(dumped, "vad.turn.engine") if (kind, block) == ("vad", "turn") else block
    return f"{kind}/{engine}/"


def _device_help(value: Any, store: Store) -> str:
    """The name stays typed, since the host's chip is a fact the index does not decide.
    The help names the chips the index has builds for, so the spelling is at hand, and
    calls out a name the index has nothing for."""
    socs = sorted(_socs(store.index or {}))
    if store.index is None:
        return _DEVICE_HELP + _DEVICE_HELP_NO_INDEX
    if not socs:
        return _DEVICE_HELP + _DEVICE_HELP_NO_CHIPS
    names = _literals(socs)
    # A refused section is shaped field by field, which skips the schema's normalisation,
    # so the name is matched as the schema would have.
    typed = str(value).strip().lower() if value else ""
    missing = f", none for `{value}`" if typed and typed not in socs else ""
    return f"{_DEVICE_HELP} The index has builds for {names}{missing}."


def _literals(names: list[str], joiner: str = "and") -> str:
    quoted = [f"`{name}`" for name in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} {joiner} {quoted[-1]}"


def _socs(index: dict[str, dict[str, Any]], prefix: str = "") -> set[str]:
    """The chips the index has RKNN builds for, under ``prefix``."""
    return {
        w.key_platform(k)[len("rknn."):]
        for k in index
        if k.startswith(prefix) and w.key_platform(k).startswith("rknn.")
    }


def _no_build_help(prefix: str, store: Store) -> str:
    """When the filter leaves nothing: what the index does have for this engine, and
    that Device is how one gets it."""
    socs = sorted(_socs(store.index or {}, prefix))
    if not socs:
        return _WEIGHTS_HELP_NO_MODEL
    return f"The index has this model for {_literals(socs, 'or')} only. Enter it under Device in Advanced to offer it."


def _wake_field(
    field: dict[str, Any], dumped: dict[str, Any], store: Store, heads: list[dict[str, Any]], help_text: str
) -> str:
    """The wake Model row as the tier switch: a Transcript pill ahead of the heads (read
    by phrase; local backend only, a cloud session has no STT for that tier), every pick
    writing ``wake.engine`` with it (``sets``), a head also filling Phrases with the phrase
    it hears unless Phrases has it. Custom is a head of your own: it clears the key like
    Transcript does, so ``customOpen`` (the engine without a key) tells the two apart."""
    phrases = _lookup(dumped, "wake.phrases") or []
    # Only the phrase the selected head hears goes with it: the rest were typed here.
    current = _head_phrase(_lookup(dumped, "wake.openwakeword.weights") or "", store)
    kept = [p for p in phrases if not current or p.casefold() != current.casefold()]
    for head in heads:
        phrase = _head_phrase(head["value"], store)
        head["sets"] = dict(_ACOUSTIC_TIER)
        if phrase:
            head["label"] = phrase.lower()
            wanted = (
                kept if phrase.casefold() in (p.casefold() for p in kept) else [*kept, phrase.lower()]
            )
            if wanted != phrases:
                head["sets"]["wake.phrases"] = wanted
    engine = _lookup(dumped, "wake.engine")
    local = dumped.get("backend") == "local"
    field.update(
        choices=[dict(_TEXT_TIER), *heads] if local else heads,
        custom="files",
        customOpen=engine == "openwakeword" and not _lookup(dumped, "wake.openwakeword.weights"),
        sets=dict(_ACOUSTIC_TIER),
    )
    if local:
        return _WAKE_HELP if heads else _WAKE_HELP_NO_HEAD
    return _WAKE_HELP_CLOUD if heads else _WAKE_HELP_CLOUD_NO_HEAD


def _phrase_help(key: str | None, phrases: list[str], store: Store) -> str:
    """A head hears one phrase, and Phrases must carry it or the transcript tier and the
    wake-phrase strip listen for other words. Said on the Phrases row, which is the one to
    edit and is never covered by a license notice, ahead of the start-time warning."""
    phrase = _head_phrase(key, store) if key else None
    if not phrase or phrase.casefold() in (p.casefold() for p in phrases):
        return ""
    return f"Comma separated. The selected head hears `{phrase.lower()}`, so include it."


def _head_phrase(key: str, store: Store) -> str | None:
    """What a head hears: the package meta once installed, else the index's naming (its
    stems are the phrase, hyphenated). Unknown for a custom key, and for a key that is
    not a head at all (another kind's, or the shared backbone) there is no phrase."""
    if not _is_head(key):
        return None
    if key in store.installed:
        try:
            phrase = json.loads((w.store_dir(key) / "meta.json").read_text(encoding="utf-8")).get("phrase")
        except (OSError, ValueError, AttributeError):
            phrase = None
        if isinstance(phrase, str) and phrase.strip():
            return phrase.strip()
    return _stem_phrase(key) if key in (store.index or {}) else None


def _stem_phrase(key: str) -> str:
    return key[len(w.WAKE_PREFIX):].rsplit("/", 1)[0].replace("-", " ")


def _is_head(key: str) -> bool:
    """A per-phrase head, not the backbone every head shares."""
    return key.startswith(w.WAKE_PREFIX) and not key.startswith(f"{w.WAKE_PREFIX}{w.BACKBONE_STEM}/")


def _weights_choices(prefix: str, store: Store) -> list[dict[str, Any]]:
    """The store keys a weights field can take here, one pill per stem: the cached index's
    entries under ``prefix`` for a platform the section runs, plus whatever is installed
    under it. A stem with several builds carries them as ``builds`` (CPU first, the device
    build last and the pill's own value, so it is the default); the languages ride along."""
    index = store.index or {}
    keys = {
        k for k in set(index) | set(store.installed)
        if k.startswith(prefix) and w.key_platform(k) in store.platforms
    }
    by_stem: dict[str, list[str]] = {}
    for key in keys:
        if not key.startswith(f"{w.WAKE_PREFIX}{w.BACKBONE_STEM}/"):  # feature models, not a head
            by_stem.setdefault(key[len(prefix):].rsplit("/", 1)[0], []).append(key)
    choices = []
    for stem, stem_keys in sorted(by_stem.items()):
        builds = sorted(stem_keys, key=lambda k: store.platforms.index(w.key_platform(k)))
        choice = _choice(builds[-1], stem, store)
        if len(builds) > 1:
            choice["builds"] = [_choice(k, _build_label(w.key_platform(k)), store) for k in builds]
        choices.append(choice)
    return choices


def _choice(key: str, label: str, store: Store) -> dict[str, Any]:
    entry = (store.index or {}).get(key) or {}
    choice: dict[str, Any] = {"value": key, "label": label, "installed": key in store.installed}
    if entry:
        choice["bytes"] = w.entry_size(entry)
        if entry.get("langs"):
            choice["langs"] = [str(lang) for lang in entry["langs"]]
        if entry.get("license"):
            choice["license"] = entry["license"]
        if entry.get("accept"):
            choice["notice"] = entry["accept"]
    return choice


def _build_label(platform: str) -> str:
    return "CPU" if platform == "onnx" else platform.removeprefix("rknn.").upper()


def _schema(path: str) -> tuple[FormKind, list[str], bool, bool]:
    """Form kind, enum choices, whether None is accepted and whether a number may be
    negative, from the pydantic schema, walking camelCase path segments."""
    model: Any = VoiceConfig
    info = None
    for segment in path.split("."):
        info = model.model_fields[to_snake(segment)]
        model = _model_of(info.annotation)
    assert info is not None
    base = info.annotation
    optional = False
    if typing.get_origin(base) in (typing.Union, types.UnionType):  # Optional[...] / X | None
        optional = type(None) in typing.get_args(base)
        base = next(a for a in typing.get_args(base) if a is not type(None))
    if typing.get_origin(base) is Literal:
        return "enum", [str(c) for c in typing.get_args(base)], optional, False
    if base is bool:
        return "bool", [], optional, False
    if base in (int, float):
        floor = next(
            (b for m in info.metadata for b in (getattr(m, "ge", None), getattr(m, "gt", None)) if b is not None),
            None,
        )
        return ("int" if base is int else "float"), [], optional, floor is None or floor < 0
    if typing.get_origin(base) is list:
        return "list", [], optional, False
    if typing.get_origin(base) is dict:
        return "json", [], optional, False
    last = path.rsplit(".", 1)[-1]
    return ("secret" if last in ("apiKey", "token", "secret") else "string"), [], optional, False


def _model_of(annotation: Any) -> Any:
    for candidate in (annotation, *typing.get_args(annotation)):
        if isinstance(candidate, type) and hasattr(candidate, "model_fields"):
            return candidate
    return None


def _lookup(dumped: dict[str, Any], path: str) -> Any:
    node: Any = dumped
    for segment in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(segment)
    return node


def _humanize(segment: str) -> str:
    words = "".join(f" {c.lower()}" if c.isupper() else c for c in segment).strip()
    return words[:1].upper() + words[1:]
