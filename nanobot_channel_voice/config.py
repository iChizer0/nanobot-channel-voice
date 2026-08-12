"""Configuration models for the ``channels.voice`` section of ``~/.nanobot/config.json``.

All models inherit :class:`_VoiceBase`: keys parse as camelCase or snake_case, a typo is
a startup error, and a key present in both spellings folds per its docstring.
Frame-counted fields scale silently with ``audio.frameMs``; perceptual/algorithmic
constants are not configurable and stay in code beside their rationale (e.g. the duck
envelope in ``audio_sink.py``). A field only one backend/engine reads says so in its
comment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from nanobot.config.schema import Base
from pydantic import ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel, to_snake
from pydantic_core import PydanticUndefined

_UNSET = object()  # "this field has no comparable default" for the twin fold


def _emptyish(value: Any) -> bool:
    """None, or the empty ''/[]/{} core's writers materialize for never-set fields."""
    return value is None or value in ("", [], {})


def resolve_openai_key(explicit: str | None) -> str | None:
    """The ONE home of the ``OPENAI_API_KEY`` fallback, shared by ``tts.apiKey`` and
    ``realtime.apiKey``. Sharp edge: with a non-OpenAI realtime provider plus an
    OpenAI-compatible LOCAL TTS server, an exported OPENAI_API_KEY reaches both.
    """
    return explicit or os.environ.get("OPENAI_API_KEY")


def parse_import_blob(raw: Any) -> dict[str, Any]:
    """Parse the WebUI ``importJson`` paste into a plain section dict.

    Accepts the bare ``channels.voice`` object, or the same object still wrapped in
    ``{"channels": {"voice": ...}}`` / ``{"voice": ...}`` (people paste whole files);
    the wrapper keys are unambiguous because no VoiceConfig field is named either.
    Raises ``ValueError`` with a message fit for a WebUI check row.
    """
    if not isinstance(raw, str):
        raise ValueError("importJson must be a JSON string (the pasted object, quoted)")
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"importJson is not valid JSON: {exc}") from None
    if isinstance(parsed, dict) and isinstance(parsed.get("channels"), dict):
        parsed = parsed["channels"]
    if isinstance(parsed, dict) and isinstance(parsed.get("voice"), dict):
        parsed = parsed["voice"]
    if not isinstance(parsed, dict):
        raise ValueError("importJson must be a JSON object: the channels.voice section")
    parsed.pop("importJson", None)  # a paste of a not-yet-consumed section must not recurse
    parsed.pop("import_json", None)
    return parsed


def _apply_import(section: dict[str, Any], blob: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge a parsed paste over the section, the paste winning. Pasted keys are
    canonicalized to camelCase and REPLACE a section twin in either spelling, so a
    paste can never end up shadowed by ``_fold_alias_twins``'s filler heuristics."""
    out = dict(section)
    for key, value in blob.items():
        camel = to_camel(key) if isinstance(key, str) else key
        prior: Any = _UNSET
        if isinstance(camel, str):
            for twin in dict.fromkeys((camel, to_snake(camel), key)):
                if twin in out:
                    popped = out.pop(twin)
                    if prior is _UNSET:
                        prior = popped
        if isinstance(value, dict):
            base = prior if isinstance(prior, dict) else {}
            out[camel] = _apply_import(base, value)
        else:
            out[camel] = value
    return out


class _VoiceBase(Base):
    """nanobot's ``Base`` leaves pydantic's ``extra="ignore"``, so a typo'd key would parse
    and silently do nothing; ``forbid`` makes it a startup error (ChannelManager logs
    "voice channel not available: ...")."""

    model_config = ConfigDict(extra="forbid")  # merged onto Base's alias config

    @model_validator(mode="before")
    @classmethod
    def _fold_alias_twins(cls, data: Any) -> Any:
        """Core's WebUI writers spell keys camelCase LITERALLY, never case-folding against
        what a hand-written config already uses, so any WebUI write leaves a snake_case
        config carrying BOTH spellings of a field, which ``forbid`` would reject wholesale;
        fold the twins instead. Which wins depends on which writer plausibly produced the
        camelCase twin:

        - the settings form saves the SUBMITTED value under the camelCase key: a
          fresh edit, so camelCase wins (also pydantic's own alias priority);
        - the enable/disable toggle merges the manifest DEFAULTS in as camelCase
          siblings ('' for unset strings/secrets): filler, so a camelCase twin that
          is empty-ish or exactly the field default loses to hand-written data.

        Equal twins fold silently; a decided conflict warns (spellings only; either
        side may hold a secret) so a shadowed edit is never silently ignored."""
        if not isinstance(data, dict):
            return data
        twins = [
            (name, alias)
            for name in cls.model_fields
            if (alias := to_camel(name)) != name and alias in data and name in data
        ]
        if not twins:
            return data
        data = dict(data)  # the caller may hold (and re-save) the original mapping
        for name, alias in twins:
            camel, snake = data[alias], data.pop(name)
            if camel == snake:
                continue
            default = cls._comparable_default(name)
            keep_snake = (_emptyish(camel) and not _emptyish(snake)) or (
                default is not _UNSET and camel == default and snake != default
            )
            if keep_snake:
                data[alias] = snake
            logger.warning(
                "voice config: '{}' and '{}' spell one setting twice with different "
                "values; keeping the {} one. Delete the other from config.json",
                alias, name, f"'{name}'" if keep_snake else f"'{alias}'",
            )
        return data

    @classmethod
    def _comparable_default(cls, name: str) -> Any:
        info = cls.model_fields[name]
        if info.default is not PydanticUndefined:
            return info.default
        if info.default_factory is not None:
            return info.default_factory()  # type: ignore[call-arg]
        return _UNSET


class OnDeviceRuntime(_VoiceBase):
    """Accelerator knobs; every on-device engine block inherits these and reads its OWN.

    ``.rknn`` runs on rknn-toolkit-lite2 (``coreMask`` selects NPU cores;
    ``target``/``deviceId`` are the full-toolkit fallback). ``.onnx`` runs on onnxruntime
    with configurable execution providers (Jetson TensorRT/CUDA off the same artifact) and
    the parallel per-provider option list.
    """

    # Store key ("stt/whisper-base/onnx") installed by ``nanobot-voice fetch``: unset
    # ``*Path`` fields resolve from the fetched files by field stem (``encoderPath`` ->
    # ``encoder.<ext>``). Explicit paths always win.
    weights: str | None = None
    core_mask: Literal["auto", "0", "1", "2", "0_1", "0_1_2", "all"] = "auto"
    target: str = "rk3588"
    device_id: str | None = None
    execution_providers: list[str] | None = None
    provider_options: list[dict] | None = None


class AudioConfig(_VoiceBase):
    """ALSA audio I/O. The default backend shells out to ``arecord``/``aplay`` against
    named PCMs, so shared ``dsnoop``/``dmix`` ``plug`` devices work by name::

        "captureDevice": "plug:mic", "playbackDevice": "plug:speaker"
    """

    # "pyalsa" = in-process libasound ([pyalsa] extra); "null" = headless/no audio.
    backend: Literal["alsa", "pyalsa", "null"] = "alsa"
    capture_device: str = "default"   # ALSA PCM names, not card/device indices
    playback_device: str = "default"
    sample_rate: int = Field(default=16000, ge=8000, le=48000)  # mono S16_LE; 16k is whisper-native
    frame_ms: Literal[10, 20, 30] = 20  # capture granularity fed to the VAD
    arecord_path: str = "arecord"
    aplay_path: str = "aplay"
    # Playback-device latency AFTER our pacing (ALSA/dmix buffer + DAC): the AEC3
    # stream-delay hint (aec="webrtc"). AEC3 tracks the true delay continuously, so this only
    # speeds convergence on boards with deep dmix buffers. Above WebRTC's 500 ms ceiling the
    # APM rejects the hint and every capture frame raises.
    playout_delay_ms: int = Field(default=50, ge=0, le=500)


class FireRedVadConfig(OnDeviceRuntime):
    """On-device FireRedVAD (``vad.engine="firered"``). Models supplied by path."""

    model_path: str | None = None   # streaming-with-cache ".onnx" (CPU) or ".rknn" (NPU)
    cmvn_path: str | None = None    # cmvn.ark (Kaldi global CMVN stats)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    smooth_frames: int = Field(default=5, ge=1)  # rolling avg over 10 ms MODEL frames (1 = off)
    # Loudness gate AND'd with the model: frames whose normalized RMS (0..1, same unit
    # as vad.energyThreshold) fall below this are non-speech even when the model says
    # speech: cuts spurious ducks from distant TVs/radios. 0 = off (default).
    min_volume: float = Field(default=0.0, ge=0.0, lt=1.0)


class TurnConfig(OnDeviceRuntime):
    """Audio-native end-of-turn model layered over the endpointer
    (``vad.turn.engine="smartturn"`` = Smart Turn v3 over ONNX/RKNN, ``[ondevice]``
    extra; needs ``audio.sampleRate=16000``). At ``consultMs`` of trailing silence
    the utterance-so-far is scored once: COMPLETE closes the turn immediately
    (typically ~300 ms before ``hangoverMs``), INCOMPLETE waits out ``hangoverMs``:
    the silence timer becomes the hard upper bound instead of the decision, so
    ``hangoverMs`` can be raised for hesitant speakers without slowing every turn."""

    engine: Literal["none", "smartturn"] = "none"
    model_path: str | None = None  # ".onnx" (CPU) or ".rknn" (NPU)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # P(complete) above => close
    # Trailing silence before the model is consulted. Must be < hangoverMs to matter;
    # the default matches stt.eagerMs so the eager transcript and the verdict land
    # together. Consults fire once per pause.
    consult_ms: int = Field(default=240, ge=20)


class VadConfig(_VoiceBase):
    """Voice-activity detection / utterance endpointing. ``engine`` picks the per-frame
    detector; the endpointing knobs apply to all of them. ``energy`` is zero-dep;
    ``webrtc`` is spectral (``[webrtc]`` extra); ``firered`` is on-device neural over
    RKNN/ONNX (``[ondevice]`` extra, own block), best for noisy mics."""

    engine: Literal["energy", "webrtc", "firered"] = "energy"
    start_frames: int = Field(default=5, ge=1)      # consecutive speech frames to open an utterance
    preroll_ms: int = Field(default=300, ge=0)      # audio kept before onset so a slow VAD doesn't clip word 1
    hangover_ms: int = Field(default=600, ge=100)   # trailing silence that ends an utterance
    # Opt-in adaptive hangover: set to make the effective hangover START here (snappy)
    # and grow toward hangoverMs only on evidence the endpointer cut a real pause
    # short (the user resuming right after a close). None = fixed hangoverMs.
    hangover_min_ms: int | None = Field(default=None, ge=100)
    min_utterance_ms: int = Field(default=200, ge=0)
    max_utterance_ms: int = Field(default=30000, ge=1000)
    # energy only; NORMALIZED rms in 0..1 (NOT int16 amplitude): ~0.01-0.05 is a
    # speech-level threshold, >0.71 is unreachable by real audio; 0 => adaptive noise floor
    energy_threshold: float = Field(default=0.0, ge=0.0, lt=1.0)
    aggressiveness: int = Field(default=2, ge=0, le=3)    # webrtc only
    firered: FireRedVadConfig = Field(default_factory=FireRedVadConfig)
    turn: TurnConfig = Field(default_factory=TurnConfig)

    @model_validator(mode="after")
    def _hangover_bounds(self) -> VadConfig:
        if self.hangover_min_ms is not None and self.hangover_min_ms >= self.hangover_ms:
            raise ValueError(
                f"vad.hangoverMinMs ({self.hangover_min_ms}) must be below "
                f"vad.hangoverMs ({self.hangover_ms}); it is the adaptive floor, "
                "hangoverMs the ceiling"
            )
        return self


class SenseVoiceSttConfig(OnDeviceRuntime):
    """On-device SenseVoice-Small (``stt.provider="sensevoice"``): non-autoregressive CTC,
    one model for zh/yue/en/ja/ko, ~5-15x faster than Whisper on CPU. Use the sherpa-onnx
    export: its ONNX metadata carries the whole front-end contract (CMVN/LFR/language ids),
    so only the model and ``tokens.txt`` are supplied by path. int8 recommended (faster AND
    smaller for ASR; REPORT-asr-tts-model-survey.md section 6.1)."""

    model_path: str | None = None
    tokens_path: str | None = None
    language: str = "auto"  # auto/zh/en/ja/ko/yue, validated against the model metadata
    use_itn: bool = True    # inverse text normalization: numbers/punctuation written out


class ZipformerSttConfig(OnDeviceRuntime):
    """On-device streaming Zipformer transducer (``stt.provider="zipformer"``).

    The STREAMING engine: audio is decoded during speech, so the transcript is ready
    ~immediately at the endpoint. One model per language (pair); use a sherpa-onnx export:
    chunk geometry and decoder context come from the model metadata. Requires 16 kHz
    capture. int8 recommended."""

    encoder_path: str | None = None
    decoder_path: str | None = None
    joiner_path: str | None = None
    tokens_path: str | None = None


class WhisperSttConfig(OnDeviceRuntime):
    """On-device Whisper encoder+decoder (``stt.provider="whisper"``), following Rockchip's
    rknn_model_zoo Whisper demo export; models + assets supplied by path. ".rknn" runs on
    the NPU, ".onnx" on CPU/GPU."""

    encoder_path: str | None = None
    decoder_path: str | None = None
    # Preferably an ORIGINAL artifact: multilingual.tiktoken (openai/whisper assets) or
    # vocab.json (HF export); a flat "<id> <token>" byte-level file also works. Detokenization
    # is always the byte-level BPE inverse map; the legacy per-subword base64 re-encoding
    # (vocab_zh.txt) is rejected.
    vocab_path: str | None = None
    mel_filters_path: str | None = None   # flat 80x201 mel filterbank text file
    # PREFERRED language, one ISO code: sets the decoder prompt's language token and is the
    # fallback when detection is off or unconfident. The export is the multilingual base
    # (all 99 language tokens), so no per-language re-export is needed.
    language: str = "en"
    # DECODABLE languages, e.g. ["en", "zh", "ja"]: bounds what may be OUTPUT (tokens in no
    # enabled script are re-picked) AND the candidate set for per-utterance language ID;
    # see stt/whisper.py. None or one entry => both off, fixed to `language`, which should
    # itself be a member (else the first entry wins, with a warning).
    languages: list[str] | None = None
    # Min softmax confidence (over the enabled set only) to accept a detection; 0 = always
    # take the top candidate. Raise on a quantized (.rknn int8) model, where a 99-way logit
    # comparison degrades faster than greedy decoding does.
    language_min_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    chunk_length: int = Field(default=20, ge=1)  # seconds; MUST match the exported model


class SttServeConfig(_VoiceBase):
    """Serve the loaded on-device STT adapter as a local OpenAI-compatible
    ``POST /v1/audio/transcriptions`` endpoint, so nanobot core's own transcription
    consumers (WebUI mic dictation, channel voice notes) run on THIS box: point core's
    ``transcription.provider`` at an OpenAI-shaped provider entry whose ``apiBase`` is this
    endpoint (use one you don't chat through, e.g. ``providers.siliconflow``). Shares the
    pipeline's SINGLE adapter instance: models are never loaded twice.
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8035, ge=0, le=65535)  # 0 = ephemeral (tests)
    # Requires "Authorization: Bearer <apiKey>"; core sends the provider entry's apiKey, so
    # give both sides the same value.
    api_key: str | None = None
    max_upload_mb: int = Field(default=32, ge=1, le=256)

    @model_validator(mode="after")
    def _non_loopback_needs_a_key(self) -> SttServeConfig:
        if (
            self.enabled
            and self.host not in ("127.0.0.1", "::1", "localhost")
            and not self.api_key
        ):
            raise ValueError(
                f"stt.serve.host={self.host!r} exposes the endpoint beyond loopback: "
                "set stt.serve.apiKey (any local process could otherwise occupy the "
                "decoder, and 0.0.0.0 means the whole network can)"
            )
        return self


class SttConfig(_VoiceBase):
    """Speech-to-text selection. ``nanobot`` delegates to nanobot's top-level
    ``transcription`` config (cloud, or a local OpenAI-compatible Whisper server) with no
    extra deps; the on-device engines each read their own block."""

    provider: Literal["nanobot", "whisper", "sensevoice", "zipformer"] = "nanobot"
    whisper: WhisperSttConfig = Field(default_factory=WhisperSttConfig)
    sensevoice: SenseVoiceSttConfig = Field(default_factory=SenseVoiceSttConfig)
    zipformer: ZipformerSttConfig = Field(default_factory=ZipformerSttConfig)
    serve: SttServeConfig = Field(default_factory=SttServeConfig)
    # Eager (speculative) STT: decode the utterance-so-far once this much trailing silence has
    # accumulated instead of waiting the full vad.hangoverMs. That gap is silence by definition,
    # so the speculative transcript is exactly valid when the endpoint confirms: pure overlap,
    # zero accuracy cost. ON-DEVICE batch STT only: the cloud delegate would waste billed calls
    # whenever the speaker resumes, and zipformer already decodes live. 0 disables; >=
    # vad.hangoverMs never fires.
    eager_ms: int = Field(default=240, ge=0)

    @model_validator(mode="after")
    def _serve_needs_an_ondevice_engine(self) -> SttConfig:
        if self.serve.enabled and self.provider == "nanobot":
            raise ValueError(
                "stt.serve needs an on-device engine (whisper/sensevoice/zipformer): "
                "stt.provider='nanobot' delegates to core transcription, and serving "
                "that back to core would be circular"
            )
        return self


class SupertonicTtsConfig(OnDeviceRuntime):
    """On-device Supertonic-3 (``tts.provider="supertonic"``): flow-matching TTS, 31
    languages in ONE model (en/ko/ja/de/fr/es/..., NO zh), 44.1 kHz out, ~99M params.

    Artifacts come straight from the ORIGINAL release (huggingface.co/Supertone/supertonic-3;
    OpenRAIL-M weights (use restrictions) with MIT inference code): four ``onnx/*.onnx``
    graphs plus ``onnx/tts.json``, ``onnx/unicode_indexer.json`` and one ``voice_styles/*.json``.
    Nothing is repackaged, so the same fp32 graphs also source int8/RKNN conversion. The text
    front-end is a per-codepoint unicode lookup (no G2P, no tokenizer); language is selected by
    wrapping the text in ``<lang>...</lang>`` tags."""

    text_encoder_path: str | None = None
    duration_predictor_path: str | None = None
    vector_estimator_path: str | None = None
    vocoder_path: str | None = None
    tts_json_path: str | None = None          # onnx/tts.json (rates + latent geometry)
    unicode_indexer_path: str | None = None   # onnx/unicode_indexer.json (65536 codepoint->id)
    voice_style_path: str | None = None       # voice_styles/<name>.json (one voice)
    language: str = "en"                      # one of the 31, validated at load; sets the text tag
    num_steps: int = Field(default=5, ge=1)   # flow-matching steps: quality vs latency
    speed: float = Field(default=1.05, gt=0)  # reference default; >1 = faster speech
    # Per-piece text budget in codepoints; 0 = the reference defaults (120 for ko/ja, 300
    # otherwise). Longer chunks are split at space/clause boundaries.
    max_len: int = Field(default=0, ge=0)


class MmsTtsConfig(OnDeviceRuntime):
    """On-device MMS-TTS / VITS (``tts.provider="mms"``), the rknn_model_zoo demo export.
    Models supplied by path."""

    encoder_path: str | None = None  # ".rknn" (NPU) or ".onnx" (CPU)
    decoder_path: str | None = None
    # Per-language MMS char vocab (HF ``vocab.json`` = ``{char: id}``); None => built-in English
    # (mms-tts-eng). Each language is its own model + vocab.json (e.g. mms-tts-deu for German).
    vocab_path: str | None = None
    # Runs before char-tokenisation. "none" for Latin (en/de); "uroman" romanizes non-Latin
    # scripts ([uroman] extra); "japanese" is kanji-aware (pyopenjtalk -> uroman, [japanese]
    # extra) for mms-tts-jpn.
    text_frontend: Literal["none", "uroman", "japanese"] = "none"
    max_length: int = Field(default=200, ge=1)  # encoder input length; MUST match the exported model
    speaking_rate: float = Field(default=1.0, gt=0)  # >1 = faster (shorter durations)


class TtsConfig(_VoiceBase):
    """Text-to-speech. The default provider speaks OpenAI-compatible ``/audio/speech`` over
    httpx, driving cloud OR any local server (Kokoro-FastAPI, piper-http, ...) by changing
    ``apiBase``: local neural TTS with no decoder here. ``mms``/``supertonic`` are the
    on-device ONNX/RKNN engines; ``system`` is the zero-dep espeak-ng/say fallback."""

    enabled: bool = True
    provider: Literal["openai", "openai_compat", "system", "mms", "supertonic"] = "openai"
    mms: MmsTtsConfig = Field(default_factory=MmsTtsConfig)
    supertonic: SupertonicTtsConfig = Field(default_factory=SupertonicTtsConfig)
    model: str = "gpt-4o-mini-tts"
    voice: str = "alloy"
    api_base: str | None = None    # e.g. http://localhost:8880/v1 for a local server
    api_key: str | None = None     # falls back to OPENAI_API_KEY env
    # wav => decoder-free aplay; pcm => gapless stream-mode playback. mp3 is deliberately NOT
    # accepted: nothing downstream decodes it, so it would leave the channel permanently silent.
    audio_format: Literal["wav", "pcm"] = "wav"
    # Rate of the server's RAW-PCM responses (audioFormat=pcm only). OpenAI/Kokoro emit
    # 24 kHz; a server emitting 22.05/44.1 kHz needs this or playback is pitch-shifted.
    pcm_sample_rate: int = Field(default=24000, gt=0)
    # The language this TTS speaks, ONE ISO 639-1 code. Two consumers: the system
    # provider's espeak/say voice, and, for engines that cannot know their own language
    # (an openai_compat server, MMS with a custom vocabPath), the declaration behind the
    # agent's "reply in this language" context line. Engines that DO know (supertonic,
    # built-in-vocab MMS) ignore it with a warning on conflict; see make_tts.
    language: str | None = None
    # openai/openai_compat PER-ATTEMPT cap; the retry ladder is bounded at 2x this, so a
    # wedged server cannot hold the turn in SPEAKING.
    timeout_s: float = Field(default=60.0, gt=0)


class ChunkerConfig(_VoiceBase):
    """How streamed reply text is split into speakable units for low latency."""

    min_chars: int = Field(default=60, ge=1)   # soft floor before flushing the first clause
    max_chars: int = Field(default=240, ge=1)  # hard cap that force-flushes run-ons
    # FIRST chunk of each turn only (clamped to min_chars): TTS cannot start until chunk 1
    # exists, so a small one cuts time-to-first-audio; later chunks keep min_chars prosody.
    min_chars_first: int = Field(default=24, ge=1)


class PerfConfig(_VoiceBase):
    """Device-performance adaptation (local mode): tune pacing WITHIN the chosen engine set
    from measurements taken on the device at session start. Invariants
    (DESIGN-local-latency-and-engines.md Part E): only PACING values are derived, never a
    behavior change; an explicit config value always wins; every derived value is logged at
    INFO with its measurement."""

    calibrate: bool = True


class PrologueConfig(_VoiceBase):
    """Filler audio while the agent is still working (local backend only; cloud modes get
    model-spoken filler instead). A neutral phrase, synthesized once with the session's own
    TTS voice and cached, plays after ``afterMs`` of THINKING with no reply, then every
    ``intervalMs``. Fillers are epoch-stamped audio, so every barge-in path kills them like
    real speech; keep phrases short (the half-duplex mic is gated while one plays)."""

    enabled: bool = False
    after_ms: int = Field(default=2000, ge=0)        # THINKING silence before filler 1
    interval_ms: int = Field(default=8000, ge=1000)  # re-filler cadence during long waits
    phrases: list[str] = Field(
        default_factory=lambda: ["One moment.", "Still working on it."]
    )


class RealtimeConfig(_VoiceBase):
    """Shared settings for the OpenAI-Realtime **dialect family** of e2e speech-to-speech
    backends (``backend`` = openai / xai / azure / qwen / glm / stepfun).

    Cloud-only. The provider does ASR + reasoning + TTS in one WebSocket session; the plugin
    owns local mic capture + speaker playback and routes tool calls to nanobot where the
    provider supports it (``DESIGN-realtime-providers.md``). Audio rate is provider-fixed
    (24 kHz mostly, 16 kHz input for Qwen/GLM) and comes from the profile, NOT from
    ``AudioConfig.sample_rate``.
    """

    # None => the provider profile's default. Override for a pinned model, a self-hosted /
    # regional endpoint (Azure resource URL, DashScope -intl), or a provider voice.
    model: str | None = None
    base_url: str | None = None           # ``?model=`` appended at connect (GA/beta)
    api_key: str | None = None            # falls back to OPENAI_API_KEY
    voice: str | None = None              # provider voice (e.g. OpenAI cedar/marin, Qwen Chelsie)
    # Replaces the built-in persona (style/identity) ONLY. The tool rules for the resolved
    # toolMode (the filler preamble, and supervisor's ask_nanobot delegation contract)
    # are appended by the channel and cannot be overridden: they are the wire contract the
    # backend depends on, and a persona that dropped them would leave supervisor mode
    # answering from the small realtime model alone. Say nothing about tools here.
    persona: str | None = None

    # A realtime model is small and non-reasoning, so it plans multi-step tool sequences badly
    # (REPORT-realtime-reasoning-latency.md).
    #   "direct"     = declare nanobot's tools to it; it plans and calls them itself. Fine for
    #                  light, single-step tool use.
    #   "supervisor" = declare ONE tool (ask_nanobot): it owns the conversation but delegates
    #                  all reasoning/tool work to nanobot's AgentLoop and speaks the finished
    #                  answer (Responder-Thinker). Robust multi-step tool use, costing a ~1-2s
    #                  delegation the mandatory filler masks. Tool-capable providers only.
    tool_mode: Literal["direct", "supervisor"] = "direct"
    server_vad: bool = True               # server-side turn detection + native barge-in
    interrupt_response: bool = True        # server auto-cancels the response on user speech (GA)
    # None => input transcription off => the cloud backend emits no InputTranscript.
    input_transcription_model: str | None = None

    # An open mic without echo cancellation feeds our own TTS back up the uplink and self-cancels
    # every turn, so "aec" requires one of ``aec_available``, top-level ``aec="hardware"`` or
    # ``aec="webrtc"``, enforced at start() (channel.py). "gated" needs no such assertion: the
    # uplink is suspended while speaking (the shell's SPEAKING mic-gate), resuming after drain.
    barge_in: Literal["aec", "gated"] = "gated"
    aec_available: bool = False           # assert hardware/OS AEC so open-mic is safe
    turn_timeout_s: float = Field(default=30.0, gt=0)  # watchdog for a missing response.done
    # Budget for ONE ask_nanobot delegation (supervisor mode); None -> turn_timeout_s. A tool-heavy
    # agent turn routinely runs past 30 s; raise THIS, not the wire watchdog, which stays tight.
    delegation_timeout_s: float | None = Field(default=None, gt=0)


class TelemetryConfig(_VoiceBase):
    """OpenTelemetry export of TOOL CALLS (opt-in, [otel] extra, lazily imported).

    Tool calls follow the OTel GenAI semantic conventions (``execute_tool {name}`` spans with
    ``gen_ai.tool.*``), portable across Langfuse / Phoenix / Braintrust / LangSmith. Voice latency
    has NO convention: the semconv covers no audio, realtime, endpointing or barge-in signal
    (semconv issue #304 is open), so it stays in ``VoiceMetrics`` in-process and is not
    exported. See ``REPORT-eval-methodology.md`` section 3.7.
    """

    enabled: bool = False
    namespace: str = "voice"  # prefix for the non-semconv tool attrs (outcome/mode/stale)
    # Opt-In in the semconv BECAUSE THEY ARE USER DATA: a voice tool call carries whatever the
    # user just said. Leave off unless the collector is trusted; names, timings and outcomes flow
    # regardless.
    capture_content: bool = False


class BargeInConfig(_VoiceBase):
    """Confirm-stage policy for the local open-mic modes (duck-then-confirm). The duck depth
    itself is the top-level ``duckDb``.
    """

    # What the confirm window does to live playback. "duck": attenuate by duckDb and keep
    # playing (a false alarm is barely audible). "pause": stop the stream entirely and
    # resume where it left off on a false verdict: the bot's leak vanishes from the mic
    # during the user's utterance, which is exactly the soft-duplex failure mode; the
    # cost is a silent gap of the confirm window's length when the alarm was false.
    # Stream-mode open-mic only; blob mode keeps the static duckDb bake.
    mode: Literal["duck", "pause"] = "duck"
    # Early-confirm word bar: a decode holding this many words that are neither our own TTS
    # (echo) nor ack phrases confirms the interrupt before the endpoint verdict: streaming
    # partials (zipformer) check it live, batch engines at their eager decode. Also the bar a
    # transcript classified as self-echo must clear to still count as an interruption.
    min_words: int = Field(default=2, ge=1)
    # Duck on SUSPICION: consecutive VAD speech frames (x audio.frameMs) while the bot speaks before
    # the reversible duck engages, deliberately below vad.startFrames, because a false dip is cheap
    # (attack 30 ms, release 250 ms) while every frame of delay is the bot talking over the user at
    # full volume; a run that dies before onset just releases. >= vad.startFrames = duck at onset.
    duck_start_frames: int = Field(default=2, ge=1)
    # On a confirmed barge-in (stream-mode TTS), append a bracketed note to the published
    # utterance saying how much of the cancelled reply the user actually heard: the in-channel
    # stand-in for history truncation (core keeps the full text; a channel cannot edit it).
    heard_marker: bool = True
    # Backchannels: a transcript made ENTIRELY of these (lower-cased word tokens) while the bot
    # works/speaks is an acknowledgement, not an interruption; the reply survives, duck releases.
    ack_phrases: list[str] = Field(
        default=[
            "ok", "okay", "yeah", "yes", "yep", "uh-huh", "mm-hmm", "right",
            "got it", "i see", "sure",
            "嗯", "好", "好的", "对", "是",
            "はい", "うん", "ええ", "そう",
        ]
    )


class VoiceConfig(_VoiceBase):
    """Top-level ``channels.voice`` config.

    ``enabled``, ``allowFrom`` and ``streaming`` have no in-repo reader: nanobot core
    consumes them. Do not remove them as dead.
    """

    enabled: bool = False

    # Reasoning brain / supplier. "local" = on-box VAD/STT/TTS + nanobot over the text bus (full
    # brain, no cloud). The rest are e2e speech-to-speech over a provider's OpenAI-Realtime-dialect
    # WebSocket ([realtime] extra; "qwen" is Alibaba, "glm" Zhipu) sharing the ``realtime.*`` block;
    # their model/endpoint/rates come from backend/profiles.py. See DESIGN-realtime-providers.md.
    backend: Literal[
        "local", "openai", "xai", "azure", "qwen", "glm", "stepfun"
    ] = "local"
    allow_from: list[str] = Field(default_factory=lambda: ["*"])  # BaseChannel allow-list
    streaming: bool = True  # core `supports_streaming`: send_delta() speaks the reply as it streams
    sender_id: str = "local"
    chat_id: str = "voice:local"

    # Duplex mode:
    #   "auto"     => half-duplex: mic suspended while speaking (the safe default).
    #   "hardware" => full-duplex: mic stays open; asserts hardware/OS echo cancellation.
    #   "soft"     => open mic WITHOUT AEC; the bot's own voice is dropped by matching the
    #                 known TTS text (self-echo rejection), so you can barge in by
    #                 talking. Best-effort; pair with duckDb for SNR.
    #   "webrtc"   => SOFTWARE echo cancellation ([aec] extra, WebRTC AEC3): full-duplex
    #                 barge-in with no hardware AEC, our own playback being the reference
    #                 subtracted from the mic before VAD/STT. Degrades to "soft" if the extra
    #                 is missing, or for WAV-blob TTS, which supplies no playout-timed
    #                 reference (raw-PCM stream-mode does). Also satisfies the CLOUD open-mic
    #                 requirement (realtime.bargeIn="aec"): it sits in front of the uplink.
    aec: Literal["auto", "soft", "webrtc", "hardware"] = "auto"

    # Self-echo rejection (every duplex mode): drop a transcript whose words are at least this
    # fraction contained in the recently-spoken TTS: it's the bot hearing itself. Half-duplex
    # included: its mic gate is a read-time approximation, and what leaks past a mistimed
    # hangover or capture lag must not publish as a user turn.
    echo_reject_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    # Include user speech in gateway logs. OFF because transcripts are personal data and logs
    # get shipped/persisted; operational lines still fire, with word counts instead of content.
    log_transcripts: bool = False
    # Open-mic duck depth in dB (0 = off; -12 ~ a quarter as loud). LOCAL backend only: the cloud
    # path interrupts server-side (truncate/cancel) and never ducks. Stream-mode playback ducks
    # DYNAMICALLY: fast-attack to this floor the moment speech is detected while the bot speaks,
    # released if the transcript verdict says it wasn't a real interruption. Blob-mode (WAV-only
    # TTS) can't change gain mid-chunk and bakes it in statically.
    duck_db: float = Field(default=-12.0, le=0.0)
    barge_in: BargeInConfig = Field(default_factory=BargeInConfig)

    # Tail after playback drains before re-listening (local only; the cloud drain has no hangover).
    playback_hangover_ms: int = Field(default=250, ge=0)

    # Stalled-agent deadman (local backend; cloud analog: realtime.turnTimeoutS): a published turn
    # with NO activity (no delta, no segment end) for this long gets a short spoken notice + /stop
    # instead of silent THINKING forever. Activity resets the clock, so long tool runs survive as
    # long as the agent streams its pre-tool status line. None disables.
    agent_timeout_s: float | None = Field(default=120.0, gt=0)
    timeout_phrase: str = "Sorry, I'm having trouble answering that. Please try again."

    # WebUI full-surface escape hatch (the manifest's secret-kind "importJson" box): the
    # WHOLE channels.voice section as one pasted JSON object, deep-merged over the section
    # at parse time (paste wins) and expanded into real config.json keys at channel start
    # by consume_import_json(), which deletes the blob. A TRANSPORT, never a durable second
    # copy; secret-kind means core never echoes it back to a browser.
    import_json: str | None = None

    # LOCAL backend: operator text appended to the voice runtime-context block that rides
    # every published utterance: the voice-scoped seam for guidance that must not leak to
    # other channels (a SOUL.md/AGENTS.md directive applies everywhere; THIS is spoken
    # turns only). Core marks runtime context "metadata only, not instructions": a nudge,
    # not a persona override; nanobot owns the local persona, cloud ones go in
    # realtime.persona.
    context: str | None = None

    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    stt: SttConfig = Field(default_factory=SttConfig)
    tts: TtsConfig = Field(default_factory=TtsConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    prologue: PrologueConfig = Field(default_factory=PrologueConfig)
    perf: PerfConfig = Field(default_factory=PerfConfig)
    realtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)

    @model_validator(mode="before")
    @classmethod
    def _merge_import_json(cls, data: Any) -> Any:
        """Fold a pending ``importJson`` paste into the section it configures. Runs in
        any order relative to ``_fold_alias_twins``: it reads both spellings itself and
        leaves exactly one behind, so the fold never sees a conflict it must guess at.
        The raw blob is kept on the field so ``start()`` knows to consume it."""
        if not isinstance(data, dict):
            return data
        raw = next(
            (data[k] for k in ("importJson", "import_json") if not _emptyish(data.get(k))),
            None,
        )
        if raw is None:
            if "importJson" in data or "import_json" in data:
                # the enable-toggle's materialized '' filler (or an emptied box)
                data = {k: v for k, v in data.items() if k not in ("importJson", "import_json")}
            return data
        data = {k: v for k, v in data.items() if k not in ("importJson", "import_json")}
        merged = _apply_import(data, parse_import_blob(raw))
        merged["importJson"] = raw
        return merged

    @model_validator(mode="after")
    def _vad_engine_supports_rate(self) -> VoiceConfig:
        """An EXPLICITLY configured neural VAD must run at the configured capture rate: statically
        knowable, and the runtime alternative is a silent downgrade to the energy fallback."""
        rate = self.audio.sample_rate
        supported = {"firered": (16000,), "webrtc": (8000, 16000, 32000, 48000)}.get(self.vad.engine)
        if supported is not None and rate not in supported:
            raise ValueError(
                f"vad.engine='{self.vad.engine}' cannot run at audio.sampleRate={rate} "
                f"(supported: {', '.join(map(str, supported))}); change the rate or the engine"
            )
        return self

    @property
    def full_duplex(self) -> bool:
        """True hardware/OS-AEC full-duplex. ``auto`` is conservative -> half-duplex."""
        return self.aec == "hardware"

    @property
    def soft_duplex(self) -> bool:
        """Open-mic barge-in WITHOUT AEC, guarded by self-echo rejection + ducking."""
        return self.aec == "soft"

    @property
    def open_mic(self) -> bool:
        """Mic stays open while the bot speaks (any duplex mode, incl. software AEC). The ONE
        derivation: the shell's mic gate and the local backend's echo-filter/duck wiring both
        read it here, so they cannot drift apart."""
        return self.full_duplex or self.soft_duplex or self.aec == "webrtc"


def consume_import_json(config_path: Path | None = None) -> int:
    """Expand a pending ``channels.voice.importJson`` paste in ``config.json`` into the
    real section keys and delete the blob; returns how many top-level keys the paste
    carried (0 = nothing pending). Read-modify-write on the FILE, not core's in-memory
    ``Config``: core's own writers load fresh per operation, so this is the same
    contract as a hand edit landing between two WebUI saves. Only the voice section is
    touched; the rest of the document is re-serialized in core's own format
    (indent=2, ensure_ascii=False).
    """
    if config_path is None:
        from nanobot.config.loader import get_config_path

        config_path = get_config_path()
    try:
        document = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError:
        return 0  # embedded/test configs that never touched a file
    channels = document.get("channels") if isinstance(document, dict) else None
    section = channels.get("voice") if isinstance(channels, dict) else None
    if not isinstance(section, dict):
        return 0
    raw = next(
        (section[k] for k in ("importJson", "import_json") if not _emptyish(section.get(k))),
        None,
    )
    if raw is None:
        return 0  # emptyish '' fillers stay put, like every other materialized secret
    blob = parse_import_blob(raw)
    base = {k: v for k, v in section.items() if k not in ("importJson", "import_json")}
    channels["voice"] = _apply_import(base, blob)
    _write_json_atomic(config_path, document)
    return len(blob)


def _write_json_atomic(path: Path, document: Any) -> None:
    """Same shape as core's config writer: temp file in place, mode preserved
    (config.json holds secrets), fsync, atomic replace."""
    import stat
    from contextlib import suppress

    tmp = path.with_name(f".{path.name}.{os.getpid()}.import.tmp")
    mode: int | None = None
    with suppress(OSError):
        mode = stat.S_IMODE(path.stat().st_mode)
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            if mode is not None:
                os.chmod(tmp, mode)
            fh.write(json.dumps(document, indent=2, ensure_ascii=False))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)
    finally:
        with suppress(OSError):
            tmp.unlink()
