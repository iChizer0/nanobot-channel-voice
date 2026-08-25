"""Provider profiles for the OpenAI-Realtime **dialect family**.

One adapter (:mod:`.openai_realtime`) covers every vendor speaking the protocol, reading
a pure-data :class:`RealtimeProfile` for what differs. Providers whose wire protocol is
NOT this dialect (Gemini Live, AWS Nova Sonic, ...) are separate backends, not profiles.
Every ``default_*`` is overridable via ``realtime.{baseUrl,model,voice}``.

``dialect`` drives only what we SEND: ``"ga"`` nests ``session.audio.*``, ``"beta"`` is
flat; the receive path matches both delta names unconditionally (they never collide).
Capability flags stay conservative: off where a vendor's tool flow is unvalidated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeVar

_V = TypeVar("_V")

Dialect = Literal["ga", "beta"]

# How a barge-in stops the in-flight response; every vendor needs SOMETHING sent.
#   "truncate" = GA: conversation.item.truncate aligns the model's memory with the audio
#                actually HEARD. Server-VAD auto-cancel (``interrupt_response``, on by
#                default) stops the response, so cancel only when interruptResponse is off.
#   "cancel"   = beta: no truncate event AND no auto-cancel, so the client must cancel.
#                The server keeps what it streamed: no playback-aligned memory here.
InterruptKind = Literal["truncate", "cancel"]

# What a ``channels.voice.backend`` value selects; the channel dispatches on this.
BackendKind = Literal["local", "openai_dialect", "gemini"]


@dataclass(frozen=True, slots=True)
class RealtimeProfile:
    """Pure data describing one OpenAI-dialect realtime supplier."""

    key: str
    dialect: Dialect
    default_model: str
    default_voice: str
    input_rate: int   # mic capture rate fed to the API, S16_LE mono; the channel
                      # builds capture at this rate, never at a hardcoded constant
    output_rate: int  # rate of the PCM the API streams back (~always 24k)
    # beta only: three mutually incompatible per-vendor vocabularies (see the table). GA
    # ignores both and sends {type: "audio/pcm", rate} built from *_rate instead.
    input_format: str = "pcm16"
    output_format: str = "pcm16"
    default_base_url: str | None = None  # None => realtime.baseUrl is REQUIRED (Azure)
    auth_header: str = "Authorization"   # the header the api-key/bearer goes in
    bearer_prefix: str = "Bearer "       # "" for a raw api-key header
    extra_headers: tuple[tuple[str, str], ...] = ()
    model_in_query: bool = True          # append ?model=<model> at connect
    # GA: voice at the session root instead of session.audio.output.voice (xAI).
    voice_in_session_root: bool = False
    interrupt: InterruptKind = "truncate"
    supports_tools: bool = True
    needs_response_create_after_tools: bool = True
    # Flatten tool JSON Schemas (nullable unions, anyOf/allOf/oneOf) for providers that
    # reject them (Qwen-Omni). OFF by default: the GA family keeps full schemas.
    flatten_tool_schema: bool = False
    # Max chars in one function_call_output: the whole output enters the model context, so
    # large payloads (base64 images) blow the window. 0 = unlimited (provider enforces).
    max_tool_output_chars: int = 0
    # Per-model voice overrides: keys match by startswith on the resolved model, longest
    # prefix wins; realtime.voice still overrides the result.
    model_voice_overrides: dict[str, str] = field(default_factory=dict)
    # Per-model capability overrides, matched the same way (qwen3 is persona-only,
    # qwen3.5 has tools).
    model_capability_overrides: dict[str, dict[str, bool | int]] = field(default_factory=dict)
    # Vendor-required session keys outside the common schema, merged into the beta session.
    session_extras: dict = field(default_factory=dict)

    @staticmethod
    def _longest_prefix_match(mapping: dict[str, _V], model: str | None) -> _V | None:
        """Value for the longest key in ``mapping`` that prefixes ``model`` (so
        ``"qwen3.5-..."`` beats ``"qwen3-..."``), or None when nothing matches."""
        best: tuple[str, _V] | None = None
        if model:
            for prefix, value in mapping.items():
                if model.startswith(prefix) and (best is None or len(prefix) > len(best[0])):
                    best = (prefix, value)
        return best[1] if best is not None else None

    def default_voice_for(self, model: str | None = None) -> str:
        """:attr:`default_voice` unless a per-model override matches."""
        voice = self._longest_prefix_match(self.model_voice_overrides, model)
        return voice if voice is not None else self.default_voice

    def capabilities_for(self, model: str | None = None) -> dict[str, bool | int]:
        """Tool capabilities with any per-model override merged on top."""
        caps: dict[str, bool | int] = {
            "supports_tools": self.supports_tools,
            "needs_response_create_after_tools": self.needs_response_create_after_tools,
            "max_tool_output_chars": self.max_tool_output_chars,
        }
        override = self._longest_prefix_match(self.model_capability_overrides, model)
        if override:
            caps.update(override)
        return caps

    def base_url(self, override: str | None) -> str:
        url = override or self.default_base_url
        if not url:
            raise RuntimeError(
                f"realtime provider '{self.key}' has no default endpoint; set "
                "channels.voice.realtime.baseUrl to the vendor's wss:// URL."
            )
        return url

    def connect_url(self, base_override: str | None, model: str) -> str:
        url = self.base_url(base_override)
        if not self.model_in_query:
            return url
        if "model=" in url.partition("?")[2]:
            return url  # a baseUrl that already pins ?model= wins; never send two
        sep = "&" if "?" in url else "?"
        return f"{url}{sep}model={model}"

    def auth_headers(self, api_key: str) -> dict[str, str]:
        headers = {self.auth_header: f"{self.bearer_prefix}{api_key}"}
        headers.update(dict(self.extra_headers))
        return headers


# ---- The preset table -------------------------------------------------------
# default_voice is best-effort; if a provider rejects it, set realtime.voice.

PROFILES: dict[str, RealtimeProfile] = {
    # ---- GA dialect (nested session.audio.*, response.output_audio.delta) ---
    "openai": RealtimeProfile(
        key="openai",
        dialect="ga",
        default_base_url="wss://api.openai.com/v1/realtime",
        default_model="gpt-realtime",
        default_voice="alloy",
        input_rate=24000,
        output_rate=24000,
    ),
    # xAI Grok Voice. Emits input_audio_transcription.updated (CUMULATIVE) instead of
    # the OpenAI *.completed event, so InputTranscript stays silent here.
    "xai": RealtimeProfile(
        key="xai",
        dialect="ga",
        default_base_url="wss://api.x.ai/v1/realtime",
        default_model="grok-voice-latest",
        default_voice="ara",  # xAI voices: ara/eve/leo/rex/sal, or an 8-char custom id
        input_rate=24000,
        output_rate=24000,
        voice_in_session_root=True,  # session.voice, not session.audio.output.voice
    ),
    # Azure OpenAI Realtime (GA endpoint): realtime.baseUrl = your resource URL,
    # realtime.model = the deployment name. Preview (?api-version=) endpoints are NOT
    # covered by this profile.
    "azure": RealtimeProfile(
        key="azure",
        dialect="ga",
        default_base_url=None,  # resource-specific: wss://<res>.openai.azure.com/openai/v1/realtime
        default_model="gpt-realtime",  # = the Azure *deployment* name
        default_voice="alloy",
        input_rate=24000,
        output_rate=24000,
        auth_header="api-key",
        bearer_prefix="",
    ),
    # ---- beta dialect (flat input_audio_format, response.audio.delta) -------
    # Alibaba DashScope. Outside mainland China use the -intl endpoint (workspace-scoped
    # *.maas.aliyuncs.com recommended, legacy domain still works). Per-session turn caps
    # apply: expect clean server closes.
    "qwen": RealtimeProfile(
        key="qwen",
        dialect="beta",
        default_base_url="wss://dashscope.aliyuncs.com/api-ws/v1/realtime",
        default_model="qwen3-omni-flash-realtime",
        default_voice="Chelsie",  # valid for qwen3-omni-flash-realtime
        model_voice_overrides={
            # qwen3.5-omni-flash-realtime drops Chelsie; its documented default is Tina.
            "qwen3.5-omni-flash-realtime": "Tina",
        },
        model_capability_overrides={
            # qwen3.5 adds tool calling and requires an explicit response.create after
            # the function_call_output to produce the final answer.
            "qwen3.5-omni-flash-realtime": {
                "supports_tools": True,
                "needs_response_create_after_tools": True,
                # Big outputs push the realtime context over its limit; the server
                # truncates.
                "max_tool_output_chars": 8000,
            },
        },
        input_rate=16000,   # DashScope Qwen-Omni takes 16k in, streams 24k out
        output_rate=24000,
        # DashScope: "only pcm is supported"; the OpenAI "pcm16" is rejected here.
        input_format="pcm",
        output_format="pcm",
        # NB: no OpenAI-Beta header — Qwen-*ASR*-Realtime needs it and shares the
        # /api-ws/v1/realtime path, but Omni documents Authorization as its only header.
        interrupt="cancel",
        # The default model is PERSONA-ONLY: the base must stay False or it gets tools it
        # can't drive (qwen3.5 re-enables via the override above).
        supports_tools=False,
        needs_response_create_after_tools=True,
        flatten_tool_schema=True,  # Qwen-Omni rejects nullable-union types + combinators
    ),
    # Zhipu GLM-Realtime. The vendor session default is client_vad; the plugin's always-on
    # server VAD overrides it. Voices: tongtong, female-tianmei, male-qn-daxuesheng,
    # male-qn-jingying, lovely_girl, female-shaonv.
    "glm": RealtimeProfile(
        key="glm",
        dialect="beta",
        default_base_url="wss://open.bigmodel.cn/api/paas/v4/realtime",
        default_model="glm-realtime-flash",  # 9B; use glm-realtime-air for 32B
        default_voice="tongtong",
        input_rate=16000,
        output_rate=24000,
        # ASYMMETRIC vocabularies: input takes wav/pcm16/pcm24 where the suffix is the
        # SAMPLE RATE, not the bit depth ("pcm16" = 16 kHz, matching input_rate); output
        # takes only "pcm", its 24 kHz implicit.
        input_format="pcm16",
        output_format="pcm",
        interrupt="cancel",
        supports_tools=True,      # results via conversation.item.create + response.create
        needs_response_create_after_tools=True,
        # GLM requires beta_fields.chat_mode in session.update (default audio).
        session_extras={"beta_fields": {"chat_mode": "audio"}},
    ),
    # StepFun step-audio realtime. Emits an extra response.thinking.delta stream
    # (ignored).
    "stepfun": RealtimeProfile(
        key="stepfun",
        dialect="beta",
        default_base_url="wss://api.stepfun.com/v1/realtime",
        default_model="step-audio-2-mini",
        default_voice="linjiajiejie",  # voice is a REQUIRED param for StepFun
        input_rate=24000,   # UNVERIFIED: StepFun documents no sample rate; 24k assumed
        output_rate=24000,  # UNVERIFIED (as above)
        input_format="pcm16",   # StepFun is the one beta provider using the OpenAI value
        output_format="pcm16",
        interrupt="cancel",
        supports_tools=True,      # console is a fork of the OpenAI beta SDK
        needs_response_create_after_tools=True,
    ),
}

OPENAI_DIALECT_PROVIDERS = frozenset(PROFILES)


def normalize_backend(name: str) -> str:
    """Normalize a ``channels.voice.backend`` value (strip + lowercase)."""
    return (name or "").strip().lower()


def backend_kind(name: str) -> BackendKind:
    key = normalize_backend(name)
    if key in OPENAI_DIALECT_PROVIDERS:
        return "openai_dialect"
    if key == "gemini":
        return "gemini"
    return "local"


def resolve_profile(name: str) -> RealtimeProfile:
    """Raises if ``name`` is not an OpenAI-dialect provider: the caller should have
    dispatched on :func:`backend_kind` first."""
    key = normalize_backend(name)
    try:
        return PROFILES[key]
    except KeyError:
        raise RuntimeError(
            f"'{name}' is not an OpenAI-dialect realtime provider; expected one of "
            f"{', '.join(sorted(OPENAI_DIALECT_PROVIDERS))}."
        ) from None


__all__ = [
    "RealtimeProfile",
    "PROFILES",
    "OPENAI_DIALECT_PROVIDERS",
    "backend_kind",
    "normalize_backend",
    "resolve_profile",
]
