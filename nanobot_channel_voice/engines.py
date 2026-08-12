"""Shared machinery for the engine registries (stt / tts / vad).

Each registry keeps its own fallback POLICY (delegate to nanobot, degrade to the
system voice, fall back to energy VAD) and wording; here lives the shape they
share: the spec table entry, dotted-attribute resolution into nested config
blocks, the required-field check, and turning a build failure into an actionable
message: a missing optional dependency names its pip extra instead of a bare
``No module named 'kaldi_native_fbank'``.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EngineSpec:
    """One selectable engine: required config fields (dotted attr path +
    camelCase name for the warning), the importable modules it needs beyond the
    hard dependencies (probed by :func:`preflight`, never imported here), and a
    lazily-importing factory. The factory signature is registry-specific
    (stt/tts pass ``cfg``; vad adds the rates)."""

    build: Callable[..., Any]
    required: tuple[tuple[str, str], ...] = ()
    modules: tuple[str, ...] = ()


def resolve_attr(cfg: Any, dotted: str) -> Any:
    """Fetch a possibly-nested config attr (``"sensevoice.model_path"``)."""
    obj = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def missing_fields(cfg: Any, spec: EngineSpec) -> list[str]:
    """The camelCase names of the spec's required fields that are unset."""
    return [name for attr, name in spec.required if not resolve_attr(cfg, attr)]


# Top-level module name of a failed import -> the pyproject extra providing it.
# Only imports surfacing through make_audio/make_stt/make_tts/make_vad reach here;
# realtime/aec/otel/text-frontend catch ImportError themselves with more context.
_EXTRA_BY_MODULE = {
    "numpy": "ondevice",
    "onnxruntime": "ondevice",
    "kaldi_native_fbank": "ondevice",
    "webrtcvad": "webrtc",
    "alsaaudio": "pyalsa",
    # A .rknn load with NEITHER runtime installed surfaces as the LAST import attempt's
    # module ("rknn", the full-toolkit fallback); "rknnlite" covers any future direct
    # import. The extra only resolves on aarch64 Linux <= py3.12 (see pyproject);
    # elsewhere the fix is a .onnx artifact, but naming the extra still points at the doc.
    "rknn": "rknn",
    "rknnlite": "rknn",
}


def describe_build_error(exc: BaseException) -> str:
    """``str(exc)``, plus the install hint when it is a missing optional extra."""
    missing = exc.name if isinstance(exc, ModuleNotFoundError) else None
    extra = _EXTRA_BY_MODULE.get((missing or "").partition(".")[0])
    if extra:
        return f"{exc}; pip install 'nanobot-channel-voice[{extra}]'"
    return str(exc)


def preflight(cfg: Any, engine: str, table: dict[str, EngineSpec], *, prefix: str = "") -> str | None:
    """Validate-time mirror of a registry's fallback triggers: the static reason
    the selected engine would degrade at start: an unresolvable ``weights``
    store key, unset required fields, or a missing optional dependency; else
    None. Engines outside the table (the zero-config defaults) are always None.
    Nothing is imported or loaded; runtime construction can still fail.
    ``prefix`` (e.g. ``"vad."``) turns the block-relative field names into full
    config keys for user-facing messages."""
    spec = table.get(engine)
    if spec is None:
        return None
    from nanobot_channel_voice.weights import WeightsError, apply_weights

    try:
        cfg = apply_weights(cfg, engine)
    except WeightsError as exc:
        return str(exc)
    missing = missing_fields(cfg, spec)
    if missing:
        return "unset: " + ", ".join(prefix + name for name in missing)
    for module in spec.modules:
        if importlib.util.find_spec(module) is None:
            extra = _EXTRA_BY_MODULE.get(module)
            hint = f"; pip install 'nanobot-channel-voice[{extra}]'" if extra else ""
            return f"missing module '{module}'{hint}"
    return None
