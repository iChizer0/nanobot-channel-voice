"""Shared machinery for the engine registries (stt / tts / vad).

Each registry keeps its own fallback POLICY and wording; here lives the shape they
share: the spec table entry, dotted-attribute resolution, the required-field check, and
turning a build failure into a message that names the missing pip extra.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EngineSpec:
    """One selectable engine: required config fields (dotted attr path + camelCase name
    for the warning), the optional modules it needs (probed by :func:`preflight`, never
    imported here), and a lazily-importing factory. Factory signature is
    registry-specific (stt/tts pass ``cfg``; vad adds the rates). ``required_any`` lists
    alternative field-sets; satisfied when any one set is fully present."""

    build: Callable[..., Any]
    required: tuple[tuple[str, str], ...] = ()
    required_any: tuple[tuple[tuple[str, str], ...], ...] = ()
    modules: tuple[str, ...] = ()


def resolve_attr(cfg: Any, dotted: str) -> Any:
    """Fetch a possibly-nested config attr (``"sensevoice.model_path"``)."""
    obj = cfg
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def missing_fields(cfg: Any, spec: EngineSpec) -> list[str]:
    """Unset required field names; for ``required_any``, the closest alternative's."""
    missing = [name for attr, name in spec.required if not resolve_attr(cfg, attr)]
    if spec.required_any:
        best: list[str] | None = None
        for alt in spec.required_any:
            gaps = [name for attr, name in alt if not resolve_attr(cfg, attr)]
            if not gaps:
                return missing
            if best is None or len(gaps) < len(best):
                best = gaps
        missing += best or []
    return missing


# Top-level module name of a failed import -> the pyproject extra providing it. Only
# make_audio/make_stt/make_tts/make_vad reach here; realtime/aec/otel/text-frontend
# catch ImportError themselves with more context.
_EXTRA_BY_MODULE = {
    "numpy": "ondevice",
    "onnxruntime": "ondevice",
    "kaldi_native_fbank": "ondevice",
    "webrtcvad": "webrtc",
    "alsaaudio": "pyalsa",
    # A .rknn load with NEITHER runtime installed surfaces as the LAST import attempt's
    # module ("rknn"); "rknnlite" covers a future direct import. The extra resolves only
    # on aarch64 Linux <= py3.12; elsewhere the real fix is a .onnx artifact.
    "rknn": "rknn",
    "rknnlite": "rknn",
    "espeakng_loader": "espeak",
}


def describe_build_error(exc: BaseException) -> str:
    """``str(exc)``, plus the install hint when it is a missing optional extra."""
    missing = exc.name if isinstance(exc, ModuleNotFoundError) else None
    extra = _EXTRA_BY_MODULE.get((missing or "").partition(".")[0])
    if extra:
        return f"{exc}; pip install 'nanobot-channel-voice[{extra}]'"
    return str(exc)


def preflight(
    cfg: Any, engine: str, table: dict[str, EngineSpec], *, prefix: str = "",
    block: str | None = None,
) -> str | None:
    """Validate-time mirror of a registry's fallback triggers: the static reason the
    selected engine would degrade at start (unresolvable ``weights`` key, unset required
    fields, missing optional dependency), else None. Engines outside the table are always
    None. Nothing is imported or loaded; runtime construction can still fail. ``prefix``
    (``"vad."``) makes field names full config keys; ``block`` names the weights
    sub-block when it differs from the engine (``vad.turn`` -> ``"smartturn"``)."""
    spec = table.get(engine)
    if spec is None:
        return None
    from nanobot_channel_voice.weights import WeightsError, apply_weights

    try:
        cfg = apply_weights(cfg, block or engine)
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
