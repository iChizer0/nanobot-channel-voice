"""The WebUI setup spec re-declares config enums (the manifest must stay
dependency-free, so it cannot import them). Parse it with ``ast``: no core
install needed, and pin every choice tuple to its config ``Literal``.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Literal, Union, get_args, get_origin

from nanobot_channel_voice.backend.profiles import PROFILES
from nanobot_channel_voice.config import VoiceConfig

_MANIFEST = Path(__file__).resolve().parents[1] / "shim/nanobot/channels/voice/manifest.py"


def _manifest_choices() -> dict[str, tuple[str, ...]]:
    tree = ast.parse(_MANIFEST.read_text())
    spec = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "SETUP_SPEC" for t in node.targets)
    )
    fields = next(kw.value for kw in spec.keywords if kw.arg == "fields")
    out: dict[str, tuple[str, ...]] = {}
    for key, value in zip(fields.keys, fields.values, strict=True):
        for kw in getattr(value, "keywords", []):
            if kw.arg == "choices":
                out[ast.literal_eval(key)] = ast.literal_eval(kw.value)
    return out


def _literal_args(annotation) -> tuple[str, ...]:
    """The Literal arm of an annotation (``bool | Literal[...]`` included)."""
    if get_origin(annotation) is Literal:
        return get_args(annotation)
    for arm in get_args(annotation):
        if get_origin(arm) is Literal:
            return get_args(arm)
    raise AssertionError(f"no Literal arm in {annotation!r}")


def test_manifest_enums_match_the_config_literals():
    # stt.provider / tts.provider stay OUT of the WebUI: their leaves collide
    # into two "Provider" labels and their non-default values need config-file
    # companions. vad.engine has a unique leaf and is exposed.
    from nanobot_channel_voice.config import VadConfig

    choices = _manifest_choices()
    assert set(choices) == {"backend", "aec", "vad.engine"}

    def literal(model, name):
        return _literal_args(model.model_fields[name].annotation)

    assert choices["backend"] == literal(VoiceConfig, "backend")
    assert set(choices["backend"]) == {"local"} | set(PROFILES)
    assert choices["aec"] == literal(VoiceConfig, "aec")
    assert choices["vad.engine"] == literal(VadConfig, "engine")


def test_union_literal_arm_is_found():
    assert _literal_args(Union[bool, Literal["a", "b"]]) == ("a", "b")
