"""On-device inference building blocks: a tiny RKNN/ONNX runtime shared by the
on-device STT, TTS and VAD adapters."""

from __future__ import annotations

from nanobot_channel_voice.ondevice.runtime import OnDeviceModel

__all__ = ["OnDeviceModel"]
