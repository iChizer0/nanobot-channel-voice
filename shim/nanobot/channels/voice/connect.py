"""Connector re-export: the manifest contract pins the connector target inside
``nanobot.channels.voice``; the implementation lives in ``nanobot_channel_voice``."""

from nanobot_channel_voice.webui_sync import VoiceSyncStore

__all__ = ["VoiceSyncStore"]
