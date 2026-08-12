"""Runtime re-export: the manifest contract pins the runtime target inside
``nanobot.channels.voice``; the implementation lives in ``nanobot_channel_voice``."""

from nanobot_channel_voice.channel import VoiceChannel

__all__ = ["VoiceChannel"]
