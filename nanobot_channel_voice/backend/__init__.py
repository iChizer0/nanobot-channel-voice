"""Pluggable voice backends behind the :class:`VoiceBackend` contract (:mod:`.base`).

``local`` wraps the on-box VAD/STT/TTS pipeline + nanobot (full brain, text bus);
the realtime backend speaks the OpenAI-Realtime dialect over a raw WebSocket for
any supplier described by a pure-data :class:`~.profiles.RealtimeProfile`.

This package re-exports nothing, and the websockets client is imported lazily
inside :mod:`.openai_realtime`, so the plugin imports fine without the
``[realtime]`` extra.
"""
