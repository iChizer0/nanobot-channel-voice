"""nanobot voice channel: speak to your agent over ALSA audio.

Local backend (default): mic -> VAD endpointing -> STT -> agent -> streamed
reply -> sentence chunker -> TTS -> speaker, with half/full/soft-duplex and
duck-then-confirm barge-in. Alternatively an OpenAI-Realtime-dialect cloud
backend (openai/xai/azure/qwen/glm/stepfun) does ASR + reasoning + TTS end to
end while tool calls still route through nanobot.
"""

from nanobot_channel_voice.channel import VoiceChannel

__all__ = ["VoiceChannel"]
