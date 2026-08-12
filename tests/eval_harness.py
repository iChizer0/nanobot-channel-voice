"""Scripted-conversation eval harness (cross-framework residual #1).

Drives a REAL LocalBackend end-to-end — endpointer, verdict ladder, chunker,
TTS worker, paced sink — with the suite's fakes standing in for hardware and
models: a flag-driven VAD, a queue-driven batch STT, and a deterministic TTS
whose audio length is proportional to the text (so heard-up-to accounting is
meaningful). The harness plays BOTH outer roles: it is the capture pump
(``user_says`` pushes frames) and the agent (``agent_replies`` feeds the bus
reply back), mirroring the LiveKit ``AgentSession.run()`` / Pipecat local-loop
shape over our own seams.

Time semantics: frames are pushed without sleeps, so VAD/endpointer time is
frame-counted and effectively instant; only sink playout paces at wall clock.
Assertions therefore target outcomes, orderings, markers and counters — keep
scripted replies short so drains stay sub-second.

Engines are constructor-injectable; a live-engine mode over
``NANOBOT_VOICE_REF_DIR`` (like test_ondevice_real.py) is the planned follow-up.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import OutputAudio, StateHint, VoiceState
from nanobot_channel_voice.backend.local import LocalBackend
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.tts.base import TtsAdapter
from nanobot_channel_voice.vad.base import Vad

_RATE = 16000
_FRAME_MS = 20
_FRAME = b"\x01\x00" * (_RATE // 1000 * _FRAME_MS)
_MS_PER_CHAR = 6.0  # deterministic speech-length model for the fake TTS


class _FlagVad(Vad):
    """is_speech mirrors the harness's 'the user is speaking now' flag."""

    def __init__(self) -> None:
        self.flag = False

    def is_speech(self, frame: bytes) -> bool:
        return self.flag

    def scale_floor(self, factor: float) -> None:
        pass


class _ScriptTts(TtsAdapter):
    """Audio whose duration tracks the text, so played_ms maps into words."""

    output_rate = _RATE

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        return pcm_to_wav_bytes(await self.synthesize_pcm(text), _RATE)

    async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
        ms = max(_FRAME_MS, _MS_PER_CHAR * len(text))
        return b"\x01\x00" * int(_RATE / 1000 * ms)


class EvalConversation:
    """One scripted conversation. Use as an async context manager."""

    def __init__(self, **cfg_over):
        cfg = VoiceConfig.model_validate({
            "aec": "soft",       # open mic: barge-in verdicts reachable while SPEAKING
            "duckDb": -12.0,
            "bargeIn": {
                "minWords": 2,
                "ackPhrases": ["ok", "right", "好的"],
                "stopPhrases": ["stop", "shut up", "wait", "停"],
                "heardMarker": True,
            },
            **cfg_over,
        })
        self.vad = _FlagVad()
        self.sink = AudioSink(NullPlayback(), mode="stream")
        self.published: list[tuple[str, str]] = []
        self.interrupts = 0
        self.states: list[VoiceState] = []
        self._stt: deque[str] = deque()

        async def transcribe(pcm: bytes) -> str:
            return self._stt.popleft() if self._stt else ""

        async def publish(text: str, token: str) -> None:
            self.published.append((text, token))

        async def interrupt() -> None:
            self.interrupts += 1

        self.backend = LocalBackend(
            cfg, vad=self.vad, tts=_ScriptTts(), sink=self.sink,
            transcribe=transcribe, publish_text=publish, interrupt=interrupt,
        )

    async def __aenter__(self) -> EvalConversation:
        await self.sink.start()
        await self.backend.start(instructions=None, tools=[], on_event=self._dispatch)
        return self

    async def __aexit__(self, *exc) -> None:
        await self.backend.close()
        await self.sink.stop()

    async def _dispatch(self, event) -> None:
        # The shell's dispatch, reduced to what the backend loop needs.
        if isinstance(event, OutputAudio):
            self.sink.enqueue(event)
        elif isinstance(event, StateHint):
            self.states.append(event.state)

    # ---- the two conversation roles --------------------------------------

    async def user_says(self, text: str, *, ms: int = 400) -> None:
        """Speak for ``ms`` (frame-counted), close the utterance, and wait for its
        verdict to be fully processed."""
        self._stt.append(text)
        self.vad.flag = True
        for _ in range(max(1, ms // _FRAME_MS)):
            await self.backend.push_audio(_FRAME)
        self.vad.flag = False
        hangover = self.backend._cfg.vad.hangover_ms // _FRAME_MS
        for _ in range(hangover + 2):
            await self.backend.push_audio(_FRAME)
        await self.backend._utt_queue.join()

    async def user_noise(self, *, speech_frames: int, silence_frames: int) -> None:
        """VAD-flagged frames with NO transcript behind them (leak/noise shape)."""
        self.vad.flag = True
        for _ in range(speech_frames):
            await self.backend.push_audio(_FRAME)
        self.vad.flag = False
        for _ in range(silence_frames):
            await self.backend.push_audio(_FRAME)

    async def agent_replies(self, text: str) -> None:
        await self.backend.speak_final(text)

    # ---- waiting ----------------------------------------------------------

    async def wait_state(self, state: VoiceState, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self.backend._turn is not state:
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"state {state} not reached (at {self.backend._turn})"
                )
            await asyncio.sleep(0.005)

    async def wait_played_ms(self, ms: float, timeout: float = 5.0) -> None:
        """Let the reply become audibly underway before acting on it."""
        deadline = time.monotonic() + timeout
        while self.sink.played_ms() < ms:
            if time.monotonic() >= deadline:
                raise AssertionError("playback never progressed")
            await asyncio.sleep(0.005)

    # ---- inspection ---------------------------------------------------------

    def texts(self) -> list[str]:
        return [t for t, _ in self.published]

    def counter(self, name: str) -> int:
        return self.backend._metrics.counters.get(name, 0)
