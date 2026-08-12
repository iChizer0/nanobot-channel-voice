"""OpenAI-compatible ``/audio/speech`` TTS adapter (httpx only).

Drives both OpenAI cloud TTS and any OpenAI-compatible local server (e.g.
Kokoro-FastAPI): point ``apiBase`` at it. Mirrors nanobot's transcription adapters:
base-vs-full-URL resolution, Bearer auth, retry/backoff on transient failures.
"""

from __future__ import annotations

import asyncio
import io
import time
import wave
from contextlib import suppress
from typing import Literal

import httpx
from loguru import logger

from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes
from nanobot_channel_voice.tts.base import TtsAdapter, is_wav

_DEFAULT_URL = "https://api.openai.com/v1/audio/speech"
_SPEECH_PATH = "audio/speech"
_RETRIES = 2
_BACKOFF_S = (0.5, 1.0)  # indexed by attempt: len() must stay >= _RETRIES
_TOTAL_BUDGET_MULT = 2.0  # whole retry ladder gets this many timeout_s
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_RETRYABLE_EXC = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)
# Encoded-audio magic: compat servers return these with HTTP 200 even for
# response_format=pcm; fed to the S16 sink they play as noise.
_COMPRESSED_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"OggS", b"fLaC")


def _speech_url(api_base: str | None) -> str:
    if not api_base:
        return _DEFAULT_URL
    base = api_base.rstrip("/")
    return base if base.endswith(_SPEECH_PATH) else f"{base}/{_SPEECH_PATH}"


def _body_text(resp: httpx.Response) -> str:
    return resp.text.strip().replace("\n", " ")[:300]


def _is_text_payload(content_type: str) -> bool:
    ct = content_type.split(";", 1)[0].strip().lower()
    # Deny-list, not allow-list: raw PCM legitimately arrives as
    # application/octet-stream or with no content-type at all.
    return ct.startswith("application/json") or ct.startswith("text/") or "+json" in ct


class OpenAITtsAdapter(TtsAdapter):
    def __init__(
        self,
        *,
        api_key: str | None,
        api_base: str | None,
        model: str,
        voice: str,
        audio_format: Literal["wav", "pcm"] = "wav",
        timeout_s: float = 60.0,
        pcm_sample_rate: int = 24000,
    ):
        self._api_key = api_key
        self._url = _speech_url(api_base)
        self._model = model
        self._voice = voice
        self._format = audio_format
        self._timeout = timeout_s
        self._pcm_sample_rate = pcm_sample_rate
        self.output_rate = pcm_sample_rate if audio_format == "pcm" else None
        self.probe_ok = self._url != _DEFAULT_URL  # only the cloud endpoint bills
        # Reused across calls: a per-chunk client would put a TCP+TLS handshake in the
        # first-audio path.
        self._client: httpx.AsyncClient | None = None
        self._log = logger.bind(component="tts-openai")

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            with suppress(Exception):
                await client.aclose()

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        text = text.strip()
        if not text:
            return b""
        if not self._api_key and self._url == _DEFAULT_URL:
            self._log.warning("OpenAI TTS needs an api key (tts.apiKey or OPENAI_API_KEY)")
            return b""

        body = {
            "model": self._model,
            "input": text,
            "voice": voice or self._voice,
            "response_format": self._format,
        }
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

        data = await self._post(body, headers)
        if not data:
            return b""
        if self._format == "pcm":
            if data.startswith(_COMPRESSED_MAGIC):
                self._log.warning("TTS ignored response_format=pcm and sent encoded audio; dropping")
                return b""
            return pcm_to_wav_bytes(data, self._pcm_sample_rate)
        if not is_wav(data):
            self._log.warning("TTS returned non-WAV data for response_format=wav; dropping")
            return b""
        return data

    async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
        if self._format != "pcm":
            raise NotImplementedError("synthesize_pcm requires tts.audioFormat=pcm")
        text = text.strip()
        if not text:
            return b""
        if not self._api_key and self._url == _DEFAULT_URL:
            self._log.warning("OpenAI TTS needs an api key (tts.apiKey or OPENAI_API_KEY)")
            return b""
        body = {
            "model": self._model,
            "input": text,
            "voice": voice or self._voice,
            "response_format": "pcm",
        }
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        pcm = await self._post(body, headers)
        if pcm.startswith(_COMPRESSED_MAGIC):
            # No decoder here; streamed raw it would be sustained full-scale noise.
            self._log.warning("TTS ignored response_format=pcm and sent encoded audio; dropping")
            return b""
        if is_wav(pcm):
            # Streamed raw, the 44-byte RIFF header is a click: strip the container.
            self._log.warning("TTS returned WAV for response_format=pcm; stripping container")
            try:
                with wave.open(io.BytesIO(pcm), "rb") as w:
                    if w.getsampwidth() != 2 or w.getnchannels() != 1:
                        self._log.warning("...and it isn't S16 mono; dropping the chunk")
                        return b""
                    if w.getframerate() != self._pcm_sample_rate:
                        self._log.warning(
                            "...at {} Hz (expected {}); playback will be off-speed",
                            w.getframerate(), self._pcm_sample_rate,
                        )
                    pcm = w.readframes(w.getnframes())
            except Exception:  # noqa: BLE001 - lied twice: not even valid WAV
                return b""
        # An odd byte count (truncated response) misaligns every later S16 sample.
        return pcm[: len(pcm) & ~1]

    @staticmethod
    async def _backoff(attempt: int, deadline: float) -> None:
        # Clamped to the deadline: the sleep is wall time the caller is waiting too.
        await asyncio.sleep(min(_BACKOFF_S[attempt], max(0.0, deadline - time.monotonic())))

    async def _post(self, body: dict, headers: dict) -> bytes:
        client = self._get_client()
        # httpx's read timeout resets on every socket read, so a server trickling one
        # byte at a time never trips it and wedges the turn in SPEAKING. Bound each
        # attempt AND the ladder: the ladder alone lets one stalled attempt eat the
        # whole budget with no retry left; per-attempt alone lets retries stack past it.
        deadline = time.monotonic() + _TOTAL_BUDGET_MULT * self._timeout
        for attempt in range(_RETRIES + 1):
            left = deadline - time.monotonic()
            if left <= 0:
                self._log.warning("TTS budget exhausted after {} attempt(s)", attempt)
                return b""
            try:
                # CancelledError is a BaseException, so barge-in still cancels through.
                resp = await asyncio.wait_for(
                    client.post(self._url, json=body, headers=headers),
                    min(self._timeout, left),
                )
            except (TimeoutError, *_RETRYABLE_EXC) as exc:
                if attempt < _RETRIES:
                    await self._backoff(attempt, deadline)
                    continue
                self._log.warning("TTS request failed: {}", exc)
                return b""
            except Exception as exc:  # noqa: BLE001
                self._log.warning("TTS request error: {}", exc)
                return b""

            if resp.status_code in _RETRYABLE_STATUS and attempt < _RETRIES:
                await self._backoff(attempt, deadline)
                continue
            if resp.status_code != 200:
                self._log.warning("TTS HTTP {}: {}", resp.status_code, _body_text(resp))
                return b""
            if _is_text_payload(resp.headers.get("content-type", "")):
                self._log.warning("TTS returned a non-audio body: {}", _body_text(resp))
                return b""
            return resp.content
        return b""
