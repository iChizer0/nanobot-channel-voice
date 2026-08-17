"""Local OpenAI-compatible transcription endpoint over the SINGLETON STT adapter.

``POST /v1/audio/transcriptions`` (multipart, the OpenAI shape) -> ``{"text": ...}``.
Core's transcription consumers (WebUI mic dictation, channel voice notes) reach the
plugin's on-device STT through this: core's provider registry is closed, but its OpenAI
adapter honors a per-provider ``apiBase``. The channel passes the SAME adapter object
the voice pipeline decodes with (real targets cannot fit a duplicate in RAM/NPU);
requests serialize on a lock. Zero new dependencies: HTTP/1.1 over
``asyncio.start_server``, multipart by boundary split, browser audio (webm/opus, m4a,
mp3, ...) transcoded by the ``ffmpeg`` BINARY; S16 WAV needs nothing. Loopback by
default; an optional bearer key (core sends the provider entry's ``apiKey``) guards
wider exposure.
"""

from __future__ import annotations

import asyncio
import hmac
import io
import json
import os
import re
import shutil
import tempfile
import wave
from contextlib import suppress

from loguru import logger

from nanobot_channel_voice.config import SttServeConfig
from nanobot_channel_voice.stt.base import SttAdapter, transcribe_chunked

_ROUTE = ("POST", "/v1/audio/transcriptions")
_HEADER_TIMEOUT_S = 10.0
_BODY_TIMEOUT_S = 60.0
_FFMPEG_TIMEOUT_S = 60.0
_STOP_GRACE_S = 3.0  # shutdown: let an in-flight decode land before cancelling handlers
_TARGET_RATE = 16000  # ffmpeg output; every adapter accepts (pcm, rate) and resamples
# Requests in flight at once: bounds concurrent ffmpeg processes and buffered bodies.
_MAX_INFLIGHT = 4
# Head budget, applied BEFORE auth and the _MAX_INFLIGHT gate. The byte budget is the
# real bound: each readline() may return up to the 64 KiB StreamReader limit.
_MAX_HEAD_LINES = 100
_MAX_HEAD_BYTES = 64 * 1024
# Decode-duration ceiling on BOTH ingest branches (ffmpeg -t, and a frame cap on the
# WAV fast path): a small compressed upload can expand to hours of PCM. Core's WebUI
# caps dictation at 120 s; 300 s of 16 kHz mono is ~9.6 MB.
_MAX_DECODE_S = 300
# ``; key=value`` / ``; key="value"`` params of one header line.
_HEADER_PARAM = re.compile(r';\s*([\w*-]+)\s*=\s*(?:"([^"]*)"|([^;\s]*))')


class _HttpError(Exception):
    def __init__(self, status: int, reason: str, detail: str):
        super().__init__(detail)
        self.status = status
        self.reason = reason
        self.detail = detail


class SttHttpServer:
    """One tiny purpose-built HTTP server; the adapter is BORROWED, never owned."""

    def __init__(self, adapter: SttAdapter, cfg: SttServeConfig):
        self._adapter = adapter
        self._cfg = cfg
        self._server: asyncio.base_events.Server | None = None
        # One serve-side decode at a time: never stack ONNX runs on a contended NPU.
        self._lock = asyncio.Lock()
        self._inflight = 0  # loop-side, no lock; excess requests get 503, not a queue
        self._handlers: set[asyncio.Task[None]] = set()
        self._log = logger.bind(component="stt-serve")

    @property
    def port(self) -> int:
        """The bound port (meaningful after start; resolves port=0)."""
        assert self._server is not None and self._server.sockets
        return self._server.sockets[0].getsockname()[1]

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, host=self._cfg.host, port=self._cfg.port
        )
        self._log.info(
            "serving on-device STT at http://{}:{}/v1/audio/transcriptions{}",
            self._cfg.host, self.port, "" if self._cfg.api_key else " (no auth: loopback only)",
        )

    async def stop(self) -> None:
        if self._server is None:
            return
        self._server.close()
        # wait_closed() returns only once every handler released its transport, and a
        # stalled upload holds that for the header + body + ffmpeg timeouts, hence the
        # cap-then-cancel; the GRACE, not the cancel, is what lets an in-flight decode
        # land, since one already inside to_thread runs on regardless.
        try:
            await asyncio.wait_for(asyncio.shield(self._server.wait_closed()), _STOP_GRACE_S)
        except asyncio.TimeoutError:
            for task in list(self._handlers):
                task.cancel()
            await self._server.wait_closed()
        self._server = None

    # ---- request handling ---------------------------------------------------

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._handlers.add(task)  # stop() cancels through these
        try:
            counted = False
            try:
                try:
                    method, path, headers = await asyncio.wait_for(
                        self._read_head(reader), timeout=_HEADER_TIMEOUT_S
                    )
                except ValueError as exc:  # oversized/garbled head (StreamReader limit)
                    raise _HttpError(400, "Bad Request", f"unreadable request head: {exc}") from None
                if (method, path.split("?", 1)[0]) != _ROUTE:
                    raise _HttpError(404, "Not Found", f"no route for {method} {path}")
                self._check_auth(headers)
                if self._inflight >= _MAX_INFLIGHT:
                    raise _HttpError(
                        503, "Service Unavailable",
                        "transcription queue is full; retry shortly",
                    )
                self._inflight += 1
                counted = True
                body = await asyncio.wait_for(
                    self._read_body(reader, headers), timeout=_BODY_TIMEOUT_S
                )
                audio, filename = _multipart_file(headers.get("content-type", ""), body)
                pcm, rate = await self._ingest(audio, filename)
                async with self._lock:
                    # Chunked: uploads (WebUI dictation runs to 120 s, the ingest cap to
                    # 300 s) routinely outrun a fixed decode window.
                    text = await transcribe_chunked(self._adapter, pcm, rate)
                payload = json.dumps({"text": text}, ensure_ascii=False).encode()
                self._respond(writer, 200, "OK", payload)
            except _HttpError as exc:
                self._log.warning("request rejected ({}): {}", exc.status, exc.detail)
                self._respond(
                    writer, exc.status, exc.reason,
                    json.dumps({"error": {"message": exc.detail}}).encode(),
                )
                # 401/413 decide before the upload finishes, and closing mid-send resets
                # the client (it reports a connection error, not our status): drain first.
                with suppress(Exception):  # gone peer: the finally closes anyway
                    await writer.drain()
                    await asyncio.wait_for(self._discard(reader), timeout=2.0)
            except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError):
                return  # client vanished mid-request: nothing to answer
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a handler bug must not kill the server
                self._log.exception("transcription request failed")
                self._respond(
                    writer, 500, "Internal Server Error",
                    b'{"error": {"message": "transcription failed; see gateway logs"}}',
                )
            with suppress(Exception):
                await writer.drain()
        finally:
            if task is not None:
                self._handlers.discard(task)
            if counted:
                self._inflight -= 1
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()

    @staticmethod
    async def _discard(reader: asyncio.StreamReader) -> None:
        while await reader.read(65536):
            pass

    async def _read_head(self, reader: asyncio.StreamReader) -> tuple[str, str, dict[str, str]]:
        request_line = (await reader.readline()).decode("latin-1").strip()
        parts = request_line.split(" ")
        if len(parts) != 3:
            raise _HttpError(400, "Bad Request", f"malformed request line: {request_line!r}")
        headers: dict[str, str] = {}
        budget = _MAX_HEAD_BYTES
        for _ in range(_MAX_HEAD_LINES):
            raw = await reader.readline()
            line = raw.decode("latin-1")
            if line in ("\r\n", "\n", ""):
                return parts[0].upper(), parts[1], headers
            budget -= len(raw)
            if budget < 0:
                break
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
        raise _HttpError(
            431, "Request Header Fields Too Large",
            f"headers exceed {_MAX_HEAD_LINES} lines / {_MAX_HEAD_BYTES} bytes",
        )

    def _check_auth(self, headers: dict[str, str]) -> None:
        if not self._cfg.api_key:
            return
        got = headers.get("authorization", "")
        want = f"Bearer {self._cfg.api_key}"
        # Constant-time: a plain != leaks the matching prefix length by timing.
        if not hmac.compare_digest(got.encode(), want.encode()):
            raise _HttpError(401, "Unauthorized", "missing or wrong bearer token")

    def _max_bytes(self) -> int:
        return self._cfg.max_upload_mb * 1024 * 1024

    async def _read_body(self, reader: asyncio.StreamReader, headers: dict[str, str]) -> bytes:
        try:
            length = int(headers.get("content-length", ""))
            if length < 0:  # readexactly(-1) would raise ValueError -> a bogus 500
                raise ValueError(length)
        except ValueError:
            raise _HttpError(
                411, "Length Required", "Content-Length is required (no chunked upload)"
            ) from None
        if length > self._max_bytes():
            raise _HttpError(
                413, "Payload Too Large",
                f"{length} bytes > stt.serve.maxUploadMb={self._cfg.max_upload_mb}",
            )
        return await reader.readexactly(length)

    # ---- audio ingest -------------------------------------------------------

    async def _ingest(self, audio: bytes, filename: str) -> tuple[bytes, int]:
        """Any uploaded container -> (S16_LE mono PCM, rate) for the adapter."""
        wav = _plain_wav_pcm(audio)
        if wav is not None:
            return wav
        return await self._ffmpeg_pcm(audio, filename)

    async def _ffmpeg_pcm(self, audio: bytes, filename: str) -> tuple[bytes, int]:
        if shutil.which("ffmpeg") is None:
            raise _HttpError(
                415, "Unsupported Media Type",
                f"'{filename}' is not S16 WAV and ffmpeg is not installed; "
                "install ffmpeg to accept browser audio (webm/opus, m4a, mp3, ...)",
            )
        # A temp FILE, not a pipe: mp4/m4a keep their index at the END, which a
        # non-seekable stdin cannot serve.
        fd, path = await asyncio.to_thread(tempfile.mkstemp, "-stt-serve")
        try:
            await asyncio.to_thread(_write_fd, fd, audio)  # closes fd
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-v", "error", "-i", path,
                "-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(_TARGET_RATE),
                "-t", str(_MAX_DECODE_S),
                "pipe:1",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                pcm, err = await asyncio.wait_for(proc.communicate(), timeout=_FFMPEG_TIMEOUT_S)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                raise _HttpError(415, "Unsupported Media Type", "ffmpeg timed out") from None
            if proc.returncode != 0 or not pcm:
                detail = err.decode("utf-8", "replace").strip()[:200] or "no audio decoded"
                raise _HttpError(
                    415, "Unsupported Media Type", f"ffmpeg could not decode '{filename}': {detail}"
                )
            return pcm, _TARGET_RATE
        finally:
            with suppress(OSError):
                await asyncio.to_thread(os.unlink, path)

    def _respond(self, writer: asyncio.StreamWriter, status: int, reason: str, body: bytes) -> None:
        writer.write(
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n".encode("latin-1") + body
        )


def _write_fd(fd: int, data: bytes) -> None:
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


def _header_params(header: str) -> dict[str, str]:
    return {
        m[1].lower(): m[2] if m[2] is not None else m[3] for m in _HEADER_PARAM.finditer(header)
    }


def _disposition(head: bytes) -> str:
    for line in head.decode("latin-1", "replace").split("\r\n"):
        if line.lower().startswith("content-disposition:"):
            return line
    return ""


def _multipart_file(content_type: str, body: bytes) -> tuple[bytes, str]:
    """The ``file`` part of an OpenAI-shape multipart upload -> (bytes, filename).

    Boundary split over a memoryview, NOT the stdlib ``email`` parser: that regex-splits
    the payload into per-line fragments and rebuilds it (seconds of CPU, ~11x transient
    RSS for a 30 MB upload) on the loop that also runs capture, sink pacing and barge-in;
    nor can it take a ``BytesIO``, whose TextIOWrapper newline-translates the bytes and
    corrupts every container."""
    if "multipart/form-data" not in content_type.lower():
        raise _HttpError(
            400, "Bad Request", "expected multipart/form-data (the OpenAI transcription shape)"
        )
    boundary = _header_params(content_type).get("boundary") or ""
    delim = b"--" + boundary.encode("latin-1")
    cursor = body.find(delim) if boundary else -1
    if cursor < 0:
        raise _HttpError(400, "Bad Request", "unparseable multipart body")
    view = memoryview(body)
    fallback: tuple[bytes, str] | None = None
    while body[cursor + len(delim):cursor + len(delim) + 2] != b"--":  # not the closing delimiter
        head_end = body.find(b"\r\n\r\n", cursor)
        if head_end < 0:
            break
        end = body.find(b"\r\n" + delim, head_end + 4)  # the CRLF belongs to the delimiter
        if end < 0:
            break
        params = _header_params(_disposition(body[cursor:head_end]))
        payload = view[head_end + 4:end]
        cursor = end + 2
        if not payload:
            continue
        filename = params.get("filename") or "upload"
        if params.get("name") == "file":
            return bytes(payload), filename
        if params.get("filename") and fallback is None:
            fallback = (bytes(payload), filename)  # tolerate a client naming the part oddly
    if fallback is not None:
        return fallback
    raise _HttpError(400, "Bad Request", "multipart body carries no 'file' part")


def _plain_wav_pcm(data: bytes) -> tuple[bytes, int] | None:
    """S16 WAV -> (mono PCM, native rate) without ffmpeg; None if not that."""
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            channels = w.getnchannels()
            if w.getsampwidth() != 2 or channels not in (1, 2):
                return None  # exotic widths/layouts: let ffmpeg do it properly
            rate = w.getframerate()
            # Capped at READ time, on frames: the bound holds for any claimed rate, and
            # neither the buffer nor the downmix below sees the tail.
            frames = w.readframes(min(w.getnframes(), _MAX_DECODE_S * rate))
    except Exception:  # noqa: BLE001 - not a readable WAV after all
        return None
    if not frames or not rate:
        return None
    if channels == 2:
        import numpy as np  # present wherever an on-device adapter is (ondevice extra)

        stereo = np.frombuffer(frames[: len(frames) // 4 * 4], dtype="<i2").reshape(-1, 2)
        # int32 accumulate + shift: mean(axis=1) would materialize a float64 4x as big.
        frames = ((stereo[:, 0].astype(np.int32) + stereo[:, 1]) >> 1).astype("<i2").tobytes()
    return frames, rate
