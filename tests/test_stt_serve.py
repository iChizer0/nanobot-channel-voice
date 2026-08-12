"""stt.serve: the local OpenAI-compatible transcription endpoint."""

from __future__ import annotations

import asyncio
import io
import shutil
import subprocess
import time
import wave
from contextlib import suppress

import httpx
import pytest

from nanobot_channel_voice.config import SttServeConfig
from nanobot_channel_voice.stt import serve as serve_mod
from nanobot_channel_voice.stt.serve import SttHttpServer


class FakeAdapter:
    """Records calls; detects overlapping decodes (the singleton must serialize)."""

    def __init__(self, text: str = "hello world"):
        self.text = text
        self.calls: list[tuple[int, int]] = []
        self._busy = False
        self.overlapped = False

    async def transcribe(self, pcm: bytes, rate: int) -> str:
        if self._busy:
            self.overlapped = True
        self._busy = True
        await asyncio.sleep(0.02)
        self._busy = False
        self.calls.append((len(pcm), rate))
        return self.text

    async def warmup(self) -> None:  # pragma: no cover - protocol completeness
        pass


def wav_bytes(frames: int = 1600, rate: int = 16000, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x01\x00" * frames * channels)
    return buf.getvalue()


def run(coro):
    return asyncio.run(coro)


async def _post(server: SttHttpServer, *, headers=None, **kwargs) -> httpx.Response:
    url = f"http://127.0.0.1:{server.port}/v1/audio/transcriptions"
    async with httpx.AsyncClient() as client:
        return await client.post(url, headers=headers, **kwargs)


def _files(payload: bytes, filename: str = "a.wav", mime: str = "audio/wav"):
    return {"file": (filename, payload, mime), "model": (None, "local")}


async def _raw_status(server: SttHttpServer, request: bytes) -> int:
    """Status of a hand-written request; httpx cannot emit the malformed heads below."""
    reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
    try:
        writer.write(request)
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), 5.0)
    finally:
        writer.close()
        with suppress(Exception):
            await writer.wait_closed()
    return int(line.split()[1])


_BOUNDARY = b"----nbVoice7d3"


def _multipart(payload: bytes, *, quote: bool = False, filename: bytes = b"a.wav"):
    """(content-type, body) in the OpenAI shape, byte-exact: no client library."""
    body = (
        b"--" + _BOUNDARY + b"\r\n"
        b'Content-Disposition: form-data; name="model"\r\n'
        b"\r\nlocal\r\n"
        b"--" + _BOUNDARY + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="' + filename + b'"\r\n'
        b"Content-Type: application/octet-stream\r\n"
        b"\r\n" + payload + b"\r\n"
        b"--" + _BOUNDARY + b"--\r\n"
    )
    spec = b'"' + _BOUNDARY + b'"' if quote else _BOUNDARY
    return "multipart/form-data; boundary=" + spec.decode(), body


async def _with_server(case, cfg: SttServeConfig | None = None, adapter: FakeAdapter | None = None):
    adapter = adapter or FakeAdapter()
    server = SttHttpServer(adapter, cfg or SttServeConfig(enabled=True, port=0))
    await server.start()
    try:
        await case(server, adapter)
    finally:
        await server.stop()


def test_wav_round_trip_uses_native_rate():
    async def case(server, adapter):
        resp = await _post(server, files=_files(wav_bytes(frames=800, rate=22050)))
        assert resp.status_code == 200
        assert resp.json() == {"text": "hello world"}
        assert adapter.calls == [(1600, 22050)]  # native rate: the adapter resamples

    run(_with_server(case))


def test_stereo_wav_is_downmixed():
    pytest.importorskip("numpy")

    async def case(server, adapter):
        resp = await _post(server, files=_files(wav_bytes(frames=1000, channels=2)))
        assert resp.status_code == 200
        assert adapter.calls == [(2000, 16000)]  # 1000 mono samples

    run(_with_server(case))


def test_bearer_auth_when_configured():
    cfg = SttServeConfig(enabled=True, port=0, api_key="s3cr3t")

    async def case(server, adapter):
        assert (await _post(server, files=_files(wav_bytes()))).status_code == 401
        assert (
            await _post(server, files=_files(wav_bytes()),
                        headers={"Authorization": "Bearer wrong"})
        ).status_code == 401
        ok = await _post(server, files=_files(wav_bytes()),
                         headers={"Authorization": "Bearer s3cr3t"})
        assert ok.status_code == 200 and adapter.calls

    run(_with_server(case, cfg=cfg))


def test_unknown_route_and_non_multipart_are_rejected():
    async def case(server, adapter):
        async with httpx.AsyncClient() as client:
            wrong = await client.post(f"http://127.0.0.1:{server.port}/nope", content=b"x")
        assert wrong.status_code == 404
        bad = await _post(server, content=b"raw bytes",
                          headers={"Content-Type": "application/octet-stream"})
        assert bad.status_code == 400
        assert not adapter.calls

    run(_with_server(case))


def test_oversize_upload_is_refused():
    cfg = SttServeConfig(enabled=True, port=0, max_upload_mb=1)

    async def case(server, adapter):
        resp = await _post(server, files=_files(b"\x00" * (1024 * 1024 + 4096)))
        assert resp.status_code == 413
        assert not adapter.calls

    run(_with_server(case, cfg=cfg))


def test_compressed_audio_without_ffmpeg_says_why(monkeypatch):
    monkeypatch.setattr(serve_mod.shutil, "which", lambda name: None)

    async def case(server, adapter):
        resp = await _post(server, files=_files(b"OggS" + b"\x00" * 64, "mic.ogg", "audio/ogg"))
        assert resp.status_code == 415
        assert "ffmpeg" in resp.json()["error"]["message"]
        assert not adapter.calls

    run(_with_server(case))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_browser_style_compressed_audio_is_transcoded(tmp_path):
    # What a WebUI MediaRecorder upload actually looks like: not WAV.
    src = tmp_path / "src.wav"
    src.write_bytes(wav_bytes(frames=4410, rate=22050))
    ogg = tmp_path / "mic.ogg"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(src), "-c:a", "libvorbis", str(ogg)],
        check=True,
    )

    async def case(server, adapter):
        resp = await _post(server, files=_files(ogg.read_bytes(), "mic.ogg", "audio/ogg"))
        assert resp.status_code == 200
        (nbytes, rate), = adapter.calls
        assert rate == 16000  # ffmpeg output contract
        assert abs(nbytes / (2 * 16000) - 0.2) < 0.05  # ~200 ms survived the round trip

    run(_with_server(case))


def test_saturated_queue_returns_503_instead_of_piling_up():
    slow = FakeAdapter()

    async def case(server, adapter):
        # Far more simultaneous uploads than _MAX_INFLIGHT: the extras must be
        # REFUSED (bounded memory/ffmpeg), not queued without limit.
        results = await asyncio.gather(
            *(_post(server, files=_files(wav_bytes())) for _ in range(serve_mod._MAX_INFLIGHT + 2))
        )
        statuses = sorted(r.status_code for r in results)
        assert set(statuses) <= {200, 503}
        assert statuses.count(503) >= 1
        assert statuses.count(200) == len(adapter.calls) <= serve_mod._MAX_INFLIGHT

    run(_with_server(case, adapter=slow))


def test_negative_content_length_answers_411():
    # Without the guard, readexactly(-1) raises deep in the handler: 500 + traceback.
    async def case(server, adapter):
        req = (
            "POST /v1/audio/transcriptions HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Content-Type: multipart/form-data; boundary=b\r\n"
            "Content-Length: -1\r\n\r\n"
        ).encode()
        assert await _raw_status(server, req) == 411
        assert not adapter.calls

    run(_with_server(case))


def test_header_flood_answers_431_by_line_count_and_by_bytes():
    async def case(server, adapter):
        head = "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: h\r\n"
        # Shape 1: tiny lines (~1.4 KB total), so only the LINE cap can catch these.
        many = "".join(f"X-Pad-{i}: v\r\n" for i in range(serve_mod._MAX_HEAD_LINES + 5))
        assert await _raw_status(server, (head + many + "\r\n").encode()) == 431
        # Shape 2: two lines, 80 KB, so only the BYTE cap can catch these. Kept under
        # the 64 KiB StreamReader limit per line so it is the budget that trips, not
        # a readline() overrun (that answers 400).
        big = "X-Pad: " + "v" * 40000 + "\r\n"
        assert await _raw_status(server, (head + big + big + "\r\n").encode()) == 431
        assert not adapter.calls

    run(_with_server(case))


def test_long_wav_is_truncated_at_the_decode_ceiling():
    # 400 s at a low rate: cheap to build, and the cap is on FRAMES so it holds
    # for whatever rate the header claims.
    rate = 1000

    async def case(server, adapter):
        resp = await _post(server, files=_files(wav_bytes(frames=400 * rate, rate=rate)))
        assert resp.status_code == 200
        assert adapter.calls == [(serve_mod._MAX_DECODE_S * rate * 2, rate)]

    run(_with_server(case))


def test_multipart_payload_survives_crlf_and_near_boundary_bytes():
    payload = (
        b"RIFF\x00\x01\x02"
        b"\r\n--" + _BOUNDARY[:-1] +          # one byte short of the delimiter
        b"\r\n\x00--" + _BOUNDARY +           # the delimiter, but not at a CRLF
        b"\r\n\r\ntail\x00"
    )
    ctype, body = _multipart(payload)
    assert serve_mod._multipart_file(ctype, body) == (payload, "a.wav")
    ctype, body = _multipart(payload, quote=True)
    assert serve_mod._multipart_file(ctype, body) == (payload, "a.wav")


def test_quoted_boundary_round_trips_over_the_wire():
    async def case(server, adapter):
        wav = wav_bytes(frames=800)
        ctype, body = _multipart(wav, quote=True)
        req = (
            f"POST /v1/audio/transcriptions HTTP/1.1\r\nHost: h\r\n"
            f"Content-Type: {ctype}\r\nContent-Length: {len(body)}\r\n\r\n"
        ).encode() + body
        assert await _raw_status(server, req) == 200
        assert adapter.calls == [(1600, 16000)]  # the exact WAV, not a mangled copy

    run(_with_server(case))


def test_stop_returns_while_a_client_holds_a_half_sent_request(monkeypatch):
    monkeypatch.setattr(serve_mod, "_STOP_GRACE_S", 0.3)

    async def case():
        server = SttHttpServer(FakeAdapter(), SttServeConfig(enabled=True, port=0))
        await server.start()
        _reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        writer.write(
            (
                "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: h\r\n"
                "Content-Type: multipart/form-data; boundary=b\r\n"
                "Content-Length: 1048576\r\n\r\n"
            ).encode()
            + b"\x00" * 16  # ... and then nothing: the body never completes
        )
        await writer.drain()
        await asyncio.sleep(0.1)  # let the handler reach readexactly()
        started = time.monotonic()
        try:
            # Without the grace + cancel, teardown waits out _BODY_TIMEOUT_S.
            await asyncio.wait_for(server.stop(), timeout=10.0)
        finally:
            writer.close()
            with suppress(Exception):
                await writer.wait_closed()
        assert time.monotonic() - started < 2.0

    run(case())


def test_concurrent_requests_never_overlap_the_decode():
    async def case(server, adapter):
        results = await asyncio.gather(
            *(_post(server, files=_files(wav_bytes())) for _ in range(4))
        )
        assert all(r.status_code == 200 for r in results)
        assert len(adapter.calls) == 4
        assert adapter.overlapped is False  # the singleton decoded strictly one at a time

    run(_with_server(case))
