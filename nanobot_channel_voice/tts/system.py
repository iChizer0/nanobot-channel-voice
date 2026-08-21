"""Zero-dependency system TTS via ``espeak-ng`` (Linux) or ``say`` (macOS): the
always-available fallback — robotic, but local and instant, so the channel still
speaks with nothing configured. Returns WAV bytes.
"""

from __future__ import annotations

import asyncio
import io
import os
import shutil
import sys
import tempfile
import wave
from contextlib import suppress

from loguru import logger

from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes
from nanobot_channel_voice.tts.base import TtsAdapter

_PIPE = asyncio.subprocess.PIPE
_TTS_TIMEOUT_S = 30.0  # cap for one speakable chunk; a wedged child is killed, not awaited


def _rewrap_stdout_wav(blob: bytes) -> bytes:
    """Re-emit a piped WAV with REAL sizes in the header: espeak-ng cannot seek stdout,
    so ``--stdout`` blobs carry placeholder chunk sizes (data = 0x7ffff000; its
    ``CloseWavFile`` returns early for stdout, never backpatching), and anything trusting
    the header (``wav_duration_ms`` -> echo-rejection holds, sink backlog, calibration)
    would read ~13.5 hours per chunk."""
    try:
        with wave.open(io.BytesIO(blob), "rb") as w:
            rate, channels, sampwidth = w.getframerate(), w.getnchannels(), w.getsampwidth()
            frames = w.readframes(w.getnframes())  # reads what is actually there
    except Exception:  # noqa: BLE001 - unparseable: hand it to playback unchanged
        return blob
    return pcm_to_wav_bytes(frames, rate, channels=channels, sampwidth=sampwidth)


def _read_file(path: str) -> bytes:
    with open(path, "rb") as fh:  # called via to_thread: keep file IO off the loop
        return fh.read()


async def _communicate(proc: asyncio.subprocess.Process, *, timeout: float) -> tuple[bytes, bytes]:
    """``communicate()`` with a hard timeout: kill and reap the child on timeout or
    cancellation, so a wedged ``espeak-ng``/``say`` neither wedges the TTS worker
    (channel stuck in SPEAKING) nor is orphaned."""
    try:
        return await asyncio.wait_for(proc.communicate(), timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        with suppress(ProcessLookupError):
            proc.kill()
        with suppress(Exception):
            await proc.wait()
        raise


class SystemTtsAdapter(TtsAdapter):
    def __init__(self, *, language: str | None = None, espeak_path: str = "espeak-ng"):
        self._language = language
        self.spoken_language = language  # None => the espeak/say default voice
        self._espeak = espeak_path
        self._resolved: tuple[str | None, str | None] | None = None
        self._log = logger.bind(component="tts-system")

    def _resolve(self) -> tuple[str | None, str | None]:
        """(kind, absolute path) of the available binary, resolved once: ``shutil.which``
        stats every PATH entry and this runs on the event loop per speakable chunk, yet
        make_tts builds the fallback even for sessions that never speak. The negative
        result is cached too, so installing espeak-ng mid-session needs a channel
        restart."""
        if self._resolved is None:
            path = shutil.which(self._espeak)
            if path:
                self._resolved = ("espeak", path)
            elif sys.platform == "darwin" and (path := shutil.which("say")):
                self._resolved = ("say", path)
            else:
                self._resolved = (None, None)
                self._log.warning("no system TTS (espeak-ng/say) available")
        return self._resolved

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        text = text.strip()
        if not text:
            return b""
        kind, exe = self._resolve()
        if kind == "espeak" and exe:
            return await self._espeak_ng(exe, text, voice)
        if kind == "say" and exe:
            return await self._macos_say(exe, text, voice)
        return b""

    async def _espeak_ng(self, exe: str, text: str, voice: str | None) -> bytes:
        text = text.replace("・", " ")  # U+30FB: espeak names it and spells neighbors
        args = [exe, "--stdout"]  # absolute path: no second PATH scan in exec
        chosen = voice or self._language
        if chosen:
            args += ["-v", chosen]
        args += ["--", text]
        try:
            proc = await asyncio.create_subprocess_exec(*args, stdout=_PIPE, stderr=_PIPE)
            out, err = await _communicate(proc, timeout=_TTS_TIMEOUT_S)
        except asyncio.TimeoutError:
            self._log.warning("espeak-ng timed out after {}s; killed", _TTS_TIMEOUT_S)
            return b""
        except OSError as exc:
            self._log.warning("espeak-ng launch failed: {}", exc)
            return b""
        if proc.returncode == 0 and out:
            return _rewrap_stdout_wav(out)
        self._log.warning("espeak-ng rc={}: {}", proc.returncode, err.decode("utf-8", "replace").strip())
        return b""

    async def _macos_say(self, exe: str, text: str, voice: str | None) -> bytes:
        # `say` writes AIFF/CAF natively: ask for a WAVE file and read it back.
        # mkstemp for a secure temp path; `say -o` overwrites the empty file.
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        args = [exe, "-o", path, "--file-format=WAVE", "--data-format=LEI16@22050"]
        if voice:
            args += ["-v", voice]
        args += ["--", text]
        try:
            proc = await asyncio.create_subprocess_exec(*args, stdout=_PIPE, stderr=_PIPE)
            _, err = await _communicate(proc, timeout=_TTS_TIMEOUT_S)
            if proc.returncode == 0 and os.path.exists(path):
                return await asyncio.to_thread(_read_file, path)
            self._log.warning("say rc={}: {}", proc.returncode, err.decode("utf-8", "replace").strip())
            return b""
        except asyncio.TimeoutError:
            self._log.warning("say timed out after {}s; killed", _TTS_TIMEOUT_S)
            return b""
        except OSError as exc:
            self._log.warning("say launch failed: {}", exc)
            return b""
        finally:
            with suppress(OSError):
                os.unlink(path)
