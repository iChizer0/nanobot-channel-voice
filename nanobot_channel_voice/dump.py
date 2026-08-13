"""Capture-segment WAV dumps for by-ear debugging (``debug.dumpAudio``).

Each segment lands as ``utt-<seq>-<HHMMSS>-<verdict>.wav`` - the audio exactly as
the VAD/STT judged it (post-AEC) - plus a ``.raw.wav`` twin of the same span
straight off the mic when a software canceller is active, so a false barge-in can
be attributed to AEC residual vs. real room sound by listening to the pair.

:meth:`submit` only enqueues (producers must never block on disk); one daemon
writer thread does all file I/O, including the startup prune. ``seq`` is dump
order, which can trail capture order: an utterance submits after its STT verdict,
a probe/blip drop immediately - the HHMMSS stamp (taken at submit) is the
capture-side anchor.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import wave
from pathlib import Path

from loguru import logger

from nanobot_channel_voice.aio import Throttle
from nanobot_channel_voice.audio.pcm import pcm_ms, pcm_rms

_QUEUE_DEPTH = 16  # segments in flight; a 30 s max utterance is ~1 MB, so <=~16 MB held


def default_dump_root() -> Path:
    """``$XDG_DATA_HOME|~/.local/share/nanobot-voice/dumps``: the weights store's
    base convention, but never ``$NANOBOT_VOICE_MODELS_DIR`` - that relocates model
    artifacts, not diagnostics."""
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "nanobot-voice" / "dumps"


class AudioDumper:
    """One session's segment writer: a timestamped directory under ``root``, a
    bounded queue, and a best-effort byte cap (older sessions pruned first, then
    the live session's oldest segments)."""

    def __init__(self, root: Path, sample_rate: int, max_bytes: int):
        self._rate = sample_rate
        self._max_bytes = max_bytes
        self._log = logger.bind(component="voice")
        self._root = root
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        session = root / stamp
        if session.exists():  # same-second restart
            session = root / f"{stamp}-{os.getpid()}"
        session.mkdir(parents=True, exist_ok=True)
        self.dir = session
        self._seq = 0
        self._written: list[tuple[Path, int]] = []  # oldest first
        self._written_bytes = 0
        self._queue: queue.Queue[tuple[str, str, bytes, bytes | None] | None] = (
            queue.Queue(maxsize=_QUEUE_DEPTH)
        )
        self._drop_warn = Throttle()
        self._thread = threading.Thread(
            target=self._writer, name="voice-dump", daemon=True
        )
        self._thread.start()
        self._log.info("audio dump on: writing capture segments to {}", session)

    # ---- producers (any thread) -------------------------------------------

    def submit(self, verdict: str, pcm: bytes, raw: bytes | None = None) -> None:
        """Queue one segment; never blocks. ``raw`` = the pre-AEC span of ``pcm``."""
        if not pcm:
            return
        try:
            self._queue.put_nowait((time.strftime("%H%M%S"), verdict, pcm, raw))
        except queue.Full:
            if self._drop_warn.ready():
                self._log.warning("audio dump backlogged; dropping a '{}' segment", verdict)

    def close(self) -> None:
        """Stop accepting work and drain what is queued (writes are ms-scale)."""
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            return  # writer wedged (dead disk): don't hang teardown behind it
        self._thread.join(timeout=5.0)

    # ---- writer thread -----------------------------------------------------

    def _writer(self) -> None:
        self._prune_old_sessions()  # off-loop: an unlink storm must not stall start()
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._write_one(*item)
            except Exception as exc:  # noqa: BLE001 - diagnostics must never kill audio
                self._log.warning("audio dump write failed: {}", exc)

    def _write_one(self, stamp: str, verdict: str, pcm: bytes, raw: bytes | None) -> None:
        self._seq += 1
        stem = f"utt-{self._seq:04d}-{stamp}-{verdict}"
        path = self.dir / f"{stem}.wav"
        self._write_wav(path, pcm)
        if raw:
            self._write_wav(self.dir / f"{stem}.raw.wav", raw)
        self._log.info(
            "dumped segment #{} {} ({} ms, rms {:.3f}{}) -> {}",
            self._seq, verdict, int(pcm_ms(len(pcm), self._rate)), pcm_rms(pcm),
            ", +raw" if raw else "", path,
        )
        while self._written_bytes > self._max_bytes and len(self._written) > 1:
            old, size = self._written.pop(0)
            try:
                old.unlink(missing_ok=True)
                self._written_bytes -= size
            except OSError:
                break  # best-effort; retried on the next write

    def _write_wav(self, path: Path, pcm: bytes) -> None:
        with wave.open(str(path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self._rate)
            wav.writeframes(pcm)
        size = path.stat().st_size
        self._written.append((path, size))
        self._written_bytes += size

    def _prune_old_sessions(self) -> None:
        """Delete oldest session dirs until what predates this session fits the cap.
        Only our own timestamped dirs are touched - anything else in the tree is a
        user's saved sample."""
        sessions: list[tuple[Path, int]] = []
        try:
            for entry in self._root.iterdir():
                if entry == self.dir or not entry.is_dir() or not entry.name[:8].isdigit():
                    continue
                sessions.append(
                    (entry, sum(f.stat().st_size for f in entry.glob("*.wav")))
                )
        except OSError:
            return
        sessions.sort()  # timestamped names: lexical == chronological
        total = sum(size for _, size in sessions)
        for path, size in sessions:
            if total <= self._max_bytes:
                break
            try:
                for f in path.glob("*.wav"):
                    f.unlink(missing_ok=True)
                path.rmdir()
                total -= size
                self._log.info("audio dump: pruned old session {}", path.name)
            except OSError:
                break
