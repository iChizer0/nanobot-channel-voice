"""Capture-segment WAV dumps for by-ear debugging (``debug.dumpAudio``).

Each segment lands as ``utt-<id>-<verdict>.wav`` (post-AEC, exactly what VAD/STT
judged) plus a ``.raw.wav`` pre-AEC twin when a software canceller is active. ``id``
is the backend's capture-segment id, the same number the ``utt #N:`` summary log line
carries; dump WRITE order can trail capture order.

``manifest.jsonl`` gets one record per segment for jq filtering. Its first line is a
``{"type": "session", ...}`` config record; readers skip lines bearing ``type``. A
record outlives its WAV (the byte cap deletes audio, never manifest lines), so ``file``
may name a pruned segment. ``index.html`` (the packaged ``dump_viewer.html``) lands
beside it: serve the directory and the session is browsable.

:meth:`submit` only enqueues (producers must never block on disk); one daemon writer
thread does all file I/O, including the startup prune.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import wave
from contextlib import suppress
from importlib import resources
from pathlib import Path

from loguru import logger

from nanobot_channel_voice.aio import Throttle
from nanobot_channel_voice.audio.pcm import pcm_ms, pcm_rms

# Bytes of PCM allowed in flight; a segment plus its pre-AEC twin both count, so the
# held RAM is this whatever the utterance length or whether a raw tap is wired. The
# depth cap only exists so a wedged writer (dead disk) refuses the close sentinel.
_QUEUE_BYTES = 16 * 1024 * 1024
_QUEUE_DEPTH = 256
# Head span for the manifest/log rms: bounds the pass, matches the summary line's cap.
_RMS_HEAD_S = 1.0


def default_dump_root() -> Path:
    """``$XDG_DATA_HOME|~/.local/share/nanobot-voice/dumps``: the weights store's base
    convention, but never ``$NANOBOT_VOICE_MODELS_DIR`` - that moves models, not
    diagnostics."""
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "nanobot-voice" / "dumps"


class AudioDumper:
    """One session's segment writer: a timestamped directory under ``root``, a bounded
    queue, and a best-effort byte cap (old sessions pruned first, then oldest
    segments)."""

    def __init__(
        self, root: Path, sample_rate: int, max_bytes: int,
        *, header: dict | None = None,
    ):
        self._rate = sample_rate
        self._max_bytes = max_bytes
        self._header = header
        self._log = logger.bind(component="voice")
        self._root = root
        root.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        session = root / stamp
        if session.exists():  # same-second restart
            session = root / f"{stamp}-{os.getpid()}"
        session.mkdir(parents=True, exist_ok=True)
        self.dir = session
        self._seq = 0  # fallback ids for seq-less submits; backend ids are authoritative
        self._manifest = None  # opened by the writer thread on first record
        self._written: list[tuple[Path, int]] = []  # oldest first
        self._written_bytes = 0
        self._queue: queue.Queue[
            tuple[str, bytes, bytes | None, int | None, dict | None] | None
        ] = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._queued_bytes = 0
        self._bytes_lock = threading.Lock()
        self._drop_warn = Throttle()
        self._thread = threading.Thread(
            target=self._writer, name="voice-dump", daemon=True
        )
        self._thread.start()
        self._log.info("audio dump on: writing capture segments to {}", session)

    # ---- producers (any thread) ---------------------------------------------

    def submit(
        self,
        verdict: str,
        pcm: bytes,
        raw: bytes | None = None,
        *,
        seq: int | None = None,
        meta: dict | None = None,
    ) -> None:
        """Queue one segment; never blocks. ``raw`` = the pre-AEC span of ``pcm``; ``seq``
        = the backend's capture-segment id (filename + manifest identity); ``meta`` =
        extra manifest fields."""
        if not pcm:
            return
        size = len(pcm) + len(raw or b"")
        with self._bytes_lock:
            if self._queued_bytes + size > _QUEUE_BYTES:
                if self._drop_warn.ready():
                    self._log.warning(
                        "audio dump backlogged; dropping a '{}' segment", verdict
                    )
                return
            self._queued_bytes += size
        try:
            self._queue.put_nowait((verdict, pcm, raw, seq, meta))
        except queue.Full:
            with self._bytes_lock:
                self._queued_bytes -= size
            if self._drop_warn.ready():
                self._log.warning("audio dump writer stalled; dropping a '{}' segment", verdict)

    def close(self) -> None:
        """Stop accepting work and drain what is queued (writes are ms-scale)."""
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            return  # writer wedged (dead disk): don't hang teardown behind it
        self._thread.join(timeout=5.0)

    # ---- writer thread ------------------------------------------------------

    def _writer(self) -> None:
        if self._header is not None:  # first manifest line, ahead of any queued record
            try:
                self._append_manifest({"type": "session", **self._header})
            except Exception as exc:  # noqa: BLE001 - diagnostics must never kill audio
                self._log.warning("audio dump session header failed: {}", exc)
        self._install_viewer()
        self._prune_old_sessions()  # off-loop: an unlink storm must not stall start()
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    return
                try:
                    self._write_one(*item)
                except Exception as exc:  # noqa: BLE001 - diagnostics must never kill audio
                    self._log.warning("audio dump write failed: {}", exc)
                finally:
                    with self._bytes_lock:
                        self._queued_bytes -= len(item[1]) + len(item[2] or b"")
        finally:
            if self._manifest is not None:
                with suppress(Exception):
                    self._manifest.close()

    def _write_one(
        self, verdict: str, pcm: bytes, raw: bytes | None,
        seq: int | None, meta: dict | None,
    ) -> None:
        if seq is None:  # defensive: every production submit carries the backend id
            self._seq += 1
            seq = self._seq
        stem = f"utt-{seq:04d}-{verdict}"
        path = self.dir / f"{stem}.wav"
        self._write_wav(path, pcm)
        if raw:
            self._write_wav(self.dir / f"{stem}.raw.wav", raw)
        dur_ms = int(pcm_ms(len(pcm), self._rate))
        rms = round(pcm_rms(pcm[: int(self._rate * _RMS_HEAD_S) * 2]), 3)
        self._log.info(
            "dumped segment #{} {} ({} ms, rms {:.3f}{}) -> {}",
            seq, verdict, dur_ms, rms, ", +raw" if raw else "", path,
        )
        # Identity keys spread second: meta can never clobber them.
        record = {
            **(meta or {}),
            "id": seq, "verdict": verdict, "file": path.name,
            "dur_ms": dur_ms, "rms": rms, "raw": raw is not None,
        }
        self._append_manifest(record)
        while self._written_bytes > self._max_bytes and len(self._written) > 1:
            old, size = self._written[0]
            try:
                old.unlink(missing_ok=True)
            except OSError:
                break  # entry stays in the ledger, so the next write really does retry
            self._written.pop(0)
            self._written_bytes -= size

    def _append_manifest(self, record: dict) -> None:
        if self._manifest is None:
            self._manifest = open(self.dir / "manifest.jsonl", "a", encoding="utf-8")
        # default=str: a non-JSON meta value costs a field, not the record.
        self._manifest.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._manifest.flush()  # a crash must not cost the records that led to it

    def _install_viewer(self) -> None:
        """Copy the packaged viewer in as ``index.html``."""
        try:
            html = (
                resources.files("nanobot_channel_voice") / "dump_viewer.html"
            ).read_bytes()
            (self.dir / "index.html").write_bytes(html)
        except Exception as exc:  # noqa: BLE001 - a missing viewer costs browsing, not audio
            self._log.warning("audio dump viewer not installed: {}", exc)

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
        """Delete oldest session dirs until what predates this session fits the cap. Only
        our own timestamped dirs are touched - anything else is a user's saved sample."""
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
                # Delete only what the dumper wrote; a parked user file survives.
                for f in path.glob("*.wav"):
                    f.unlink(missing_ok=True)
                (path / "manifest.jsonl").unlink(missing_ok=True)
                (path / "index.html").unlink(missing_ok=True)
                with suppress(OSError):
                    path.rmdir()  # left standing if the user parked files inside
                total -= size
                self._log.info("audio dump: pruned old session {}", path.name)
            except OSError:
                break
