"""Capture-segment WAV dumps for by-ear debugging (``debug.dumpAudio``).

Each segment lands as ``utt-<id>-<verdict>.wav`` - the audio exactly as the
VAD/STT judged it (post-AEC) - plus a ``.raw.wav`` twin of the same span straight
off the mic when a software canceller is active, so a false barge-in can be
attributed to AEC residual vs. real room sound by listening to the pair. ``id``
is the backend's capture-segment id, the same number the ``utt #N:`` summary log
line carries, so a WAV maps to its log record exactly even though dump WRITE
order can trail capture order (an utterance submits after its STT verdict, a
probe/blip drop immediately).

``manifest.jsonl`` in the session directory gets one record per segment (id,
verdict, duration, rms, plus whatever metadata the backend attached - close
shape, STT cost, VAD confidence, capture-side wall stamp), so a dump directory
is filtered with jq before anything is listened to.

:meth:`submit` only enqueues (producers must never block on disk); one daemon
writer thread does all file I/O, including the startup prune.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import wave
from contextlib import suppress
from pathlib import Path

from loguru import logger

from nanobot_channel_voice.aio import Throttle
from nanobot_channel_voice.audio.pcm import pcm_ms, pcm_rms

_QUEUE_DEPTH = 16  # segments in flight; a 30 s max utterance is ~1 MB, so <=~16 MB held
# Head span for the manifest/log rms (triage metadata, not an unbounded pass over a
# max-length utterance), matching the backend summary line's cap.
_RMS_HEAD_S = 1.0


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
        self._seq = 0  # fallback ids for seq-less submits; backend ids are authoritative
        self._manifest = None  # opened by the writer thread on first record
        self._written: list[tuple[Path, int]] = []  # oldest first
        self._written_bytes = 0
        self._queue: queue.Queue[
            tuple[str, bytes, bytes | None, int | None, dict | None] | None
        ] = queue.Queue(maxsize=_QUEUE_DEPTH)
        self._drop_warn = Throttle()
        self._thread = threading.Thread(
            target=self._writer, name="voice-dump", daemon=True
        )
        self._thread.start()
        self._log.info("audio dump on: writing capture segments to {}", session)

    # ---- producers (any thread) -------------------------------------------

    def submit(
        self,
        verdict: str,
        pcm: bytes,
        raw: bytes | None = None,
        *,
        seq: int | None = None,
        meta: dict | None = None,
    ) -> None:
        """Queue one segment; never blocks. ``raw`` = the pre-AEC span of ``pcm``;
        ``seq`` = the backend's capture-segment id (filename + manifest identity);
        ``meta`` = extra manifest fields the backend judged at close/verdict time."""
        if not pcm:
            return
        try:
            self._queue.put_nowait((verdict, pcm, raw, seq, meta))
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
        # Writer stamps second so identity keys can never be clobbered by meta;
        # default=str keeps a future non-JSON meta value from silently costing
        # the whole record.
        record = {
            **(meta or {}),
            "id": seq, "verdict": verdict, "file": path.name,
            "dur_ms": dur_ms, "rms": rms, "raw": raw is not None,
        }
        if self._manifest is None:
            self._manifest = open(self.dir / "manifest.jsonl", "a", encoding="utf-8")
        self._manifest.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        self._manifest.flush()  # a crash must not cost the records that led to it
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
                # Delete only what the dumper wrote; a user file parked in the dir
                # survives (and keeps the dir alive), matching the "anything else
                # in the tree is a user's saved sample" contract.
                for f in path.glob("*.wav"):
                    f.unlink(missing_ok=True)
                (path / "manifest.jsonl").unlink(missing_ok=True)
                with suppress(OSError):
                    path.rmdir()  # left standing if the user parked files inside
                total -= size
                self._log.info("audio dump: pruned old session {}", path.name)
            except OSError:
                break
