"""The WebUI's Apply: sync the weights store to the section, with progress, then start.

Core's channel connector seam (``ChannelPlugin.connector``) drives this: a poll answering
``succeeded`` makes core enable the channel, which is when a pending patch applies. The
run fetches what the resolved setup runs, again when the index has changed it, and
removes what this same flow installed and it no longer runs. ``plan=true`` reports what a
run would do from the cached index, or re-attaches one already going. ``refresh=true``
reloads the index in a background task, since core answers a socket's requests one at a
time and the form would wait behind it, and so does a plan that finds the cache loaded
from another index than the section names.
Nothing else here touches the network.
"""

from __future__ import annotations

import asyncio
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from nanobot.channels.connect import ChannelConnectError, QueryParams, query_first

from nanobot_channel_voice import weights as w
from nanobot_channel_voice.sync import SyncPlan, plan_sync, run_sync, voice_section

MANAGED_BY = "webui"
INDEX_TIMEOUT_S = 10.0
POLL_INTERVAL_MS = 1000
CLOSE_WAIT_S = 2.0  # at shutdown: let the thread reach its next chunk, then leave it


@dataclass
class _Session:
    id: str
    plan: SyncPlan
    index: dict[str, dict[str, Any]]
    task: asyncio.Task[None] | None = None
    stopped: bool = False
    delivered: bool = False  # its last word reached a panel, and core with it
    # (key, file, bytes fetched so far) as run_sync counts them: one store, so no lock
    at: tuple[str | None, str | None, int] = (None, None, 0)
    result: tuple[int, int, dict[str, str]] | None = None  # run_sync's: fetched, freed, kept
    error: str | None = None

    @property
    def finished(self) -> bool:
        return self.result is not None or self.error is not None

    def snapshot(self) -> dict[str, Any]:
        key, file, done = self.at
        return {
            "stage": "done" if self.finished else "fetch",
            "key": key,
            "file": file,
            "done_bytes": self.plan.fetch_bytes if self.result else done,
            "total_bytes": self.plan.fetch_bytes,
            "keys_done": len(self.plan.fetch) if self.result else self.plan.fetch.index(key) if key else 0,
            "keys_total": len(self.plan.fetch),
        }


class VoiceSyncStore:
    """One sync at a time, in the gateway process; the download runs in a thread, as does
    the index reload."""

    def __init__(self) -> None:
        self._session: _Session | None = None
        self._starting = asyncio.Lock()
        self._reload: asyncio.Task[None] | None = None
        self._reload_error: str | None = None
        # The index the last reload failed on: a cache from another one reloads by itself,
        # but not again for these, or every poll would retry a dead link. Retry does.
        self._failed: list[str] | None = None

    async def handle(self, action: str, query: QueryParams) -> dict[str, Any]:
        if action == "start":
            if _flag(query, "plan"):
                # A run still going, or one that ended while nobody polled: the panel that
                # reopens follows it to the end, and core hears its "succeeded" once.
                if (pending := self._session) is not None and not pending.delivered:
                    return self._deliver(pending)
                return await self._planned(refresh=_flag(query, "refresh"))
            accepted = {k for k in (query_first(query, "accept") or "").split(",") if k}
            return await self.start(accepted)
        session_id = (query_first(query, "session_id") or "").strip()
        session = self._session
        if session is None or session.id != session_id:
            raise ChannelConnectError("no such voice sync session", status=404)
        if action == "poll":
            return self._deliver(session)
        if action == "cancel":
            return self.cancel(session)
        raise ChannelConnectError(f"unsupported voice sync action: {action}", status=404)

    async def close(self) -> None:
        if (session := self._session) is not None and session.task is not None:
            session.stopped = True
            with suppress(TimeoutError, asyncio.CancelledError):
                # The thread stops between chunks; shutdown does not wait out a stalled link.
                await asyncio.wait_for(asyncio.shield(session.task), CLOSE_WAIT_S)
        if self._reloading():
            self._reload.cancel()  # type: ignore[union-attr]  # the fetch itself times out on its own

    def _running(self) -> _Session | None:
        session = self._session
        if session is None or session.task is None or session.task.done():
            return None
        return session

    def _deliver(self, session: _Session) -> dict[str, Any]:
        """The session's status, its last word only once: core (re)starts the channel on
        every "succeeded" it sees, so a second poll of a finished run must not answer one."""
        payload = self._status(session)
        if session.finished:
            session.delivered = True
            self._session = None
        return payload

    # ---- plan -------------------------------------------------------------------

    def plan(self, *, refreshing: bool = False) -> dict[str, Any]:
        """What a run would do, from the cache as it is now. Offline, the cache stands and
        the status carries the last reload's error."""
        root = w.store_root()
        index, cached_unix, sources = _cached(root)
        return {
            "session_id": "",
            "status": "planned",
            "index": {"cached_unix": cached_unix, "refreshing": refreshing, "error": self._reload_error},
            "plan": _plan_payload(self._plan(root, index, sources), index),
        }

    async def _planned(self, *, refresh: bool) -> dict[str, Any]:
        """The plan, a reload behind it when asked or when the cache is another index's
        than the section names. An index that just failed reloads again only when asked,
        and the failure of one the section no longer names stops being the status."""
        configured, cached = await asyncio.to_thread(_sources, w.store_root())
        if configured is not None and configured != self._failed:
            self._failed = self._reload_error = None
            if cached != configured:
                self._start_reload()
        if refresh:
            self._start_reload()  # one at a time: a no-op while one runs
        return await asyncio.to_thread(self.plan, refreshing=self._reloading())

    def _start_reload(self) -> None:
        """Reload the index in the background, one reload at a time; the cache it writes
        is what the next plan reads."""
        if not self._reloading():
            self._reload_error = None
            self._reload = asyncio.create_task(self._reload_index())

    def _reloading(self) -> bool:
        return self._reload is not None and not self._reload.done()

    async def _reload_index(self) -> None:
        sources = None
        try:
            sources = await asyncio.to_thread(_configured)
            await asyncio.to_thread(w.refresh_index, sources, w.store_root(), timeout=INDEX_TIMEOUT_S)
            self._failed = None
        except Exception as exc:  # noqa: BLE001 - nobody awaits this: a silent failure would read as a clean reload
            self._reload_error = str(exc) or type(exc).__name__
            self._failed = sources
            if not isinstance(exc, (w.WeightsError, ChannelConnectError)):
                logger.exception("voice: model index reload failed")

    def _plan(
        self, root: Path, index: dict[str, dict[str, Any]], sources: list[str] | None,
    ) -> SyncPlan:
        from nanobot.config.loader import get_config_path

        from nanobot_channel_voice.config import section_index

        try:
            section = voice_section(get_config_path())
        except w.WeightsError as exc:
            raise ChannelConnectError(str(exc)) from None
        try:
            # An installed model is judged by the section's own index, never another's cache.
            current = sources == section_index(section)
        except ValueError:
            current = False
        return plan_sync(
            section, index, root, managed_by=MANAGED_BY, used_only=True, updates=current,
        )

    # ---- run --------------------------------------------------------------------

    async def start(self, accepted: set[str]) -> dict[str, Any]:
        # The whole start is under the lock: two panels pressing Apply at once would else
        # both pass the check while the first is still planning, and fetch one key twice.
        async with self._starting:
            return await self._start(accepted)

    async def _start(self, accepted: set[str]) -> dict[str, Any]:
        if self._running() is not None:
            raise ChannelConnectError("a voice model sync is already running", status=409)
        root = w.store_root()
        index, _cached_unix, sources = await asyncio.to_thread(_cached, root)
        plan = await asyncio.to_thread(self._plan, root, index, sources)
        # Never a download from an index the section has moved away from: until the one it
        # names has loaded, the cache lists the other one's files. An Apply that downloads
        # nothing does not wait on it, offline or behind a blocked hub.
        if plan.fetch or plan.unknown:
            configured = await asyncio.to_thread(_configured)
            if sources != configured:
                if self._reloading() or self._failed != configured:
                    self._start_reload()  # one at a time: a no-op while one runs
                    raise ChannelConnectError("the model index is still loading, apply once it has", status=409)
                raise ChannelConnectError(f"the model index has not loaded: {self._reload_error}")
        if plan.unknown:
            raise ChannelConnectError(
                f"not in the model index: {', '.join(plan.unknown)}; fetch by hand or pick "
                "an indexed model"
            )
        unaccepted = sorted(set(plan.notices) - accepted)
        if unaccepted:
            raise ChannelConnectError(f"accept the notice for {', '.join(unaccepted)} first")
        # What the run needs at its peak: it fetches everything before it removes anything,
        # so the space a prune will free is not space the download can use.
        if plan.free_bytes and plan.fetch_bytes > plan.free_bytes:
            raise ChannelConnectError(
                f"not enough disk space: {_size(plan.fetch_bytes)} needed, "
                f"{_size(plan.free_bytes)} free under {root}"
            )
        if plan.empty:
            # Nothing to move: "succeeded" still (re)starts the channel, the Apply half.
            return {"session_id": "", "status": "succeeded", "message": "Models are in place."}
        session = _Session(id=uuid.uuid4().hex, plan=plan, index=index)
        session.task = asyncio.create_task(self._run(session, root))
        self._session = session
        return self._status(session)

    async def _run(self, session: _Session, root: Path) -> None:
        try:
            session.result = await asyncio.to_thread(
                run_sync, session.plan, session.index, root,
                managed_by=MANAGED_BY,
                progress=lambda key, file, done: setattr(session, "at", (key, file, done)),
                should_stop=lambda: session.stopped,
            )
        except Exception as exc:  # noqa: BLE001 - a job nobody awaits: any failure must end the polls
            session.error = str(exc) or type(exc).__name__
            if not isinstance(exc, (w.WeightsError, OSError)):
                logger.exception("voice: model sync failed")

    def cancel(self, session: _Session) -> dict[str, Any]:
        """Ask the run to stop and answer at once: the thread notices between chunks, and
        the poll that follows reports how it ended. Waiting here would hold the socket for
        a chunk of a slow download, and a run that has just finished would answer
        "succeeded" — a restart nobody asked for."""
        session.stopped = True
        if session.finished:
            return self._deliver(session)
        return self._status(session)

    def _status(self, session: _Session) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": session.id,
            "interval_ms": POLL_INTERVAL_MS,
            "progress": session.snapshot(),
        }
        if session.result is not None:
            fetched, freed, kept = session.result
            plan = session.plan
            new, updated = len(plan.fetch) - len(plan.update), len(plan.update) - len(kept)
            moved = [f"fetched {new}"] if new else []
            if updated:
                moved.append(f"updated {updated}")
            parts = [f"{' and '.join(moved)} ({fetched / 1e6:,.0f} MB)"] if moved else []
            if plan.prune:
                parts.append(f"removed {len(plan.prune)} ({freed / 1e6:,.0f} MB)")
            message = f"Models {', '.join(parts)}." if parts else "Models unchanged."
            payload.update(status="succeeded", message=message)
            if kept:
                # The run still succeeds, since the installed models run: the panel shows why.
                payload["warning"] = "; ".join(
                    f"{key} stays as installed: {why}" for key, why in kept.items()
                )
        elif session.error is not None:
            cancelled = session.stopped
            payload.update(status="cancelled" if cancelled else "failed", message=session.error)
        else:
            payload["status"] = "pending"
        return payload


def _cached(root: Path) -> tuple[dict[str, dict[str, Any]], int, list[str] | None]:
    """The cached index, when it was fetched and what from; empty, 0 and None when
    nothing is cached."""
    cached = w.cached_index(root)
    return cached if cached else ({}, 0, None)


def _sources(root: Path) -> tuple[list[str] | None, list[str] | None]:
    """The index the section names, None when that cannot be read (the form or the plan
    says why), and the one the cache was loaded from, None when nothing is cached."""
    try:
        configured = _configured()
    except ChannelConnectError:
        configured = None
    cached = w.cached_index(root)
    return configured, (cached[2] if cached else None)


def _configured() -> list[str]:
    """The model index the gateway's section names. ``ChannelConnectError`` says why it
    cannot tell."""
    from nanobot.config.loader import get_config_path

    from nanobot_channel_voice.config import section_index

    try:
        return section_index(voice_section(get_config_path()))
    except (w.WeightsError, ValueError) as exc:
        raise ChannelConnectError(str(exc)) from None


def _plan_payload(plan: SyncPlan, index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "fetch": [
            {
                "key": k,
                "bytes": plan.sizes[k],
                "update": k in plan.update,
                "license": index[k].get("license"),
                "notice": plan.notices.get(k),
            }
            for k in plan.fetch
        ],
        "prune": [{"key": k, "bytes": plan.sizes[k]} for k in plan.prune],
        "unknown": plan.unknown,
        "fetch_bytes": plan.fetch_bytes,
        "prune_bytes": plan.prune_bytes,
        "free_bytes": plan.free_bytes,
    }


def _size(n: int) -> str:
    """Bytes as the refusal should read them: a shortfall of a few MB never rounds to 0."""
    for unit, step in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= step:
            return f"{n / step:,.1f} {unit}"
    return f"{n} bytes"


def _flag(query: QueryParams, key: str) -> bool:
    return (query_first(query, key) or "").strip().lower() in {"1", "true", "yes"}
