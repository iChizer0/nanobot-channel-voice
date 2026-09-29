"""The gateway's one weights-store writer: the WebUI's Apply, and the channel's own fetch.

Core's channel connector seam (``ChannelPlugin.connector``) drives the Apply: a poll
answering ``succeeded`` makes core enable the channel, which is when a pending patch
applies. The run fetches what the resolved setup runs, again when the index has changed
it, and removes what this same flow installed and it no longer runs. ``plan=true`` reports
what a run would do from the cached index, re-attaches an Apply still going, and says what
the channel's own fetch is doing (``background``). ``refresh=true`` reloads the index in a
background task, since core answers a socket's requests one at a time and the form would
wait behind it, and so does a plan that finds the cache loaded from another index than the
section names. One store per process (:func:`store`): staged ``.partial-<pid>-*`` files
and the cache's temp file are unique per process only, so one run and one reload go at a
time. The channel's own run is never answered as ``succeeded``: the channel rebuilds itself.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from nanobot.channels.connect import ChannelConnectError, QueryParams, query_first

from nanobot_channel_voice import weights as w
from nanobot_channel_voice.sync import (
    SyncPlan,
    accepted_notices,
    plan_sync,
    run_sync,
    voice_section,
)

MANAGED_BY = "webui"
INDEX_TIMEOUT_S = 10.0
INDEX_FRESH_S = 60.0  # the channel's own fetch reads a cache this young instead of reloading
POLL_INTERVAL_MS = 1000
CLOSE_WAIT_S = 2.0  # at shutdown: let the thread reach its next chunk, then leave it
DELIVERY_GRACE_S = 10.0  # a finished Apply's restart is on its way while a panel polls


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
    transient: bool = False  # a later attempt may pass
    ended: float = 0.0  # monotonic, once finished
    owner: Any = None  # the channel's keeper, on its own run
    # An Apply queued behind the channel's own run: the notices it accepted (it plans
    # again after that run), and what that run landed of its plan (keys, bytes).
    after: _Session | None = None
    accepted: frozenset[str] = frozenset()
    carried: tuple[int, int] = (0, 0)

    @property
    def finished(self) -> bool:
        return self.result is not None or self.error is not None

    def snapshot(self) -> dict[str, Any]:
        if self.after is not None:
            return self.after.snapshot()  # the run it waits behind is the one moving bytes
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


_STORE: VoiceSyncStore | None = None


def store() -> VoiceSyncStore:
    """The process's store: core's connector factory, and the channel's keeper."""
    global _STORE
    if _STORE is None:
        _STORE = VoiceSyncStore()
    return _STORE


def own_fetch() -> Any:
    """The running channel's keeper status (``keeper.Status``), None when no channel keeps
    its models. For the validator, which must not create a store."""
    keeper = None if _STORE is None else _STORE._keeper
    return None if keeper is None else keeper.status()


class VoiceSyncStore:
    """One run at a time in the gateway process, an Apply or the channel's own; the
    download runs in a thread, as does the index reload."""

    def __init__(self) -> None:
        self._session: _Session | None = None  # the WebUI's Apply
        self._auto: _Session | None = None  # the channel's own run
        self._keeper: Any = None
        self._starting = asyncio.Lock()
        self._reload: asyncio.Task[None] | None = None
        self._reload_error: str | None = None
        self._reload_transient = False
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
        if action == "cancel" and (auto := self._auto) is not None and auto.id == session_id:
            return self._cancel_own(auto)
        session = self._session
        if session is None or session.id != session_id:
            raise ChannelConnectError("no such voice sync session", status=404)
        if action == "poll":
            return self._deliver(session)
        if action == "cancel":
            return self.cancel(session)
        raise ChannelConnectError(f"unsupported voice sync action: {action}", status=404)

    async def close(self) -> None:
        runs = [s for s in (self._session, self._auto) if s is not None and s.task is not None]
        for session in runs:
            session.stopped = True
        if pending := {s.task for s in runs if not s.task.done()}:  # type: ignore[union-attr]
            # The thread stops between chunks; shutdown does not wait out a stalled link.
            await asyncio.wait(pending, timeout=CLOSE_WAIT_S)
        if self._reloading():
            self._reload.cancel()  # type: ignore[union-attr]  # the fetch itself times out on its own

    def _running(self) -> _Session | None:
        session = self._session
        if session is None or session.task is None or session.task.done():
            return None
        return session

    def _own_running(self) -> _Session | None:
        session = self._auto
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

    # ---- the channel's own fetch -------------------------------------------------

    def attach(self, keeper: Any) -> None:
        self._keeper = keeper

    def detach(self, keeper: Any) -> None:
        """A keeper leaving with its channel: its run stops, unless an Apply waits behind it
        for the same bytes."""
        if self._keeper is keeper:
            self._keeper = None
        own = self._auto
        queued = self._session is not None and self._session.after is own
        if own is not None and own.owner is keeper and not own.finished and not queued:
            own.stopped = True

    async def wait_idle(self) -> None:
        """Until no run holds the store."""
        while pending := {
            s.task for s in (self._session, self._auto)
            if s is not None and s.task is not None and not s.task.done()
        }:
            await asyncio.wait(pending)

    def applying(self) -> bool:
        """An Apply runs, or has just finished and a polling panel is about to hand core the
        restart it asks for."""
        session = self._session
        if session is None or session.delivered:
            return False
        return not session.finished or time.monotonic() - session.ended < DELIVERY_GRACE_S

    async def run_own(self, plan: SyncPlan, index: dict[str, dict[str, Any]], owner: Any) -> _Session | None:
        """Start the channel's own run; None when another run holds the store by now."""
        async with self._starting:
            if self._running() is not None or self._own_running() is not None:
                return None
            session = _Session(id=uuid.uuid4().hex, plan=plan, index=index, owner=owner)
            session.task = asyncio.create_task(self._run(session, w.store_root()))
            self._auto = session
            return session

    def _cancel_own(self, own: _Session) -> dict[str, Any]:
        """The panel's Cancel on the channel's own run: it stops, and the channel fetches
        nothing more on its own until it restarts."""
        _stop(own)
        return {"session_id": own.id, "status": "cancelled", "message": "The background download stopped."}

    async def index_for(self, sources: list[str]) -> dict[str, dict[str, Any]]:
        """The index ``sources`` name, read afresh unless the cache from them is young:
        through the one reload when they are the section's own (the panel's cache with it),
        else privately. A failed reload falls back to a cache from them."""
        root = w.store_root()
        cached = await asyncio.to_thread(w.cached_index, root)
        if cached and cached[2] == sources and time.time() - cached[1] < INDEX_FRESH_S:
            return cached[0]
        configured, _ = await asyncio.to_thread(_sources, root)
        if sources != configured:
            return await asyncio.to_thread(w.load_index, sources, timeout=INDEX_TIMEOUT_S)
        self._start_reload()
        if self._reload is not None:
            await asyncio.wait({self._reload})
        cached = await asyncio.to_thread(w.cached_index, root)
        if cached and cached[2] == sources:
            return cached[0]
        raise w.WeightsError(
            self._reload_error or "the model index did not load", transient=self._reload_transient,
        )

    def background(self) -> dict[str, Any] | None:
        """What the channel's own fetch is doing (``fetching``, ``waiting`` to retry,
        ``stopped``), or ``reloading`` until the pipeline has loaded what landed; None while
        it has nothing to do."""
        keeper = self._keeper
        status = None if keeper is None else keeper.status()
        if status is None or (status.state == "idle" and not status.reload):
            return None
        payload: dict[str, Any] = {
            "state": "reloading" if status.state == "idle" else status.state,
            "keys": list(status.keys),
            "error": status.error,
            "retry_unix": status.retry_unix,
        }
        own = self._auto
        if status.state == "fetching" and own is not None and not own.finished:
            payload.update(session_id=own.id, progress=own.snapshot())
        return payload

    # ---- plan -------------------------------------------------------------------

    def plan(self, *, refreshing: bool = False) -> dict[str, Any]:
        """What a run would do, from the cache as it is now. Offline, the cache stands and
        the status carries the last reload's error."""
        root = w.store_root()
        index, cached_unix, sources = _cached(root)
        plan, accepted = self._plan(root, index, sources)
        return {
            "session_id": "",
            "status": "planned",
            "index": {"cached_unix": cached_unix, "refreshing": refreshing, "error": self._reload_error},
            "plan": _plan_payload(plan, index, accepted),
            "background": self.background(),
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
            self._reload_transient = getattr(exc, "transient", False)
            self._failed = sources
            if not isinstance(exc, (w.WeightsError, ChannelConnectError)):
                logger.exception("voice: model index reload failed")
        else:
            if self._keeper is not None:
                self._keeper.poke()  # the network answers, and the index may list more

    def _plan(
        self, root: Path, index: dict[str, dict[str, Any]], sources: list[str] | None,
    ) -> tuple[SyncPlan, frozenset[str]]:
        """The Apply's plan, and the notices the section accepts on its own."""
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
        plan = plan_sync(section, index, root, managed_by=MANAGED_BY, used_only=True, updates=current)
        return plan, accepted_notices(section)

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
        plan, standing = await asyncio.to_thread(self._plan, root, index, sources)
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
        _refuse_unknown(plan)
        accepted = accepted | standing
        _refuse_unaccepted(plan, accepted)
        # What the run needs at its peak: it fetches everything before it removes anything,
        # so the space a prune will free is not space the download can use.
        if plan.free_bytes and plan.fetch_bytes > plan.free_bytes:
            raise ChannelConnectError(
                f"not enough disk space: {_size(plan.fetch_bytes)} needed, "
                f"{_size(plan.free_bytes)} free under {root}"
            )
        # The channel's own run: the Apply waits behind it, and first stops it unless its
        # setup runs every model that run still fetches.
        own = self._own_running()
        if own is not None and not await asyncio.to_thread(_covers, plan, own, root):
            _stop(own)
        if plan.empty:
            # Nothing to move: "succeeded" still (re)starts the channel, the Apply half.
            return {"session_id": "", "status": "succeeded", "message": "Models are in place."}
        session = _Session(
            id=uuid.uuid4().hex, plan=plan, index=index, after=own, accepted=frozenset(accepted),
        )
        session.task = asyncio.create_task(self._run(session, root))
        self._session = session
        return self._status(session)

    async def _run(self, session: _Session, root: Path) -> None:
        try:
            if (before := session.after) is not None:
                await asyncio.wait({before.task})  # type: ignore[arg-type]
                session.after = None
                if session.stopped:
                    raise w.WeightsError("sync cancelled")
                await self._replan(session, root)
            session.result = await asyncio.to_thread(
                run_sync, session.plan, session.index, root,
                managed_by=MANAGED_BY,
                progress=lambda key, file, done: setattr(session, "at", (key, file, done)),
                should_stop=lambda: session.stopped,
            )
        except Exception as exc:  # noqa: BLE001 - a job nobody awaits: any failure must end the polls
            session.error = str(exc) or type(exc).__name__
            session.transient = getattr(exc, "transient", False)
            if not isinstance(exc, (w.WeightsError, OSError, ChannelConnectError)):
                logger.exception("voice: model sync failed")
        finally:
            session.ended = time.monotonic()

    async def _replan(self, session: _Session, root: Path) -> None:
        """An Apply after the run it waited behind: plan again, what that run landed now
        counted as carried."""
        index, _cached_unix, sources = await asyncio.to_thread(_cached, root)
        plan, standing = await asyncio.to_thread(self._plan, root, index, sources)
        _refuse_unknown(plan)
        _refuse_unaccepted(plan, session.accepted | standing)
        gone = [k for k in session.plan.fetch if k not in plan.fetch]
        session.carried = (len(gone), sum(session.plan.sizes[k] for k in gone))
        session.plan, session.index = plan, index

    def cancel(self, session: _Session) -> dict[str, Any]:
        """Ask the run to stop and answer at once: the thread notices between chunks, and
        the poll that follows reports how it ended. Waiting here would hold the socket for
        a chunk of a slow download, and a run that has just finished would answer
        "succeeded" — a restart nobody asked for. A queued Apply stops the run it waits
        behind too: that is the download on screen."""
        session.stopped = True
        if (before := session.after) is not None:
            _stop(before)
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
            carried, carried_bytes = session.carried
            plan = session.plan
            new = len(plan.fetch) - len(plan.update) + carried
            updated = len(plan.update) - len(kept)
            moved = [f"fetched {new}"] if new else []
            if updated:
                moved.append(f"updated {updated}")
            parts = [f"{' and '.join(moved)} ({(fetched + carried_bytes) / 1e6:,.0f} MB)"] if moved else []
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


def _stop(own: _Session) -> None:
    """Stop the channel's own run, and its keeper's fetching until the channel restarts: a
    Cancel, or an Apply that takes over. A run already over is left to what it landed."""
    if not own.finished:
        own.stopped = True
        if own.owner is not None:
            own.owner.suspend()


def _refuse_unknown(plan: SyncPlan) -> None:
    if plan.unknown:
        raise ChannelConnectError(
            f"not in the model index: {', '.join(plan.unknown)}; fetch by hand or pick "
            "an indexed model"
        )


def _refuse_unaccepted(plan: SyncPlan, accepted: set[str] | frozenset[str]) -> None:
    if unaccepted := sorted(set(plan.notices) - accepted):
        raise ChannelConnectError(f"accept the notice for {', '.join(unaccepted)} first")


def _covers(plan: SyncPlan, own: _Session, root: Path) -> bool:
    """Whether the Apply fetches every model the channel's own run has yet to land."""
    have = w.installed(root)
    return {k for k in own.plan.fetch if k not in have} <= set(plan.fetch)


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


def _plan_payload(
    plan: SyncPlan, index: dict[str, dict[str, Any]], accepted: frozenset[str],
) -> dict[str, Any]:
    return {
        "fetch": [
            {
                "key": k,
                "bytes": plan.sizes[k],
                "update": k in plan.update,
                "license": index[k].get("license"),
                "notice": plan.notices.get(k),
                "accepted": k in accepted,
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
