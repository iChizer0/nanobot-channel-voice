"""The running channel's own models (``models.autoFetch``): what its setup runs and the store
lacks is fetched in the background through the gateway's one sync store, a failure a later
attempt may pass is tried again on ``models.retryS``, and the channel hears when models
land. Nothing is updated or removed, nor fetched for an edit the channel does not run yet,
nor under a license notice ``models.acceptNotices`` does not list.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Collection
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot_channel_voice import weights as w
from nanobot_channel_voice.aio import cancel_and_wait
from nanobot_channel_voice.sync import complete, plan_missing
from nanobot_channel_voice.webui_sync import store


@dataclass(frozen=True)
class Status:
    """``idle`` (nothing it can fetch is missing), ``fetching``, ``waiting`` to retry at
    ``retry_unix``, or ``stopped``: a failure no retry passes, a Cancel, an Apply."""

    state: str = "idle"
    keys: tuple[str, ...] = ()  # fetching: the run's; waiting or stopped: the ones it lacks
    error: str | None = None
    retry_unix: float | None = None
    held: tuple[str, ...] = ()  # missing under a notice nobody accepted
    unknown: tuple[str, ...] = ()  # missing, and not in the index
    reload: bool = False  # models landed that the pipeline has yet to load


class ModelKeeper:
    """One per channel instance: :meth:`start` with it, :meth:`close` with its stop."""

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg
        self._status = Status()
        self._task: asyncio.Task[None] | None = None
        self._changed = asyncio.Event()
        self._poked = asyncio.Event()
        self._suspended = False
        self._settled = 0  # attempts finished
        self._told: set[str] = set()
        self._last_error: str | None = None
        self.landed = asyncio.Event()  # a run landed models; the channel clears it

    def start(self) -> None:
        store().attach(self)
        self._task = asyncio.create_task(self._loop())

    async def close(self) -> None:
        store().detach(self)
        task, self._task = self._task, None
        await cancel_and_wait(task)

    def status(self) -> Status:
        return self._status

    def poke(self) -> None:
        """Plan again now: the index reloaded, so the network answers."""
        if not self._suspended:
            self._poked.set()

    def loaded(self) -> None:
        """The pipeline was (re)built: what landed before is in it."""
        self._set(reload=False)

    def suspend(self) -> None:
        """Fetch nothing more until the channel restarts: a Cancel, or an Apply took over."""
        self._suspended = True
        self._set(state="stopped", error=None, retry_unix=None)

    async def until(self, ready: Callable[[], bool], keys: Collection[str]) -> str | None:
        """None once ``ready()`` (read in a thread) holds; else why ``keys`` will not come."""
        while True:
            self._changed.clear()
            # Read before the store: a model landing in between then shows as ready.
            status, settled = self._status, self._settled
            if await asyncio.to_thread(ready):
                return None
            if held := [k for k in keys if k in status.held]:
                return (
                    f"the license notice of {', '.join(held)} is not accepted (accept it under "
                    "Models in the WebUI, or list it in models.acceptNotices)"
                )
            if unknown := [k for k in keys if k in status.unknown]:
                return f"the model index does not list {', '.join(unknown)}"
            if status.state == "stopped":
                return status.error or "the download was stopped"
            if status.state == "idle" and settled:
                return "the model index has nothing to fetch for them"
            await self._changed.wait()

    def _set(self, **changes: Any) -> None:
        self._status = replace(self._status, **changes)
        self._changed.set()

    async def _loop(self) -> None:
        failures = 0
        while True:
            try:
                error = await self._attempt()
            except Exception as exc:  # noqa: BLE001 - nobody awaits this task: a bug must still settle it
                logger.exception("voice: the background model fetch failed")
                error = w.WeightsError(str(exc) or type(exc).__name__)
            self._settled += 1
            self._changed.set()
            self._poked.clear()  # the attempt's own index reload is no news
            if self._suspended:
                return
            if error is None:
                failures = 0
                self._last_error = None
            else:
                pauses = self._cfg.models.retry_s
                if error.transient and pauses:
                    pause = pauses[min(failures, len(pauses) - 1)]
                    failures += 1
                    self._set(state="waiting", error=str(error), retry_unix=time.time() + pause)
                    self._log_failure(error, pause)
                    with suppress(TimeoutError):
                        await asyncio.wait_for(self._poked.wait(), pause)
                    continue
                self._set(state="stopped", error=str(error), retry_unix=None)
                self._log_failure(error, None)
            await self._poked.wait()

    async def _attempt(self) -> w.WeightsError | None:
        """Plan what is missing and fetch it; the failure, if any."""
        sync, root = store(), w.store_root()
        while True:
            await sync.wait_idle()  # an Apply holds the store: after it, plan afresh
            if await asyncio.to_thread(complete, self._cfg, root):
                self._set(state="idle", keys=(), error=None, retry_unix=None, held=(), unknown=())
                return None
            try:
                index = await sync.index_for(list(self._cfg.index))
            except w.WeightsError as exc:
                return exc
            plan, held = await asyncio.to_thread(
                plan_missing, self._cfg, index, root, accepted=self._cfg.models.accept_notices,
            )
            self._tell(held, plan.unknown)
            found = {"held": tuple(held), "unknown": tuple(plan.unknown)}
            if not plan.fetch:
                self._set(state="idle", keys=(), error=None, retry_unix=None, **found)
                return None
            if plan.free_bytes and plan.fetch_bytes > plan.free_bytes:
                self._set(keys=tuple(plan.fetch), **found)
                return w.WeightsError(
                    f"not enough disk space: {plan.fetch_bytes / 1e6:,.0f} MB needed, "
                    f"{plan.free_bytes / 1e6:,.0f} MB free under {root}"
                )
            run = await sync.run_own(plan, index, self)
            if run is None:
                continue
            self._set(state="fetching", keys=tuple(plan.fetch), error=None, retry_unix=None, **found)
            logger.info(
                "voice: fetching {} in the background ({:,.0f} MB); the channel reloads once {}",
                ", ".join(plan.fetch), plan.fetch_bytes / 1e6,
                "it lands" if len(plan.fetch) == 1 else "they land",
            )
            await asyncio.wait({run.task})  # type: ignore[arg-type]
            landed = await asyncio.to_thread(_landed, plan.fetch, root)
            if landed:
                logger.info("voice: fetched {} in the background", ", ".join(landed))
                self._set(reload=True)
                self.landed.set()
            if run.stopped:
                return None  # a Cancel, an Apply or a stop: whoever stopped it says what follows
            lacking = tuple(k for k in plan.fetch if k not in landed)
            if run.error is None:
                self._set(state="idle", keys=lacking, error=None, retry_unix=None)
                return None
            self._set(keys=lacking)
            return w.WeightsError(run.error, transient=run.transient)

    def _tell(self, held: list[str], unknown: list[str]) -> None:
        """Say once per key why the channel will not fetch it on its own."""
        for key in held:
            if key not in self._told:
                logger.info(
                    "voice: not fetching {} in the background: its license notice is not "
                    "accepted (accept it under Models in the WebUI, or list it in "
                    "channels.voice.models.acceptNotices)", key,
                )
        for key in unknown:
            if key not in self._told:
                logger.info("voice: {} is not in the model index, so nothing fetches it", key)
        self._told.update(held, unknown)

    def _log_failure(self, error: w.WeightsError, pause: float | None) -> None:
        repeat = str(error) == self._last_error
        self._last_error = str(error)
        then = (
            f"trying again in {pause / 60:g} min" if pause and pause >= 60
            else f"trying again in {pause:g} s" if pause
            else "not trying again: Apply in the WebUI or `nanobot-voice sync` fetches them"
        )
        (logger.info if repeat else logger.warning)(
            "voice: the background model fetch failed ({}); {}", error, then,
        )


def _landed(keys: list[str], root: Path) -> list[str]:
    have = w.installed(root)
    return [k for k in keys if k in have]
