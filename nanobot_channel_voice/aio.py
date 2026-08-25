"""Tiny asyncio/timing helpers shared across the plugin: one definition each,
so callers cannot drift in which edge cases they tolerate."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any


def cancel_task(task: asyncio.Task | None) -> None:
    if task is not None and not task.done():
        task.cancel()


async def cancel_and_wait(task: asyncio.Task | None) -> None:
    """Cancel ``task`` and await its exit, swallowing cancellation AND failure:
    a teardown caller has nowhere to send a dying task's exception. Transparent
    to the CALLER's own cancellation: a CancelledError aimed at the awaiting
    task is re-raised rather than mistaken for the child's, because nanobot's
    ChannelManager cancels ``channel.stop()`` from above."""
    if task is None:
        return
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        cur = asyncio.current_task()
        if cur is not None and cur.cancelling():
            raise
    except Exception:  # noqa: BLE001
        pass


async def wait_until(deadline: Callable[[], float]) -> None:
    """Sleep until the MOVING ``deadline()`` (``time.monotonic`` domain) passes: recomputed
    on every wake, so one waiter can watch several clocks (the ``min`` of their deadlines)."""
    while True:
        now = time.monotonic()
        due = deadline()
        if now >= due:
            return
        await asyncio.sleep(due - now)


async def wait_for_stall(last_activity: Callable[[], float], budget_s: float) -> None:
    """Sleep until ``budget_s`` has passed since the MOVING ``last_activity()``
    stamp (``time.monotonic`` domain): activity pushes the stamp forward, so only
    a full budget of silence returns. A deadman, not a cap: the caller decides
    what recovery means."""
    await wait_until(lambda: last_activity() + budget_s)


class Throttle:
    """Rate-limit a repeating warning to once per ``interval_s``. :meth:`ready`
    latches the clock only when it returns True, so suppressed calls never push
    the window forward; the first call is always ready."""

    __slots__ = ("_interval", "_last")

    def __init__(self, interval_s: float = 30.0):
        self._interval = interval_s
        self._last = float("-inf")

    def ready(self) -> bool:
        now = time.monotonic()
        if now - self._last < self._interval:
            return False
        self._last = now
        return True


def put_drop_oldest(q: asyncio.Queue, item: Any) -> Any | None:
    """Non-blocking put that DROPS THE OLDEST queued item on overflow, keeping
    the queue near real time. Returns the dropped item (``task_done`` already
    called), or None. Event-loop side only: asyncio.Queue is not thread-safe."""
    dropped = None
    if q.full():
        with suppress(asyncio.QueueEmpty):
            dropped = q.get_nowait()
            q.task_done()
    with suppress(asyncio.QueueFull):
        q.put_nowait(item)
    return dropped
