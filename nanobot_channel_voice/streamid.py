"""Parsing helpers for nanobot core's stream ids.

One home for the wire knowledge that a stream id is
``"<session_key>:<time_ns of turn start>[:<segment>]"`` (nanobot's AgentLoop):
the staleness guards key off the embedded start time, so an upstream format
change breaks one parser, not scattered heuristics. Degrades toward accepting
live turns: an id carrying no plausible ``time_ns`` yields ``None``, which
callers treat as "no verdict", never stale.
"""

from __future__ import annotations

# A time_ns tail is ~19 digits this century; 15+ rejects segment counters and
# ordinary ids while accepting any plausible nanosecond timestamp.
_MIN_NS_DIGITS = 15


def base_of(stream_id: str | None) -> str | None:
    """The turn-stable stream base: the id minus its trailing ``:<segment>``."""
    return stream_id.rsplit(":", 1)[0] if stream_id else None


def started_ns(stream_id: str | None) -> int | None:
    """The embedded turn-start ``time_ns``, or None when the id carries none.
    Accepts a full id (``base:segment``) or a bare base: the last few
    colon-separated parts are scanned for a plausible ns timestamp."""
    if not stream_id:
        return None
    for part in reversed(stream_id.rsplit(":", 2)):
        if part.isdigit() and len(part) >= _MIN_NS_DIGITS:
            return int(part)
    return None
