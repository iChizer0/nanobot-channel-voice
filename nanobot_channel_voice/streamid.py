"""Turn/delegation identity: parse core's stream ids, mint our own tokens.

One home for the wire knowledge that a stream id is
``"<session_key>:<time_ns of turn start>[:<segment>]"`` (nanobot's AgentLoop):
the staleness guards key off the embedded start time, so an upstream format
change breaks one parser, not scattered heuristics. Degrades toward accepting
live turns: an id carrying no plausible ``time_ns`` yields ``None``, which
callers treat as "no verdict", never stale.
"""

from __future__ import annotations

import itertools
import time

# A time_ns tail is ~19 digits this century; 15+ rejects segment counters and
# ordinary ids while accepting any plausible nanosecond timestamp.
_MIN_NS_DIGITS = 15

_TOKEN_SEQ = itertools.count()


def unique_token() -> str:
    """Mint a process-unique opaque token (the plugin's turn/delegation identities,
    compared by equality only). Wall-clock ns alone is NOT unique: two mints inside
    one clock quantum (~1 us on macOS) share a timestamp, and an identity collision
    lets a dead turn's staleness gate swallow a live turn's reply. The sequence tail
    removes that possibility."""
    return f"{time.time_ns()}-{next(_TOKEN_SEQ)}"


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
