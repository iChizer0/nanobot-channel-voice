"""Optional OpenTelemetry export of the voice channel's TOOL CALLS.

Active only when ``channels.voice.telemetry.enabled`` is set and the [otel] extra
is installed; a no-op otherwise.

What crosses the process boundary (see ``REPORT-eval-methodology.md`` section 3.7):

* **Tool calls: exported, portably.** OTel GenAI semconv defines the span name
  ``execute_tool {gen_ai.tool.name}``, kind INTERNAL, with ``gen_ai.tool.name``,
  ``gen_ai.tool.call.id``, ``gen_ai.tool.type`` and ``error.type``.
  ``gen_ai.tool.call.arguments`` / ``.result`` are **Opt-In** in the spec on
  privacy grounds and stay off unless ``capture_content`` is set. Outcome / mode
  / staleness have no semconv equivalent and ride in the ``voice.*`` namespace.
* **Voice latency: NOT exported.** The semconv has no audio operation, no
  realtime/WebSocket session convention and no turn-taking / barge-in /
  endpointing / e2e-voice-latency metric (issue #304 is still open), so TTFA,
  think time, continuation and barge-in stay in ``VoiceMetrics``.

No TracerProvider, exporter or resource is created here: wiring the SDK is the
host application's job (as in LiveKit and Pipecat): a library installing a
global provider fights whatever the host configured.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

from loguru import logger

from nanobot_channel_voice.config import TelemetryConfig


class _NoopSpan:
    """Stands in for a real span so callers never branch on availability."""

    def set_attribute(self, key: str, value) -> None: ...
    def record_exception(self, exc: BaseException) -> None: ...
    def set_status_error(self, message: str) -> None: ...


_NOOP = _NoopSpan()


class _RealSpan:
    __slots__ = ("_span", "_status", "_status_cls")

    def __init__(self, span, status_cls, status_code) -> None:
        self._span = span
        self._status_cls = status_cls
        self._status = status_code

    def set_attribute(self, key: str, value) -> None:
        self._span.set_attribute(key, value)

    def record_exception(self, exc: BaseException) -> None:
        self._span.record_exception(exc)

    def set_status_error(self, message: str) -> None:
        self._span.set_status(self._status_cls(self._status.ERROR, message))


class VoiceTracer:
    """Thin facade over an OTel tracer, or a no-op when unavailable. Resolved once
    per channel at startup, so the hot path is an attribute check, not a
    try/except import per tool call."""

    def __init__(self, cfg: TelemetryConfig) -> None:
        self._cfg = cfg
        # Set unconditionally: the disabled path still satisfies every attribute read.
        self._tracer = None
        self._status_cls = None
        self._status_code = None
        self._span_kind = None
        self._ns = (cfg.namespace or "voice").rstrip(".")
        if not cfg.enabled:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode
        except ImportError:
            logger.bind(component="voice").warning(
                "telemetry.enabled is set but OpenTelemetry is not installed; "
                "metrics stay in-process (pip install 'nanobot-channel-voice[otel]')"
            )
            return
        self._tracer = trace.get_tracer("nanobot_channel_voice")
        self._status_cls = Status
        self._status_code = StatusCode
        self._span_kind = trace.SpanKind.INTERNAL
        logger.bind(component="voice").info(
            "voice telemetry enabled (OTel; content capture={})", cfg.capture_content
        )

    @contextmanager
    def tool_span(self, name: str, call_id: str, arguments: str | None = None):
        """One ``execute_tool`` span per tool call, per the GenAI semconv. Yields a
        span facade the caller stamps the outcome on; a body exception is recorded,
        then re-raised."""
        if self._tracer is None:
            yield _NOOP
            return
        # Span name `execute_tool {tool}` and operation `execute_tool` are both
        # mandated by the spec; deviating breaks platform auto-classification.
        with self._tracer.start_as_current_span(
            f"execute_tool {name}", kind=self._span_kind
        ) as raw:
            span = _RealSpan(raw, self._status_cls, self._status_code)
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", name)
            span.set_attribute("gen_ai.tool.call.id", call_id)
            span.set_attribute("gen_ai.tool.type", "function")
            if self._cfg.capture_content and arguments:
                span.set_attribute("gen_ai.tool.call.arguments", arguments)
            try:
                yield span
            except asyncio.CancelledError:
                # Teardown, not failure (the caller stamps outcome="cancelled");
                # marking ERROR would inflate the rate. Escapes unmarked:
                # use_span catches Exception, not BaseException.
                raise
            except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
                # Catches the BaseExceptions use_span ignores; an ordinary
                # exception is recorded here and once more by use_span on exit.
                span.record_exception(exc)
                span.set_status_error(type(exc).__name__)
                raise

    def tool_outcome(
        self, span, *, outcome: str, mode: str, stale: bool, result: str | None
    ) -> None:
        """Stamp a finished tool call's classification onto its span. ``error.type``
        is the semconv field a backend keys "this span failed" off, so it carries
        the OUTCOME, not just exceptions: a nanobot tool failure arrives as an
        ordinary return (``VoiceShell._on_tool_call``) and would export as ok."""
        span.set_attribute(f"{self._ns}.tool.outcome", outcome)
        span.set_attribute(f"{self._ns}.tool.mode", mode)
        # Stale = barged in while this ran: not an error, but wasted work to show.
        span.set_attribute(f"{self._ns}.tool.stale", stale)
        if outcome not in ("ok", "cancelled"):
            span.set_attribute("error.type", outcome)
            span.set_status_error(f"tool {outcome}")
        if self._cfg.capture_content and result is not None:
            span.set_attribute("gen_ai.tool.call.result", result)
