"""``Vad`` makes a per-frame speech/non-speech decision over S16_LE mono PCM frames of
the configured ``frame_ms``; ``Endpointer`` turns the decision stream into utterances,
so backends only answer "is this frame speech?"."""

from __future__ import annotations

import abc


class Vad(abc.ABC):
    # True when is_speech() is costly enough (neural inference) that the caller must
    # run it off the event loop; False keeps the default path free of thread-hop overhead.
    heavy: bool = False

    @abc.abstractmethod
    def is_speech(self, frame: bytes) -> bool: ...

    def reset(self) -> None:
        """Reset any streaming state. The ``Endpointer`` calls this after each utterance
        and before re-listening, so no state crosses a half-duplex mic gap."""

    def release(self) -> None:
        """Give back accelerator resources. Idempotent; only the neural backend holds
        an NPU/ORT context."""

    def scale_floor(self, factor: float) -> None:
        """Duck synchronization: when the sink ducks playback by a known gain, the
        acoustic leak the mic hears drops by exactly that factor, so an adaptive floor
        can apply it INSTANTLY instead of re-converging over ~a second."""
