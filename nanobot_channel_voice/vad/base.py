"""``Vad`` makes a per-frame speech/non-speech decision over S16_LE mono PCM frames of
the configured ``frame_ms``; ``Endpointer`` turns the decision stream into utterances,
so backends only answer "is this frame speech?"."""

from __future__ import annotations

import abc

from nanobot_channel_voice.audio.pcm import pcm_rms


class Vad(abc.ABC):
    # True when is_speech() is costly enough (neural inference) that the caller must
    # run it off the event loop.
    heavy: bool = False

    # Probability behind the latest is_speech() decision (the Endpointer aggregates it
    # per utterance). None on webrtc/energy; repeats across sub-window frames.
    last_prob: float | None = None

    # Loudness gate AND'd with the model decision; 0.0 = off. Set by the neural
    # engines from their own config block.
    _min_volume: float = 0.0

    @abc.abstractmethod
    def is_speech(self, frame: bytes) -> bool: ...

    def _gated(self, speech: bool, frame: bytes) -> bool:
        """Loudness AND'd with the model: distant TV speech is real speech to the model
        but too quiet to be the user. Apply to every RETURNED decision (held sub-window
        state included), never to the model run, or the streaming cache desyncs."""
        if speech and self._min_volume > 0.0:
            return pcm_rms(frame) >= self._min_volume
        return speech

    def reset(self) -> None:
        """Reset streaming state. Called after each utterance and before re-listening,
        so no state crosses a half-duplex mic gap."""

    def release(self) -> None:
        """Give back accelerator resources. Idempotent."""

    def scale_floor(self, factor: float) -> None:
        """Duck sync: the mic's acoustic leak drops by exactly the sink's duck gain, so
        an adaptive floor applies it instantly instead of re-converging over ~a second."""
