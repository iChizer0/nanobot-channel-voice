"""Notification cues (``earcons.*``): the receipt and attention tones, or the operator's
own short WAVs, shaped once at build time."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from loguru import logger

from nanobot_channel_voice.audio.pcm import fade_tail_pcm, pcm_ms, resample_pcm, ring_ms, wav_pcm

from .audio_sink import scale_pcm, trim_lead_silence, trim_tail_silence

# Past this a cue is a jingle: it gates the half-duplex mic and delays what queues behind it.
CUE_MAX_MS = 600
_CUE_MAX_FILE_B = 2_000_000  # refuse absurd files unread: ~10 s of 48 k stereo already


class Cue(NamedTuple):
    pcm: bytes  # S16 mono
    rate: int
    # A built-in tone's ring past its body: too faint to open an onset, so it needs no
    # guard of its own. 0.0 for a custom clip, whose tail may be speech.
    ring_s: float


def cue_pcm(
    path: str | None,
    synth: Callable[[int], bytes],
    rate: int,
    *,
    keep_file_rate: bool = False,
    gain_db: float = 0.0,
) -> Cue:
    """One cue. A custom WAV wins, resampled to ``rate`` unless ``keep_file_rate`` (blob
    playback follows the header); an unusable file degrades loudly to ``synth``'s built-in.
    Edge-trim runs BEFORE the length cap, so padding spends no budget; a real cut fades."""
    pcm, out_rate = b"", rate
    if path:
        try:
            size = Path(path).stat().st_size
            if size > _CUE_MAX_FILE_B:
                raise ValueError(f"{size / 1e6:.1f} MB; a cue asset should be tiny")
            src, src_rate = wav_pcm(Path(path).read_bytes())
            if not src:
                raise ValueError("not a readable S16 WAV")
            if keep_file_rate:
                out_rate = src_rate
            else:
                src = resample_pcm(src, src_rate, rate)
            src = trim_lead_silence(src, out_rate, cap_ms=20.0)
            src = trim_tail_silence(src, out_rate, cap_ms=120.0)
            cap = int(out_rate * CUE_MAX_MS / 1000) * 2
            if len(src) > cap:
                logger.warning(
                    "voice: earcon '{}' is {:.0f} ms; truncating to {} ms "
                    "(a cue must stay short)",
                    path, pcm_ms(len(src), out_rate), CUE_MAX_MS,
                )
                src = fade_tail_pcm(src[:cap], out_rate)
            pcm = src
        except Exception as exc:  # noqa: BLE001 - degrade loudly, never mute
            logger.warning("voice: earcon file '{}' unusable ({}); using the built-in", path, exc)
            pcm, out_rate = b"", rate
    ring_s = 0.0
    if not pcm:
        pcm = synth(rate)
        ring_s = ring_ms(pcm, rate) / 1000.0
    if gain_db:
        pcm = scale_pcm(pcm, 10.0 ** (gain_db / 20.0))
    return Cue(pcm, out_rate, ring_s)


__all__ = ["CUE_MAX_MS", "Cue", "cue_pcm"]
