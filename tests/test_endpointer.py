"""Endpointer segmentation over a scripted VAD decision stream."""

from __future__ import annotations

from nanobot_channel_voice.config import VadConfig
from nanobot_channel_voice.vad import Endpointer, resolve_preroll_ms
from nanobot_channel_voice.vad.base import Vad

FRAME_MS = 20


class ScriptedVad(Vad):
    def __init__(self, decisions: list[bool], probs: list[float] | None = None):
        self._decisions = list(decisions)
        self._probs = list(probs) if probs is not None else None

    def is_speech(self, frame: bytes) -> bool:
        if self._probs:
            self.last_prob = self._probs.pop(0)
        return self._decisions.pop(0) if self._decisions else False


def frame(i: int) -> bytes:
    return bytes([i % 251]) * 4  # distinct, tiny frames


def make(decisions: list[bool], **kw) -> Endpointer:
    probs = kw.pop("probs", None)
    defaults = dict(
        frame_ms=FRAME_MS,
        start_frames=3,
        hangover_ms=100,       # 5 frames
        min_utterance_ms=60,   # 3 ACTIVE frames
        max_utterance_ms=1000,  # 50 frames
        preroll_ms=40,         # 2 frames of pre-trigger context
    )
    defaults.update(kw)
    return Endpointer(ScriptedVad(decisions, probs), **defaults)


def push_all(ep: Endpointer, n: int, start: int = 0):
    """Push frames start..start+n-1; collect the (utterance, closed_reason) pairs emitted."""
    outs = []
    for i in range(n):
        out = ep.push(frame(start + i))
        if out is not None:
            outs.append((out, ep.closed_reason))
    return outs


def test_onset_needs_start_frames_and_short_runs_die():
    ep = make([True, True, False, True])
    ep.push(frame(0)), ep.push(frame(1))
    assert not ep.in_speech and ep.speech_run == 2
    ep.push(frame(2))  # run dies
    assert ep.speech_run == 0
    ep.push(frame(3))
    assert not ep.in_speech  # a single frame never opens (start_frames=3)


def test_utterance_includes_preroll_and_closes_by_silence():
    # f0,f1 silence (pre-roll), f2..f4 speech (onset), f5..f9 silence (hangover)
    ep = make([False, False, True, True, True] + [False] * 5)
    outs = push_all(ep, 10)
    assert len(outs) == 1
    utterance, reason = outs[0]
    assert reason == "silence"
    assert utterance == b"".join(frame(i) for i in range(10))
    assert not ep.in_speech


def test_min_utterance_gates_on_active_frames_not_padding():
    # Onset of exactly 3 speech frames, then straight to hangover: active=3.
    ep = make([True, True, True] + [False] * 5, min_utterance_ms=100)  # need 5 active
    outs = push_all(ep, 8)
    assert outs == []  # blip rejected: internal/trailing silence must not count


def test_max_length_close_reports_max_reason():
    ep = make([True] * 60)
    outs = push_all(ep, 55)
    assert len(outs) == 1
    _, reason = outs[0]
    assert reason == "max"


def test_eager_snapshot_taken_once_per_silence_run():
    # eager at 40ms (2 frames) < hangover (5 frames)
    ep = make([True, True, True, False, False, False, False, False], eager_ms=40)
    ep.push(frame(0)), ep.push(frame(1)), ep.push(frame(2))  # onset
    assert ep.take_eager() is None
    ep.push(frame(3))
    assert ep.take_eager() is None       # silence_run=1 < eager mark
    ep.push(frame(4))                     # silence_run=2 == eager mark
    snap = ep.take_eager()
    assert snap is not None and snap.startswith(frame(0))
    assert ep.take_eager() is None        # consumed
    for i in range(5, 8):
        out = ep.push(frame(i))
    assert out is not None                # hangover completes the close


def _eager_snapshot_frames(eager_ms: int) -> list[int]:
    """Indices of the frames whose push left a snapshot behind (polled per frame:
    the hangover close resets and would swallow the evidence if polled at the end)."""
    ep = make([True, True, True] + [False] * 6, eager_ms=eager_ms)
    hits = []
    for i in range(9):
        ep.push(frame(i))
        if ep.take_eager() is not None:
            hits.append(i)
    return hits


def test_eager_disabled_at_or_past_hangover():
    assert _eager_snapshot_frames(40) == [4]   # control: a snapshot IS observable here
    assert _eager_snapshot_frames(100) == []   # eager mark == hangover (5 frames)
    assert _eager_snapshot_frames(200) == []   # eager mark past hangover: unreachable


def test_start_frames_override_halves_the_onset_bar():
    ep = make([True, True])
    ep.start_frames_override = 1
    ep.push(frame(0))
    assert ep.in_speech  # continuation hysteresis: one frame is enough


def test_reset_clears_streaming_state():
    ep = make([True, True, True])
    push_all(ep, 3)
    assert ep.in_speech
    ep.reset()
    assert not ep.in_speech and ep.speech_run == 0


# ---- resolve_preroll_ms -----------------------------------------------------

def test_preroll_floor_energy_engine():
    assert resolve_preroll_ms(VadConfig(), FRAME_MS) == 300  # config wins
    assert resolve_preroll_ms(VadConfig(preroll_ms=0), FRAME_MS) == 2 * FRAME_MS + 40


def test_preroll_floor_tracks_firered_smoothing():
    cfg = VadConfig.model_validate(
        {"engine": "firered", "prerollMs": 0, "firered": {"smoothFrames": 5}}
    )
    assert resolve_preroll_ms(cfg, FRAME_MS) == 25 + 4 * 10 + 80


# ---- consult tier (end-of-turn model) --------------------------------------
# consult_ms=40 = 2 frames of silence before the snapshot; hangover stays 5 frames.

def test_consult_snapshot_once_per_pause_with_audio_so_far():
    ep = make([False, False, True, True, True, False, False, False], consult_ms=40)
    for i in range(7):
        assert ep.push(frame(i)) is None
    snap = ep.take_consult()
    assert snap is not None
    gen, pcm = snap
    # Everything buffered at the mark: preroll f0,f1 + speech f2..f4 + silence f5,f6.
    assert pcm == b"".join(frame(i) for i in range(7))
    assert ep.take_consult() is None  # consumed; no re-snapshot within the same pause
    ep.push(frame(7))
    assert ep.take_consult() is None


def test_close_now_ends_the_utterance_early():
    ep = make([False, False, True, True, True, False, False], consult_ms=40)
    for i in range(7):
        assert ep.push(frame(i)) is None
    gen, _pcm = ep.take_consult()
    utterance = ep.close_now(gen)
    assert utterance == b"".join(frame(i) for i in range(7))
    assert ep.closed_reason == "eou"
    assert ep.closed_silence_ms == 40  # 2 silence frames, not the 100 ms hangover
    assert not ep.in_speech


def test_close_now_stale_when_speech_resumed_after_snapshot():
    ep = make([True, True, True, False, False, True, False], consult_ms=40)
    for i in range(7):
        assert ep.push(frame(i)) is None
    gen, _pcm = ep.take_consult()
    assert gen is not None
    # Speech resumed at f5 (bumping the active count): the verdict is about a pause
    # that no longer ends the utterance.
    assert ep.close_now(gen) is None
    assert ep.in_speech


def test_close_now_gen_guard_rejects_a_superseded_pause():
    ep = make(
        [True, True, True, False, False, True, True, False, False],
        consult_ms=40,
    )
    for i in range(5):
        ep.push(frame(i))
    gen1, _ = ep.take_consult()
    for i in range(5, 9):
        ep.push(frame(i))
    gen2, _ = ep.take_consult()
    assert gen2 == gen1 + 1
    assert ep.close_now(gen1) is None      # verdict about pause 1 cannot close pause 2
    assert ep.close_now(gen2) is not None  # the current pause's verdict can


def test_close_now_never_closes_a_blip_under_min_utterance():
    # min 3 ACTIVE frames; only 2 speech frames before the pause.
    ep = make([True, True, False, False] + [False] * 6,
              start_frames=2, min_utterance_ms=60, consult_ms=40)
    outs = []
    for i in range(5):
        out = ep.push(frame(i))
        if out:
            outs.append(out)
    snap = ep.take_consult()
    assert snap is not None
    gen, _pcm = snap
    assert ep.close_now(gen) is None  # under min: leave it to the hangover close
    for i in range(5, 10):
        out = ep.push(frame(i))
        if out:
            outs.append(out)
    assert outs == []  # the hangover close dropped it as a blip


def test_consult_disabled_when_at_or_past_hangover():
    ep = make([True, True, True] + [False] * 5, consult_ms=100)  # == hangover
    for i in range(8):
        ep.push(frame(i))
    assert ep.take_consult() is None


def test_natural_close_records_full_hangover_silence():
    ep = make([False, False, True, True, True] + [False] * 5)
    outs = push_all(ep, 10)
    assert len(outs) == 1
    assert ep.closed_silence_ms == 100


def test_close_now_before_eager_mark_flags_uncovered():
    """A model close firing before the final run's eager re-mark must flag that a
    still-valid eager task belongs to an EARLIER pause (it would truncate the
    utterance if handed off)."""
    # eager at 3 silence frames, consult at 2: the verdict can close at frame 2.
    ep = make([True, True, True, False, False] + [False] * 5,
              consult_ms=40, eager_ms=60)
    for i in range(5):
        assert ep.push(frame(i)) is None
    gen, _pcm = ep.take_consult()
    utterance = ep.close_now(gen)  # closes at 2 silence frames, before the eager mark
    assert utterance is not None
    assert ep.eager_covered is False


def test_close_now_at_or_past_eager_mark_is_covered():
    ep = make([True, True, True, False, False, False] + [False] * 4,
              consult_ms=40, eager_ms=40)
    for i in range(6):
        assert ep.push(frame(i)) is None  # 3 silence frames: past both marks
    gen, _pcm = ep.take_consult()
    assert ep.close_now(gen) is not None
    assert ep.eager_covered is True


def test_natural_close_is_always_covered():
    ep = make([True, True, True] + [False] * 5, eager_ms=40)
    outs = push_all(ep, 8)
    assert len(outs) == 1
    assert ep.eager_covered is True


def test_consult_snapshot_is_capped_to_the_model_window():
    ep = make([True, True, True, False, False], consult_ms=40, consult_cap_bytes=8)
    for i in range(5):
        ep.push(frame(i))
    _gen, pcm = ep.take_consult()
    assert len(pcm) == 8  # only the tail the model scores, not the whole buffer


def test_close_snapshots_active_ms_and_probability_stats():
    # Onset run 0.6/0.8/1.0, one in-speech speech frame 0.9, then hangover silence.
    ep = make(
        [True, True, True, True] + [False] * 5,
        probs=[0.6, 0.8, 1.0, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
    )
    outs = push_all(ep, 9)
    assert len(outs) == 1 and outs[0][1] == "silence"
    assert ep.closed_active_ms == 4 * FRAME_MS
    assert ep.closed_prob_peak == 1.0
    assert ep.closed_prob_mean == (0.6 + 0.8 + 1.0 + 0.9) / 4


def test_dead_candidate_probabilities_do_not_leak_into_the_next_utterance():
    # A two-frame run dies (0.95s), then the real utterance runs at 0.5.
    ep = make(
        [True, True, False] + [True] * 4 + [False] * 5,
        probs=[0.95, 0.95, 0.1] + [0.5] * 4 + [0.1] * 5,
    )
    outs = push_all(ep, 12)
    assert len(outs) == 1
    assert ep.closed_prob_peak == 0.5  # the dead run's 0.95 must not survive
    assert ep.closed_prob_mean == 0.5


def test_prob_stats_are_none_for_binary_vads():
    ep = make([True, True, True] + [False] * 5)  # ScriptedVad without probs
    outs = push_all(ep, 8)
    assert len(outs) == 1
    assert ep.closed_prob_peak is None and ep.closed_prob_mean is None
