"""VoiceMetrics: quantile honesty, turn timeline latches, call lifecycle."""

from __future__ import annotations

from nanobot_channel_voice.metrics import VoiceMetrics


def test_quantiles_refuse_unsupported_sample_sizes():
    m = VoiceMetrics()
    for v in [10.0] * 9:
        m.observe("x_ms", v)
    s = m.snapshot()["latency_ms"]["x_ms"]
    assert s["n"] == 9 and s["p50"] == 10.0
    assert s["p90"] is None and s["p99"] is None  # 9 < 10 and < 1000
    m.observe("x_ms", 20.0)
    s = m.snapshot()["latency_ms"]["x_ms"]
    # Nearest-rank: ceil(10 * 0.9) = 9th order statistic, a value that
    # actually occurred (the ninth 10.0), never an interpolation.
    assert s["p90"] == 10.0 and s["max"] == 20.0
    assert s["p99"] is None


def test_negative_observations_are_dropped():
    m = VoiceMetrics()
    m.observe("x_ms", -5.0)
    assert "x_ms" not in m.snapshot()["latency_ms"]


def test_ttfa_is_latched_per_anchor():
    m = VoiceMetrics()
    m.turn_anchor()
    m.turn_first_audio()
    m.turn_first_audio()  # later frames of the same turn must not re-record
    assert m.snapshot()["latency_ms"]["ttfa_ms"]["n"] == 1


def test_unanchored_first_audio_counts_instead_of_recording():
    m = VoiceMetrics()
    m.turn_first_audio()
    assert m.counters["ttfa_unanchored"] == 1
    assert "ttfa_ms" not in m.snapshot()["latency_ms"]


def test_continuation_reanchors_to_a_separate_metric():
    m = VoiceMetrics()
    m.turn_anchor()
    m.turn_first_audio()          # ttfa
    m.turn_continuation()
    m.turn_first_audio()          # post-tool audio: continuation, not TTFA
    lat = m.snapshot()["latency_ms"]
    assert lat["ttfa_ms"]["n"] == 1
    assert lat["continuation_ms"]["n"] == 1


def test_anchor_backdating_never_goes_negative():
    m = VoiceMetrics()
    m.turn_anchor(offset_ms=-100.0)  # a caller bug must not move the anchor forward
    m.turn_first_audio()
    assert m.snapshot()["latency_ms"]["ttfa_ms"]["n"] == 1


def test_call_seen_is_idempotent():
    m = VoiceMetrics()
    m.call_seen("c1", "read_file")
    m.call_seen("c1", "read_file")  # provider re-announce
    assert m.counters["tool_calls"] == 1


def test_call_lifecycle_records_exec_by_mode():
    m = VoiceMetrics()
    m.call_seen("c1", "t")
    m.call_dispatched("c1", epoch=0)
    m.call_spawned("c1")
    m.call_finished("c1", outcome="ok", mode="direct")
    assert m.counters["tool_ok"] == 1
    assert m.snapshot()["latency_ms"]["tool_exec_ms.direct"]["n"] == 1
    assert m.snapshot()["inflight"] == 0


def test_call_stale_detects_epoch_change_and_counts():
    m = VoiceMetrics()
    m.call_seen("c1", "t")
    m.call_dispatched("c1", epoch=0)
    assert m.call_stale("c1", sink_epoch=0) is False
    assert m.call_stale("c1", sink_epoch=1) is True
    assert m.counters["tool_stale"] == 1


def test_calls_dropped_counts_only_still_open_calls():
    m = VoiceMetrics()
    m.call_seen("open", "t")
    m.call_seen("done", "t")
    m.call_spawned("done")
    m.call_finished("done", outcome="ok", mode="direct")
    assert m.calls_dropped({"open", "done", "never-seen"}, "session_lost") == 1
    assert m.counters["tool_dropped.session_lost"] == 1


def test_calls_abandoned_spares_already_dispatched():
    m = VoiceMetrics()
    m.call_seen("undispatched", "t")
    m.call_seen("dispatched", "t")
    m.call_dispatched("dispatched", epoch=0)
    m.calls_abandoned({"undispatched", "dispatched"})
    # The dispatched call is still answered later; only the other is dropped.
    assert m.counters["tool_dropped.cancelled"] == 1
    assert m.snapshot()["inflight"] == 1


def test_summary_line_mentions_ttfa_and_calls():
    m = VoiceMetrics()
    m.turn_anchor()
    m.turn_first_audio()
    m.call_seen("c1", "t")
    line = m.summary_line()
    assert "tool_calls=1" in line and "ttfa_p50=" in line


def test_has_data_true_for_latency_only_sessions():
    m = VoiceMetrics()
    assert not m.has_data
    m.observe("x_ms", 1.0)
    assert m.has_data  # a pure-conversation session still reports
