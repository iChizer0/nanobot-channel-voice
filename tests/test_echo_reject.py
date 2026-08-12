"""SelfEchoFilter: containment verdicts and audibility-stamped eviction."""

from __future__ import annotations

import nanobot_channel_voice.echo_reject as er
from nanobot_channel_voice.echo_reject import SelfEchoFilter, words_of


def test_words_of_lowercases_and_tokenizes_unicode():
    assert words_of("Hello, WORLD! 你好") == {"hello", "world", "你好"}
    assert words_of("...") == set()


def test_containment_threshold():
    f = SelfEchoFilter(threshold=0.6)
    f.note_spoken("the quick brown fox jumps")
    assert f.is_self_echo("the quick brown") is True          # 3/3 contained
    assert f.is_self_echo("a completely different sentence") is False
    # Mixed: 2 of 4 words known -> 0.5 < 0.6 -> passes through as user speech.
    assert f.is_self_echo("the quick zebra dances") is False


def test_fresh_words_subtracts_spoken_only():
    f = SelfEchoFilter()
    f.note_spoken("turn on the light")
    assert f.fresh_words("turn the light off now") == {"off", "now"}
    assert f.fresh_words("") == set()


def test_empty_states_never_match():
    f = SelfEchoFilter()
    assert f.is_self_echo("anything") is False  # nothing spoken yet
    f.note_spoken("words")
    assert f.is_self_echo("") is False


def test_eviction_runs_from_audibility_not_feed_time(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(er.time, "monotonic", lambda: now[0])
    f = SelfEchoFilter(threshold=0.5, window_secs=1.0)
    f.note_spoken("early words here", hold_ms=0.0)       # audible at t=0
    f.note_spoken("later words spoken", hold_ms=2000.0)  # audible until ~t=2
    now[0] = 1.5  # past the window for the first entry only (0 + 1.0 < 1.5)
    assert f.is_self_echo("early words here") is False    # evicted
    assert f.is_self_echo("later words spoken") is True   # hold_ms kept it alive


def test_reset_forgets_everything():
    f = SelfEchoFilter()
    f.note_spoken("something spoken")
    f.reset()
    assert f.is_self_echo("something spoken") is False
