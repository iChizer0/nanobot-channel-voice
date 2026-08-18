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


def test_units_of_uses_cjk_bigrams():
    from nanobot_channel_voice.echo_reject import units_of

    assert units_of("Hello world") == {"hello", "world"}
    assert units_of("今天天气") == {"今天", "天天", "天气"}
    assert units_of("好") == {"好"}                    # lone CJK char: unigram
    assert units_of("WiFi密码") == {"wifi", "密码"}    # mixed run splits by script


def test_cjk_echo_is_detected_whatever_the_punctuation():
    f = SelfEchoFilter()
    f.note_spoken("今天天气很好，")   # the chunker feeds per clause
    f.note_spoken("适合出门。")
    assert f.is_self_echo("今天天气很好适合出门")       # streaming STT: no punct
    assert f.is_self_echo("今天天气很好。适合。出门。")  # different punct: still echo
    assert not f.is_self_echo("不要说了换个话题")        # genuine speech passes
    # A pure echo yields (almost) no fresh evidence: only the chunk-seam bigram.
    assert len(f.fresh_words("今天天气很好适合出门")) <= 1


def test_zh_wake_phrase_containment_for_the_echo_veto():
    f = SelfEchoFilter()
    f.note_spoken("只要说小助手就能唤醒我。")
    assert not f.fresh_words("小助手")  # fully contained -> the veto fires
    assert f.fresh_words("关灯")        # a different phrase stays fresh


def test_number_reading_variance_still_matches():
    # STT renders spoken digits per ITS normalization, not the TTS text's: a
    # character-output zh model hears 7点45分 as 七点四十五分, a word-output en
    # model hears 7:45 as "seven forty five", ITN goes the other way.
    for spoken, heard in [
        ("现在是7点45分。", "现在是七点四十五分"),
        ("温度是23.5度。", "温度是二十三点五度"),
        ("房间号是404。", "房间号是四零四"),          # digitwise reading
        ("还有四十五分钟。", "还有45分钟"),           # reverse: ITN STT
        ("It is 7:45 now.", "it is seven forty five now"),
    ]:
        f = SelfEchoFilter()
        f.note_spoken(spoken)
        assert f.is_self_echo(heard), (spoken, heard)
        assert not f.fresh_words(heard), (spoken, heard)


def test_latin_respacing_still_matches():
    # STT respaces/hyphenates Latin runs differently than the TTS text: units
    # bridge via substring-of-the-latin-stream, both split and fused directions.
    for spoken, heard in [
        ("请打开WiFi设置。", "请打开Wi-Fi设置"),
        ("请打开WiFi设置。", "请打开WI FI设置"),
        ("请打开WiFi设置。", "请打开ＷｉＦｉ设置"),   # fullwidth folds via NFKC
        ("play some music", "playsomemusic"),
    ]:
        f = SelfEchoFilter()
        f.note_spoken(spoken)
        assert f.is_self_echo(heard), (spoken, heard)
        assert not f.fresh_words(heard), (spoken, heard)
    # Genuine distinct words never absorb: "why fight" is no substring of "wifi".
    f = SelfEchoFilter()
    f.note_spoken("请打开WiFi设置。")
    assert not f.is_self_echo("why fight")


def test_protected_units_stay_fresh_through_absorption():
    from nanobot_channel_voice.echo_reject import units_of

    # "stop" hides inside spoken "unstoppable": scoring may absorb it (echo
    # containment), but fresh evidence keeps it, so the stop override upstream
    # still sees the kill switch.
    f = SelfEchoFilter(protect=units_of("stop"))
    f.note_spoken("unstoppable progress")
    assert f.fresh_words("stop") == {"stop"}
    # Exactly-spoken protected words still subtract: no false stop on echo.
    f.note_spoken("please stop doing that")
    assert "stop" not in f.fresh_words("please stop doing that")


def test_fresh_seq_bridges_units_to_lexicon_tokens():
    from nanobot_channel_voice.backend.local import LocalBackend

    f = SelfEchoFilter()
    f.note_spoken("正在为你播放音乐。")
    text = "别说了"  # fused zh stop said through the leak
    fresh = f.fresh_words(text)
    assert fresh  # its bigrams are not the bot's words
    # The fused token survives WHOLE, so PhraseMatcher can segment it as a stop.
    assert LocalBackend._fresh_seq(text, fresh) == ["别说了"]
