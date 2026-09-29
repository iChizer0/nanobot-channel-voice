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


def test_recent_text_can_be_bounded_to_what_is_still_in_earshot(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(er.time, "monotonic", lambda: now[0])
    f = SelfEchoFilter()
    f.note_spoken("say hey nanobot to wake me")
    f.note_spoken("and here is the rest of it", hold_ms=6000.0)
    now[0] = 5.0  # the phrase went quiet 5 s ago; the tail is still playing
    assert "nanobot" in f.recent_text()
    assert "nanobot" not in f.recent_text(3.0)
    assert "the rest of it" in f.recent_text(3.0)


def test_reset_forgets_everything():
    f = SelfEchoFilter()
    f.note_spoken("something spoken")
    f.reset()
    assert f.is_self_echo("something spoken") is False


def test_cut_drops_the_unheard_tail_and_ends_the_hold(monkeypatch):
    """Playback stopped: the newest notes named as unheard go (a user repeating those
    words is fresh), a text with no units was never noted and is skipped, and what
    remains stops holding from a playout that never came."""
    now = [0.0]
    monkeypatch.setattr(er.time, "monotonic", lambda: now[0])
    f = SelfEchoFilter(threshold=0.5, window_secs=1.0)
    f.note_spoken("alpha beta gamma", hold_ms=5000.0)
    f.note_spoken("...", hold_ms=5000.0)
    f.note_spoken("delta epsilon zeta", hold_ms=5000.0)
    f.cut(["...", "delta epsilon zeta"])
    assert f.is_self_echo("delta epsilon zeta") is False
    assert f.is_self_echo("alpha beta gamma") is True
    f.cut(["never spoken words"])  # a mismatch stops the walk: nothing else goes
    assert f.is_self_echo("alpha beta gamma") is True
    now[0] = 1.5  # the hold ended at the cut (t=0): the window has run out
    assert f.is_self_echo("alpha beta gamma") is False


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
    # A pure echo yields no fresh evidence, across the seam as within a piece.
    assert not f.fresh_words("今天天气很好适合出门")
    # Nor with three clauses in one piece, no punctuation heard.
    f = SelfEchoFilter()
    f.note_spoken("今天天气很好，最高气温二十度，适合出门。")
    assert not f.fresh_words("今天天气很好最高气温二十度适合出门")


def test_a_fragment_across_a_mark_or_a_seam_is_ours():
    """A short tail heard across a clause mark, a piece seam or a read-out number at a
    mark is echo, not half fresh; a number between two characters still parts them."""
    f = SelfEchoFilter()
    f.note_spoken("今天天气很好，适合出门散步，晚上会下雨。")
    assert f.is_self_echo("步晚上")
    f.note_spoken("明天转晴。")
    assert f.is_self_echo("雨明天")
    f = SelfEchoFilter()
    for piece in ("今天天气很好，", "适合出门，", "晚上会下雨。"):
        f.note_spoken(piece)
    assert f.is_self_echo("门晚上")  # the seam is with the piece just before
    f = SelfEchoFilter()
    f.note_spoken("好的。")
    f.note_spoken("20度以上。")  # starts with a number: read out, it joins the seam
    assert f.is_self_echo("的二十")
    f = SelfEchoFilter()
    f.note_spoken("气温是20，适合出门。")
    assert f.is_self_echo("十适合")
    f = SelfEchoFilter()
    f.note_spoken("气温是20，")  # the number ends the piece: read out, it runs on
    f.note_spoken("适合出门。")
    assert f.is_self_echo("十适合")
    f = SelfEchoFilter()
    f.note_spoken("气温20度。")
    assert f.fresh_words("温度") == {"温度"}


def test_one_misheard_character_is_no_fresh_evidence():
    """STT on our own leak swaps, drops or adds a character now and then: the reply
    explains it anywhere in the run but at its very edges, where it is one unit."""
    for pieces, heard in [
        (["今天天气很好，最高气温二十度。"], "今天天气很好最高气温二十度"),
        (["今天天气很好，", "最高气温二十度。"], "今天天气很好最高气温二十度"),  # per clause
        (["明天最高气温二十度。"], "明天最高气温二十度"),  # no character repeats at the start
    ]:
        f = SelfEchoFilter()
        for piece in pieces:
            f.note_spoken(piece)
        for i in range(len(heard) + 1):
            if i < len(heard):
                swapped, dropped = heard[:i] + "器" + heard[i + 1 :], heard[:i] + heard[i + 1 :]
                assert len(f.fresh_words(swapped)) <= (i in (0, len(heard) - 1)), swapped
                assert not f.fresh_words(dropped), dropped
            added = heard[:i] + "器" + heard[i:]
            assert len(f.fresh_words(added)) <= (i in (0, len(heard))), added
    f = SelfEchoFilter()
    f.note_spoken("今天天气很好，最高气温二十度。")
    assert not f.fresh_words("今天天器很好最高汽温二十度")  # two, far apart
    assert not f.fresh_words("今天天器很号最高气温二十度")  # two, one apart
    g = SelfEchoFilter()
    g.note_spoken("明天最高气温25度。")
    assert not g.fresh_words("明天最高气温二十无度")  # in a number read out
    g = SelfEchoFilter()
    g.note_spoken("我们先吃饭，我们明天去爬山。")
    assert not g.fresh_words("我们朋天去爬山")  # its context is ours at any occurrence
    g = SelfEchoFilter()
    g.note_spoken("今天天气很好。")
    assert not g.fresh_words("天器很")  # a run too short for three a side
    # What we never said still counts, run into our words or not.
    assert f.fresh_words("今天天气很好等一下") == {"好等", "等一", "一下"}


def test_our_words_in_a_new_order_stay_fresh():
    """A correction made of the reply's own words: the bigrams joining them were never
    said, and no misheard character explains them ("是一个" is one character a side
    from 是这个, too little to call the 这 misheard)."""
    f = SelfEchoFilter()
    f.note_spoken("这个问题不是很难，就是一个小问题，意思是明天会下雨。")
    assert f.fresh_words("不是这个意思") == {"是这", "个意"}


def test_a_stop_word_is_never_explained_away():
    """Said into our own words, a stop keeps its evidence and reads as the stop alone;
    a backchannel there is a misheard character like any other."""
    from nanobot_channel_voice.echo_reject import units_of

    f = SelfEchoFilter(protect=units_of("stop 停 闭嘴 别说了"))
    f.note_spoken("今天天气很好，最高气温二十度。")
    for heard, fresh, stretches in [
        ("今天天气很好停最高气温二十度", {"好停", "停最"}, ["停"]),
        ("今天天气很好最高气温二十度停", {"度停"}, ["停"]),
        ("今天天气很好闭嘴最高气温二十度", {"好闭", "闭嘴", "嘴最"}, ["闭嘴"]),
        ("今天stop最高", {"stop"}, ["stop"]),  # glued on by a bilingual STT
        ("停最高气温二十度", {"停最"}, ["停"]),  # the stop leads the run
        ("今天天气很好嗯最高气温二十度", set(), []),
    ]:
        assert f.fresh_words(heard) == fresh, heard
        assert f.fresh_readings(heard, fresh)[-1] == stretches, heard
    f.note_spoken("这件事我说了算。")
    assert f.fresh_words("别说了") == {"别说"}  # 说了 is ours; the stop still holds
    assert f.fresh_readings("别说了", {"别说"}) == [["别说了"]]
    f = SelfEchoFilter(protect=units_of("停"))
    f.note_spoken("请暂停播放音乐。")
    assert not f.fresh_words("请暂停拨放音乐")  # our own stop word beside a misheard one
    f = SelfEchoFilter(protect=units_of("停"))
    f.note_spoken("请打开WiFi设置。")
    assert f.fresh_readings("打开wifi设置停", {"置停"})[-1] == ["停"]  # our Latin stays out
    # Nor is a stop's character the misheard one, nor did it lose a neighbour we said.
    for spoken, heard, fresh, stretches in [
        ("天气很好就说了这么多。", "天气很好别说了这么多", {"好别", "别说"}, ["别说"]),
        ("天气很好，暂停一下。", "天气很好停", {"好停"}, ["停"]),
        ("请稍等一下，今天天气很好。", "等一下今天天气很好停", {"好停"}, ["停"]),
    ]:
        f = SelfEchoFilter(protect=units_of("停 别说了 等一下"))
        f.note_spoken(spoken)
        assert f.fresh_words(heard) == fresh, heard
        assert f.fresh_readings(heard, fresh)[-1] == stretches, heard


def test_no_bridge_makes_a_stop_word_ours():
    """Bigrams across a clause mark, and a number read out, are ours for echo, never
    for a stop."""
    from nanobot_channel_voice.echo_reject import units_of

    for spoken, heard, unit in [
        ("もう少し待ちましょう。いい天気ですね。", "もういい", "うい"),
        ("推荐北京、上海、广州等，一共三个城市。", "等一下", "等一"),
        ("好的，停一下。", "停", "停"),  # its reading 停1下 holds 停 on its own
    ]:
        f = SelfEchoFilter(protect=units_of(heard))
        f.note_spoken(spoken)
        assert unit in f.fresh_words(heard), heard


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
        ("气温是20，适合出门。", "气温是二十适合出门"),  # read out, run on past the mark
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


def test_fresh_readings_bridge_units_to_lexicon_tokens():
    f = SelfEchoFilter()
    f.note_spoken("正在为你播放音乐。")
    text = "别说了"  # fused zh stop said through the leak
    fresh = f.fresh_words(text)
    assert fresh  # its bigrams are not the bot's words
    # The fused token survives WHOLE, so PhraseMatcher can segment it as a stop.
    assert f.fresh_readings(text, fresh) == [["别说了"]]
    # Run into our words, the whole token is one reading and its fresh stretch another.
    fresh = f.fresh_words("正在播放别说了")
    assert f.fresh_readings("正在播放别说了", fresh) == [["正在播放别说了"], ["别说了"]]
