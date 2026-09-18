"""SentenceChunker + sanitize: cut rules and the incremental fence machine."""

from __future__ import annotations

from nanobot_channel_voice.chunker import SentenceChunker, sanitize


def collect(chunker: SentenceChunker, *deltas: str) -> list[str]:
    out: list[str] = []
    for d in deltas:
        out.extend(chunker.feed(d))
    tail = chunker.flush()
    if tail:
        out.append(tail)
    return out


# ---- sanitize ---------------------------------------------------------------

def test_sanitize_strips_markdown():
    assert sanitize("**bold** and _em_ and `code`") == "bold and em and code"
    assert sanitize("[text](http://x) ![img](http://y)").strip() == "text"
    assert sanitize("# Header\n> quote\n- bullet") == "Header\nquote\nbullet"


def test_sanitize_drops_bracketed_placeholders_with_their_content():
    # The whitelist alone would drop only the brackets and SPEAK the words: a
    # clock-less model answered date questions with a literal "[Current Date and
    # Time]".
    assert sanitize("[Current Date and Time]").strip() == ""
    assert sanitize("It is [Current Date] today.").strip() == "It is today."
    assert sanitize("现在是【当前时间】。").strip() == "现在是 。"
    assert sanitize("time is {current_time} now").strip() == "time is now"
    assert sanitize("today: <date>").strip() == "today:"
    # A parroted per-turn note is the same failure through a different door.
    assert sanitize("[time now: 2026-08-19 (Wednesday) 21:33, UTC-07:00]").strip() == ""
    # Markdown links resolved first keep their text; CJK quotes carry real speech.
    assert sanitize("see [docs](http://x)").strip() == "see docs"
    assert sanitize("他说「现在八点」了").strip() == "他说「现在八点」了"
    assert sanitize("a < b and c > d") == "a b and c d"  # <,> whitelisted out, words kept


def test_sanitize_smart_quotes_fold_before_whitelist():
    assert sanitize("“Hi” ‘there’") == '"Hi" \'there\''


def test_sanitize_unspeakable_becomes_space_never_glue():
    assert sanitize("A→B") == "A B"
    assert sanitize("Hi \U0001f44b there") == "Hi there"


def test_sanitize_keeps_the_single_codepoint_degree_units():
    from nanobot_channel_voice.tts.text_frontend import verbalize_numbers_en, verbalize_numbers_zh

    # ℃/℉ were not whitelisted like °, so the unit vanished before any frontend ran.
    assert sanitize("今天25℃，很热") == "今天25℃，很热"
    assert verbalize_numbers_zh(sanitize("今天25℃，很热")) == "今天二十五摄氏度，很热"
    assert sanitize("It is 77℉ outside") == "It is 77℉ outside"
    assert verbalize_numbers_en(sanitize("It is 77℉ outside")) == (
        "It is seventy seven degrees Fahrenheit outside"
    )


def test_sanitize_folds_the_math_minus_to_a_hyphen():
    from nanobot_channel_voice.tts.text_frontend import verbalize_numbers_en, verbalize_numbers_zh

    # U+2212 was dropped as unspeakable: "−5°C" became " 5°C", a sign flip. Folded to
    # "-" so every downstream pass (sign, range) sees the one minus it knows.
    assert sanitize("−5°C") == "-5°C"
    assert verbalize_numbers_en(sanitize("−5°C")) == "minus five degrees Celsius"
    assert verbalize_numbers_zh(sanitize("−5°C")) == "零下五摄氏度"
    assert verbalize_numbers_zh(sanitize("5−10分钟")) == "五到十分钟"


def test_sanitize_folds_fullwidth_percent_yen_and_hyphen():
    from nanobot_channel_voice.tts.text_frontend import verbalize_numbers_zh

    # ％/￥/－ were not whitelisted: "增长5％。" lost its percent, "（￥450）" its currency,
    # and "5－10分钟" fused into 五十分钟. Folded to the twins the frontends read.
    assert sanitize("增长5％。") == "增长5%。"
    assert verbalize_numbers_zh(sanitize("增长5％。")) == "增长百分之五。"
    assert sanitize("（￥450）") == "（¥450）"
    assert verbalize_numbers_zh(sanitize("（￥450）")) == "（四百五十元）"
    assert sanitize("5－10%") == "5-10%"
    assert verbalize_numbers_zh(sanitize("5－10分钟")) == "五到十分钟"


def test_sanitize_strips_emphasis_only_at_word_edges():
    # An intraword marker is not emphasis: "_case_" inside snake_case_name was stripped
    # as a pair, gluing the words. Left over, a marker becomes a space, never nothing.
    assert sanitize("snake_case_name") == "snake case name"
    assert sanitize("file_name_here.txt") == "file name here.txt"
    assert sanitize("2*3*4") == "2 3 4"
    assert sanitize("a*b*c") == "a b c"
    # Pairs at word edges still strip, CJK glue included ("**重要**的" is how zh emphasis
    # is written), and a dunder is a pair at both edges.
    assert sanitize("**bold** text, *em*, ~~gone~~.") == "bold text, em, gone."
    assert sanitize("**重要**的事情") == "重要的事情"
    assert sanitize("__init__") == "init"


# ---- cut rules --------------------------------------------------------------

def test_sentence_cut_needs_following_separator():
    c = SentenceChunker(min_chars=60, max_chars=240)
    # "3.14" must not split; the terminator cuts once a space follows.
    assert c.feed("Pi is 3.14 exactly. And") == ["Pi is 3.14 exactly."]


def test_clause_cut_skips_grouped_number_commas():
    c = SentenceChunker(min_chars=6, max_chars=240)
    # The floor lands inside 1,902,567,338: digit-flanked commas are number
    # punctuation, so the cut waits for the real clause boundary after 美元.
    chunks = collect(c, "总收入达到了1,902,567,338美元，", "非常可观。")
    assert chunks[0].endswith("美元，")
    assert "1,902,567,338" in chunks[0]


def test_number_comma_at_delta_boundary_waits():
    c = SentenceChunker(min_chars=6, max_chars=240)
    # A delta ending "…1,902," is ambiguous: the comma must not cut until the
    # next delta shows whether a digit follows.
    assert c.feed("总收入达到了1,902,") == []
    chunks = c.feed("567,338美元，好的。")
    assert any("1,902,567,338" in ch for ch in chunks)


def test_terminator_at_buffer_end_waits_for_next_delta():
    # A '.' as the last buffered char can't prove it's a sentence end yet.
    c = SentenceChunker(min_chars=60, max_chars=240)
    assert c.feed("Hello.") == []
    assert c.feed(" More") == ["Hello."]


def test_cjk_terminator_stands_alone():
    # No following separator needed (unlike '.'), but at the buffer end it holds
    # one delta: a closing 」/" may still arrive and belongs to this sentence.
    c = SentenceChunker(min_chars=60, max_chars=240)
    assert c.feed("你好。") == []
    assert c.feed("再见") == ["你好。"]
    assert c.flush() == "再见"
    c = SentenceChunker(min_chars=60, max_chars=240)
    assert c.feed("你好。再见。") == ["你好。"]  # mid-buffer: cuts without waiting
    assert c.flush() == "再见。"


def test_closers_travel_with_their_sentence():
    # Cutting at the terminator would orphan a silent 」/" at the next chunk's head.
    c = SentenceChunker(min_chars=6, max_chars=240)
    chunks = []
    for delta in ("他说「你好", "。」", "然后走了。"):
        chunks += c.feed(delta)
    assert chunks == ["他说「你好。」"]
    assert c.flush() == "然后走了。"
    c = SentenceChunker(min_chars=6, max_chars=240)
    assert c.feed('He said "stop." Next.') == ['He said "stop."']
    # A CJK terminator run is one boundary, not a cut plus an orphaned "！" blip.
    c = SentenceChunker(min_chars=4, max_chars=240)
    assert c.feed("什么？！走吧。x") == ["什么？！", "走吧。"]


def test_first_chunk_floor_cuts_earlier_then_steady_floor_applies():
    c = SentenceChunker(min_chars=60, max_chars=240, min_chars_first=10)
    # First chunk: the clause comma at/after the 10-char floor cuts early.
    assert c.feed("A tiny clause, ") == ["A tiny clause,"]
    # Steady state: the same shape now buffers (comma before min_chars=60).
    assert c.feed("Another bit, ") == []


def test_punct_only_chunk_keeps_the_first_chunk_floor():
    c = SentenceChunker(min_chars=60, max_chars=240, min_chars_first=10)
    # An emoji-only sentence sanitizes to bare punctuation: emitted, but not speech —
    # the clause after it still cuts at min_chars_first, not min_chars.
    assert c.feed("🎉！") == []  # terminator at the buffer end waits for the delta
    assert c.feed("A tiny clause, ") == ["！", "A tiny clause,"]


def test_ordered_list_markers_never_become_chunks():
    c = SentenceChunker(min_chars=60, max_chars=240, min_chars_first=24)
    # The marker's "." looked like a sentence end: every item cost a "1." chunk of its
    # own (a synthesis call, a seam, and sentence-final prosody on the digit). Voiced as
    # "one," instead: the number stays, since "42. That is..." is indistinguishable.
    assert collect(c, "Here is the plan.\n", "1. Buy the milk.\n", "2. Call the plumber.\n") == [
        "Here is the plan.", "1, Buy the milk.", "2, Call the plumber.",
    ]
    # A reply that OPENS with a list: its first chunk is what TTFA latches on.
    c = SentenceChunker(min_chars=60, max_chars=240, min_chars_first=24)
    assert collect(c, "1. Check the oil.\n2. Top up the coolant.\n")[0] == "1, Check the oil."
    # A sentence that opens with a number is not a list item and loses nothing.
    c = SentenceChunker(min_chars=6, max_chars=240, min_chars_first=6)
    assert collect(c, "42. That is the answer to everything.") == [
        "42, That is the answer to everything.",
    ]


def test_a_bare_number_sentence_keeps_its_digits():
    # The ordered marker needs a SAME-LINE space after it, so "42." stays speech.
    c = SentenceChunker(min_chars=10, max_chars=240, min_chars_first=6)
    assert collect(c, "42.") == ["42."]


def test_abbreviations_and_initialisms_do_not_end_a_sentence():
    c = SentenceChunker(min_chars=60, max_chars=240, min_chars_first=24)
    # "We met Dr." with sentence-final prosody plus a 140 ms seam, mid-name.
    assert collect(c, "We met Dr. Smith and Mr. Jones at 5 p.m. in the lobby.") == [
        "We met Dr. Smith and Mr. Jones at 5 p.m. in the lobby.",
    ]
    c = SentenceChunker(min_chars=6, max_chars=240, min_chars_first=6)
    assert c.feed("The U.S. team won. Next.") == ["The U.S. team won."]
    c = SentenceChunker(min_chars=6, max_chars=240, min_chars_first=6)
    assert c.feed("Ms. Smith arrived. Next.") == ["Ms. Smith arrived."]


def test_initials_bind_when_a_capitalised_word_follows():
    # A lone capital + "." read as a sentence end: "George W." got sentence-final
    # prosody and a seam pause mid-name. An initial sits between a capitalised word (or
    # a dotted initial, or the sentence start) and a capitalised word.
    for text, chunks in (
        ("George W. Bush was president. Then he left.",
         ["George W. Bush was president.", "Then he left."]),
        ("Dr. J. K. Rowling wrote it. Then more.", ["Dr. J. K. Rowling wrote it.", "Then more."]),
        ("J. K. Rowling wrote it. Then more.", ["J. K. Rowling wrote it.", "Then more."]),
        ("John F. Kennedy spoke. Then he left.", ["John F. Kennedy spoke.", "Then he left."]),
        ("Flight No. 5 leaves at 3:45pm. OK.", ["Flight No. 5 leaves at 3:45pm.", "OK."]),
        # The accepted side of the asymmetry: after a capitalised word, a sentence that
        # ENDS in a lone capital loses its pause when the next one is capitalised too.
        ("Meet K. Then go.", ["Meet K. Then go."]),
        ("Vitamin C. Then D.", ["Vitamin C. Then D."]),
        # A lowercase or non-capitalised follow-up keeps the cut; "no." binds only as
        # the capitalised "No." before a digit.
        ("It was plan B. and then some.", ["It was plan B.", "and then some."]),
        ("The answer is no. Next question.", ["The answer is no.", "Next question."]),
    ):
        c = SentenceChunker(min_chars=6, max_chars=240, min_chars_first=6)
        assert collect(c, text) == chunks, text
    # Streamed: the follow-up context arrives in a later delta, so the dot waits.
    c = SentenceChunker(min_chars=6, max_chars=240, min_chars_first=6)
    assert c.feed("George W.") == []
    assert c.feed(" Bush won. Then") == ["George W. Bush won."]
    c = SentenceChunker(min_chars=6, max_chars=240, min_chars_first=6)
    assert c.feed("Plan A. ") == []
    assert c.feed("Then B") == []
    assert c.flush() == "Plan A. Then B"


def test_a_lone_capital_after_a_lowercase_word_or_digit_ends_its_sentence():
    # "25°C." bound to the capitalised "Tomorrow": a weather reply's first chunk was
    # held until the NEXT sentence ended (TTFA), and the seam between them was lost.
    for text, chunks in (
        ("It is 25°C. Tomorrow will be warmer.", ["It is 25°C.", "Tomorrow will be warmer."]),
        ("It is 77°F. Tomorrow is cooler.", ["It is 77°F.", "Tomorrow is cooler."]),
        ("It is 25 °C. Then rain.", ["It is 25 °C.", "Then rain."]),
        ("Highs of 68-72°F. The wind is calm.", ["Highs of 68-72°F.", "The wind is calm."]),
        ("Take exit 42B. Room 5 is on the left.", ["Take exit 42B.", "Room 5 is on the left."]),
        ("Your seat is 14C. Gate 7 opens at noon.", ["Your seat is 14C.", "Gate 7 opens at noon."]),
        ("Apartment 5A. Then ring twice.", ["Apartment 5A.", "Then ring twice."]),
        ("Use 5 V. Then check.", ["Use 5 V.", "Then check."]),
        ("He got an A. She got a B. They left.", ["He got an A.", "She got a B.", "They left."]),
        ("It was plan B. Then C.", ["It was plan B.", "Then C."]),
        ("I chose option C. What about you?", ["I chose option C.", "What about you?"]),
        ("So did I. Then we went.", ["So did I.", "Then we went."]),
        ("The answer is B. Correct!", ["The answer is B.", "Correct!"]),
        ("My grade was an A. Then I left.", ["My grade was an A.", "Then I left."]),
        ("Step 1: press A. Step 2: press B.", ["Step 1: press A.", "Step 2:", "press B."]),
    ):
        c = SentenceChunker(min_chars=6, max_chars=240, min_chars_first=6)
        assert collect(c, text) == chunks, text
    # Streamed: the first chunk goes out at the next delta, not at the next sentence end.
    c = SentenceChunker(min_chars=6, max_chars=240, min_chars_first=6)
    assert c.feed("It is 25°C.") == []
    assert c.feed(" Tomorrow will") == ["It is 25°C."]


def test_lowercase_lookalikes_still_end_a_sentence():
    # "st"/"ms" bind only as capitalised titles: an ordinal or a unit ends its sentence.
    for text, first in (
        ("He came in 1st. Then he rested.", "He came in 1st."),
        ("It took 250 ms. Then it ran.", "It took 250 ms."),
    ):
        c = SentenceChunker(min_chars=6, max_chars=240, min_chars_first=6)
        assert c.feed(text) == [first]


def test_the_dot_guards_still_end_ordinary_sentences():
    c = SentenceChunker(min_chars=6, max_chars=240, min_chars_first=6)
    # Neither an abbreviation nor an initialism: the cut stands.
    assert c.feed("The answer is no. Next question.") == ["The answer is no."]
    c = SentenceChunker(min_chars=6, max_chars=240, min_chars_first=6)
    assert c.feed("It costs 3.14 dollars. Really.") == ["It costs 3.14 dollars."]


def test_max_chars_force_split_at_last_space():
    c = SentenceChunker(min_chars=10, max_chars=40)
    words = "word " * 20  # no sentence punctuation at all
    chunks = c.feed(words)
    assert chunks, "hard cap must force a flush"
    assert all(len(ch) <= 40 for ch in chunks)


# ---- fence machine ----------------------------------------------------------

def test_fence_in_single_delta_dropped():
    c = SentenceChunker(min_chars=10, max_chars=240)
    text = "".join(collect(c, "Here:\n```python\nprint(1)\n```\nDone."))
    assert "print" not in text
    assert "Here:" in text and "Done." in text


def test_fence_split_across_deltas_and_heldback_backticks():
    c = SentenceChunker(min_chars=10, max_chars=240)
    text = " ".join(collect(c, "``", "`\nsecret_code()\n``", "`\nok done."))
    assert "secret_code" not in text
    assert "ok done." in text


def test_mid_sentence_backticks_are_prose_not_an_opener():
    c = SentenceChunker(min_chars=10, max_chars=240)
    text = " ".join(collect(c, "wrap it in ``` fences to format.\nThen speak on."))
    assert "fences to format." in text
    assert "Then speak on." in text  # parity did NOT flip; nothing was muted


def test_two_prose_backtick_runs_do_not_swallow_the_words_between():
    c = SentenceChunker(min_chars=10, max_chars=240)
    # Neither run is at a line start, so neither opens a fence: treating the
    # first as an opener would let the second close it and eat "fences".
    assert c.feed("Wrap it in ``` fences ``` like this.\n") == ["Wrap it in fences like this."]


def test_unclosed_fence_at_flush_drops_the_rest():
    c = SentenceChunker(min_chars=10, max_chars=240)
    text = " ".join(collect(c, "Answer:\n```\nhidden stuff"))
    assert "hidden" not in text
    assert "Answer:" in text


def test_indented_fence_in_list_item_is_recognized():
    c = SentenceChunker(min_chars=10, max_chars=240)
    text = " ".join(collect(c, "- item\n  ```\nhidden()\n  ```\nafter."))
    assert "hidden" not in text
    assert "after." in text


def test_crlf_fence_recognized():
    c = SentenceChunker(min_chars=10, max_chars=240)
    text = " ".join(collect(c, "Line.\r\n```\r\nhidden\r\n```\r\nok."))
    assert "hidden" not in text
    assert "ok." in text


def test_flush_resets_first_chunk_floor():
    c = SentenceChunker(min_chars=60, max_chars=240, min_chars_first=10)
    assert c.feed("Short clause, and the rest keeps going for a while now")
    c.flush()
    assert c.feed("Tiny again, more text following")
