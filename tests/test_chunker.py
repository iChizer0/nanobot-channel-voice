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


def test_sanitize_smart_quotes_fold_before_whitelist():
    assert sanitize("“Hi” ‘there’") == '"Hi" \'there\''


def test_sanitize_unspeakable_becomes_space_never_glue():
    assert sanitize("A→B") == "A B"
    assert sanitize("Hi \U0001f44b there") == "Hi there"


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
    c = SentenceChunker(min_chars=60, max_chars=240)
    assert c.feed("你好。") == ["你好。"]


def test_first_chunk_floor_cuts_earlier_then_steady_floor_applies():
    c = SentenceChunker(min_chars=60, max_chars=240, min_chars_first=10)
    # First chunk: the clause comma at/after the 10-char floor cuts early.
    assert c.feed("A tiny clause, ") == ["A tiny clause,"]
    # Steady state: the same shape now buffers (comma before min_chars=60).
    assert c.feed("Another bit, ") == []


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
