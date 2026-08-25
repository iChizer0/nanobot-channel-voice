"""Lexicon matching: entirety + contiguity for spaced scripts, greedy
segmentation for fused CJK runs."""

from __future__ import annotations

from nanobot_channel_voice.phrases import (
    FILLER_WORDS,
    PhraseLexicon,
    covered,
    phrase_within,
    pure_command,
    tokens_of,
)

STOP = PhraseLexicon([
    "stop", "stop it", "shut up", "that's enough", "never mind", "wait",
    "停", "停止", "别说了", "够了",
    "ストップ", "止めて", "待って",
])
ACK = PhraseLexicon(["ok", "okay", "got it", "嗯", "好的", "うん"])


def _stop(text: str) -> bool:
    return pure_command(tokens_of(text), STOP, ACK, extra=FILLER_WORDS)


# ---- spaced scripts ---------------------------------------------------------

def test_bare_and_decorated_stops():
    assert _stop("stop")
    assert _stop("Stop!")
    assert _stop("please stop")
    assert _stop("no no stop")
    assert _stop("okay okay stop")  # ack words may ride along
    assert _stop("stop it")
    assert _stop("that's enough")   # apostrophe splits: contiguous (that, s, enough)
    assert _stop("wait wait")


def test_content_is_never_a_command():
    assert not _stop("stop the music")     # non-lexicon word -> content
    assert not _stop("wait what")
    assert not _stop("")
    assert not _stop("okay")               # companions alone: no command phrase
    assert not _stop("please")


def test_multiword_phrases_need_contiguity_and_loose_words_never_hit():
    assert _stop("shut up")
    assert not _stop("up")          # a word OF a phrase is not the phrase
    assert not _stop("up up up")
    # Scattered words covered by companions still lack a contiguous phrase.
    assert not _stop("never okay mind")


# ---- fused CJK runs ---------------------------------------------------------

def test_cjk_single_runs_segment_greedily():
    assert _stop("停")
    assert _stop("停停停")       # repetition fuses into one \w+ token
    assert _stop("好的停")       # ack + stop fused
    assert _stop("别说了")
    assert _stop("停止")
    assert not _stop("公交车停靠站")  # contains 停 but is not decomposable command material


def test_japanese_with_polite_fillers():
    assert _stop("ストップ")
    assert _stop("止めてください")  # ください is filler vocabulary
    assert _stop("待って")


def test_cjk_ack_repetition_is_covered():
    # The _is_ack upgrade: unspaced repetition must read as backchannel material.
    assert covered(tokens_of("好的好的"), ACK)
    assert covered(tokens_of("うんうん"), ACK)
    assert not covered(tokens_of("好的走吧"), ACK)
    assert not covered([], ACK)


# ---- per-token hit (the early-confirm / echo-override primitive) ------------

def test_single_token_hits():
    assert pure_command(["stop"], STOP, ACK, extra=FILLER_WORDS)
    assert pure_command(["停停停"], STOP, ACK, extra=FILLER_WORDS)
    assert not pure_command(["up"], STOP, ACK, extra=FILLER_WORDS)
    assert not pure_command(["okay"], STOP, ACK, extra=FILLER_WORDS)


# ---- phrase inside free content (the goal trigger) --------------------------

_GOAL = PhraseLexicon([
    "keep working on", "don't stop until", "stop", "持续处理", "持续 处理",
])


def test_phrase_within_finds_a_trailing_commitment():
    assert phrase_within("book the flight and keep working on it until it's done", _GOAL)
    assert phrase_within("don't stop until you find one", _GOAL)


def test_phrase_within_honours_token_boundaries_for_spaced_scripts():
    # The padded join is what keeps a single-word phrase off word interiors.
    assert phrase_within("stop the timer", _GOAL)
    assert not phrase_within("set a stopwatch", _GOAL)


def test_phrase_within_reaches_inside_a_fused_cjk_run():
    # PhraseMatcher.present cannot: its decomposition demands the WHOLE token be
    # lexicon singles, so a trigger surrounded by real content never matches.
    assert phrase_within("持续处理这个问题直到解决", _GOAL)
    assert not phrase_within("这个问题很难", _GOAL)


def test_phrase_within_ignores_spacing_in_an_unspaced_phrase():
    # Spacing is meaningless on both sides of an unspaced script, so a configured
    # "持续 处理" must find 持续处理 in a transcript that never spaces it.
    assert phrase_within("请持续处理", PhraseLexicon(["持续 处理"]))
    # And an apostrophe splits identically in phrase and transcript.
    assert phrase_within("please don't stop until it works", _GOAL)
