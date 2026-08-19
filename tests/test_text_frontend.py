"""English number verbalization + the MMS tokenizer's silent unmapped-char drops."""

from __future__ import annotations

import pytest

from nanobot_channel_voice.tts.text_frontend import verbalize_numbers_en


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Your meeting is at 7:45", "Your meeting is at seven forty five"),
        ("at 8:05 sharp", "at eight oh five sharp"),
        ("around 12:00", "around twelve o'clock"),
        ("pi is 3.14", "pi is three point one four"),
        ("in 1989", "in nineteen eighty nine"),          # a year, not a quantity
        ("in 2008", "in two thousand eight"),            # …but not pairwise in the 2000s
        ("battery at 87%", "battery at eighty seven percent"),
        ("August 3rd", "August third"),
        ("the 21st time", "the twenty first time"),
        ("1,234 items", "one thousand two hundred thirty four items"),
        ("0 issues", "zero issues"),
        # A leading zero marks an identifier; the cardinal reading deletes it.
        ("agent 007", "agent zero zero seven"),
        ("badge 0042", "badge zero zero four two"),
        ("zip 02138", "zip zero two one three eight"),
        # Multi-dot runs: the decimal rule matched only the first dot and left a "."
        ("router at 192.168.1.1", "router at one nine two point one six eight point one point one"),
        # Greediness bounds the run; a lookahead barring "." ate the sentence period.
        ("version 3.14.2.", "version three point fourteen point two."),
        ("no digits here", "no digits here"),
        # Sequences: structural evidence, then the trigger lexicon.
        ("Call 555-1234", "Call five five five, one two three four"),
        ("my zip is 94105", "my zip is nine four one zero five"),
        ("extension 8021", "extension eight zero two one"),
        ("Call +1 415 555 2671",
         "Call one, four one five, five five five, two six seven one"),
        ("that's flight UA837", "that's flight UA eight three seven"),
        # …and the guards that keep quantities quantities.
        ("1000000 residents", "one million residents"),   # unit beats the length rule
        ("it takes 10-15 minutes", "it takes ten-fifteen minutes"),  # a range, not an id
        ("COVID-19 spread", "COVID-nineteen spread"),     # too short to be glue
        ("the account ending in 1234",
         "the account ending in one two three four"),     # trigger outranks the year cue
        ("from 2020-2024", "from twenty twenty to twenty twenty four"),
        # A reading must never fuse with the letters beside it.
        ("board at gate B12", "board at gate B twelve"),
        # "+1-415-…" matches the international AND the hyphen pattern; overlapping spans
        # used to read the whole number twice.
        ("Call +1-415-555-2671",
         "Call one, four one five, five five five, two six seven one"),
        # Cues weak enough to be ordinary prose are not triggers or year cues.
        ("I have no idea, 4821 is wrong",
         "I have no idea, four thousand eight hundred twenty one is wrong"),
        ("a value of 1234", "a value of one thousand two hundred thirty four"),
        ("seat 14C", "seat fourteen C"),
    ],
)
def test_verbalize_numbers_en(text, expected):
    assert verbalize_numbers_en(text) == expected


@pytest.mark.parametrize(
    ("run", "value"),
    [
        ("四十五", 45), ("十", 10), ("十二", 12), ("二十", 20),
        ("一百零五", 105), ("两千零二十六", 2026), ("三万五千", 35000),
        ("零", 0), ("七", 7), ("一二三", 123),  # no unit char: positional
        ("四零四", 404), ("一亿", 100000000),
        ("一亿零五万", 100050000), ("三亿五千万", 350000000),
        ("一万亿", 10**12), ("五万三千", 53000), ("万", 10000),
        ("好", None),  # not numeral material
    ],
)
def test_zh_numeral_value(run, value):
    from nanobot_channel_voice.tts.text_frontend import zh_numeral_value

    assert zh_numeral_value(run) == value


def test_space_digit_sequences_is_engine_native():
    from nanobot_channel_voice.tts.text_frontend import space_digit_sequences

    # espeak names spaced digits in every voice, so this path needs no word table.
    assert space_digit_sequences("my zip is 94105") == "my zip is 9 4 1 0 5"
    assert space_digit_sequences("Call 555-1234") == "Call 5 5 5, 1 2 3 4"
    assert space_digit_sequences("in 1999") == "in 19 99"
    assert space_digit_sequences("in 2008") == "in 2008"      # espeak's own reading is right
    assert space_digit_sequences("1234 items") == "1234 items"


def test_huge_digit_runs_are_read_out():
    # Phone-number/id territory: scale words would be absurd; read the digits.
    out = verbalize_numbers_en("call 8005551212000")
    assert out == "call eight zero zero five five five one two one two zero zero zero"


def test_mms_english_pipeline_keeps_number_content():
    np = pytest.importorskip("numpy")  # noqa: F841 - mms.py imports numpy
    from nanobot_channel_voice.tts import mms

    # Regression: mms-tts-eng has no "7"/"8"/"9"; unverbalized "7:45" tokenized
    # to just "4 5" with no warning.
    verbalized = verbalize_numbers_en("Your meeting is at 7:45")
    ids, mask = mms.preprocess_input(verbalized, 200, mms._ENG_VOCAB)
    kept = (int(mask.sum()) - 1) // 2  # 2 ids per kept char, one closing 0
    # Every char of the verbalized text maps into the vocab (spaces included).
    assert kept == len(verbalized)


def test_mms_preprocess_drops_unmapped_chars_silently():
    pytest.importorskip("numpy")
    from nanobot_channel_voice.tts import mms

    # The tokenizer is the faithful reference port and drops silently; reporting is the
    # shell's speakability guard (test_ondevice_guard.py), one policy for both engines.
    ids, mask = mms.preprocess_input("naïve 7", 200, mms._ENG_VOCAB)  # ï and 7 unmappable
    assert (int(mask.sum()) - 1) // 2 == len("nave ")
