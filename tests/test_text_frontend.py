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
        ("in 1989", "in one thousand nine hundred eighty nine"),
        ("battery at 87%", "battery at eighty seven percent"),
        ("August 3rd", "August third"),
        ("the 21st time", "the twenty first time"),
        ("1,234 items", "one thousand two hundred thirty four items"),
        ("0 issues", "zero issues"),
        ("no digits here", "no digits here"),
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
