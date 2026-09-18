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
        # Percent is amount-aware and runs before the decimal/grouped passes, which
        # would eat the digits and strand a "%" no char vocab can voice.
        ("growth of 3.5%", "growth of three point five percent"),
        ("up 1,234% overall", "up one thousand two hundred thirty four percent overall"),
        # A glued unit is still a decimal; \b never fired between digit and letter.
        ("add 3.5kg of flour", "add three point five kg of flour"),
        ("the v3.5 release", "the v three point five release"),
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
        # A range, not an id -- and it keeps a connective: espeak names a bare hyphen
        # ("five dash ten") and the char-vocab engines drop it, fusing the two numbers.
        ("it takes 10-15 minutes", "it takes ten to fifteen minutes"),
        ("3-5 days", "three to five days"),
        ("5−10 minutes", "five to ten minutes"),  # U+2212: canned phrases skip sanitize()
        ("order 3-5", "order three-five"),               # no unit behind it: left alone
        ("about 1/2 cup", "about one half cup"),
        ("2/3 of users", "two thirds of users"),
        # Not proper fractions: the numbers still read, but never as "twenty fourths".
        ("open 24/7", "open twenty four/seven"),
        ("a 16/9 screen", "a sixteen/nine screen"),
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
        # Dates, currency and degrees rewrite before the number passes.
        ("released on 2026-08-19", "released on August nineteenth, twenty twenty six"),
        ("due 2026/8/1", "due August first, twenty twenty six"),
        ("back in 2008-05-03", "back in May third, two thousand eight"),
        ("on 2026.08.19.", "on August nineteenth, twenty twenty six."),
        ("at 2026-08-19T21:33", "at August nineteenth, twenty twenty six twenty one thirty three"),
        # Written-out dates: ordinal day, spoken year — the shape "what's the
        # date" answers take.
        ("Today is August 20, 2026.", "Today is August twentieth, twenty twenty six."),
        ("due August 19", "due August nineteenth"),
        ("back in August 2026", "back in August twenty twenty six"),
        # …with the unit and calendar guards that keep quantities quantities.
        ("May 5 minutes later", "May five minutes later"),
        ("March 2000 meters up", "March two thousand meters up"),
        ("meet on September 31", "meet on September thirty one"),
        ("May 12:30 works", "May twelve thirty works"),  # a month and a clock, not a day
        ("May 12: the schedule", "May twelfth: the schedule"),
        # Decades; a 2-digit one needs a determiner ("30s" is thirty seconds).
        ("the 1990s were wild", "the nineteen nineties were wild"),
        ("music of the 2000s", "music of the two thousands"),
        ("in her 30s", "in her thirties"),
        ("mid-2020s style", "mid-twenty twenties style"),
        ("a 30s timeout", "a thirty s timeout"),
        ("the 1990's were wild", "the nineteen nineties were wild"),  # possessive homophone
        # A hyphen glued to a word is a compound, not a minus sign.
        ("wind-3°C reading", "wind-three degrees Celsius reading"),
        ("costs $5.99", "costs five point nine nine dollars"),
        # Trailing fraction zeros are price formatting, not speech.
        ("$5.00 total", "five dollars total"),
        ("$3.50 each", "three point five dollars each"),
        ("a $1.00 fee", "a one dollar fee"),
        ("just $1", "just one dollar"),
        ("a $1,000 grant", "a one thousand dollars grant"),
        ("about ¥199", "about one hundred ninety nine yuan"),
        ("it's 25°C out", "it's twenty five degrees Celsius out"),
        ("low of 1°C", "low of one degree Celsius"),
        ("oven to 350 ℉", "oven to three hundred fifty degrees Fahrenheit"),
        # The sign survives only on temperatures, where a leading hyphen is
        # unambiguous; the digit lookbehind keeps "20-30°C" a range.
        ("it is -3.5°C now", "it is minus three point five degrees Celsius now"),
        ("−5℃ tonight", "minus five degrees Celsius tonight"),
        # A degree mark is a unit: the range gets its connective (espeak said "dash",
        # the zh lexicon fused the ends), and the ℃/℉ spellings too.
        ("between 20-30°C", "between twenty to thirty degrees Celsius"),
        ("highs of 70-80°F", "highs of seventy to eighty degrees Fahrenheit"),
        ("20-30℃ today", "twenty to thirty degrees Celsius today"),
        ("20-30 °C", "twenty to thirty degrees Celsius"),
        # A "%" on the HIGH end alone also marks a range; the low end need not repeat it.
        ("5-10%", "five to ten percent"),
        ("a 10-20% discount", "a ten to twenty percent discount"),
        ("5-10 %", "five to ten percent"),
        ("5%-10%", "five percent to ten percent"),
        # Sentence-final: the high end's decimal guard sat AFTER its optional "%", so
        # the match backtracked to a bare "10" and the range lost its connective.
        ("Growth of 5-10%.", "Growth of five to ten percent."),
        ("40-50%.", "forty to fifty percent."),
        ("5-10 %.", "five to ten percent."),
        ("5-10%...", "five to ten percent..."),
        ("5-10.5%", "five to ten point five percent"),
        # …while a sentence-final pair with no unit keeps its id/year reading.
        ("call 555-1234.", "call five five five, one two three four."),
        ("from 2020-2024.", "from twenty twenty to twenty twenty four."),
        ("order 5-10.", "order five, one zero."),
        # The clock reading is padded like every other renderer: a glued am/pm fused
        # ("three forty fivepm", espeak: fˈaɪvəpəm).
        ("3:45pm", "three forty five pm"),
        ("12:05am", "twelve oh five am"),
        # A meridiem already says the hour is on the hour: "nine am", not "nine o'clock am".
        ("at 9:00am", "at nine am"),
        ("at 9:00 a.m.", "at nine a.m."),
        ("12:00 PM", "twelve PM"),
        ("9:00 amazing", "nine o'clock amazing"),
        ("at 9:00:00 am", "at nine zero zero am"),  # h:mm:ss keeps its three fields
        # A grouped amount with a glued suffix: \b never fired between "0" and "k", and
        # the fallthrough read "000" as a leading-zero sequence.
        ("1,000km", "one thousand km"),
        ("1,000x faster", "one thousand x faster"),
        # A grouped power of ten with a glued "s" is a plural scale word, not "thousand s".
        ("1,000s of users", "thousands of users"),
        ("10,000s of users", "tens of thousands of users"),
        ("100,000s of users", "hundreds of thousands of users"),
        ("1,000,000s of dollars", "millions of dollars"),
        ("$1,000k", "$one thousand k"),
        ("the 1,000th user", "the one thousandth user"),
        # A decimal is never an ordinal: "99.9th" reached the ordinal pass as "9th"
        # ("ninety nine.ninth"). It reads as the decimal pass leaves it, suffix and all.
        ("the 99.9th percentile", "the ninety nine point nine th percentile"),
        ("the 1,000.5th", "the one thousand point five th"),
        ("the 1.2nd", "the one point two nd"),
        # A lone k/m/b/x after a number is a magnitude or multiplier, not id glue.
        ("100k", "one hundred k"),
        ("100k users", "one hundred k users"),
        ("10x faster", "ten x faster"),
        ("part 100X4", "part one zero zero X four"),   # letter+digit is still glue
        # The Latin unit abbreviations zh already knew (ms, mg, hz, kw…) are units here
        # too: glued, they read digit-wise as id glue ("one zero zero ms").
        ("Latency is 100ms", "Latency is one hundred ms"),
        ("500mg", "five hundred mg"),
        ("100Hz", "one hundred Hz"),
        ("100mph", "one hundred mph"),
        ("100px", "one hundred px"),
        ("100kW", "one hundred kW"),
        ("100Mbps", "one hundred Mbps"),
        ("100-200ms", "one hundred to two hundred ms"),
        # A glued suffix after a decimal made the currency pass backtrack to the integer
        # part ("one dollar.five k"); it falls through to the k path, like "$100k".
        ("$1.5k", "$one point five k"),
        ("$2.5b", "$two point five b"),
        # A grouped amount is explicit quantity notation: named at any size.
        ("1,000,000,000,000 stars", "one trillion stars"),
        ("2,500,000,000,000,000", "two quadrillion five hundred trillion"),
        ("$1,000,000,000,000", "one trillion dollars"),
        # A scale word rides in front of the relocated unit.
        ("about $5 million", "about five million dollars"),
        ("a $1.5 billion round", "a one point five billion dollars round"),
        ("$1,234.56 total", "one thousand two hundred thirty four point five six dollars total"),
        # Shapes the currency pass cannot own fall back to the plain readings.
        ("a $100k budget", "a $one hundred k budget"),
        ("tickets are $20-30", "tickets are $twenty-thirty"),
        ("it costs $5 million dollars", "it costs $five million dollars"),
        # An invalid month or impossible day is not a date; the id keeps its
        # sequence reading.
        ("id 1234-56-78", "id one two three four, five six, seven eight"),
        ("meet on 2026-02-30", "meet on two zero two six, zero two, three zero"),
    ],
)
def test_verbalize_numbers_en(text, expected):
    assert verbalize_numbers_en(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Trigger nouns (number, file, account, order, status...) must not read everyday
        # quantities digit-wise: a run under three digits reads the same either way, and a
        # modifier (total/balance) or possessor ("has" before the run, "of" not directly
        # before it) after the trigger vetoes it. Measured at no recall cost.
        ("the number of users is 45", "the number of users is forty five"),
        ("the number of users is 4821", "the number of users is four thousand eight hundred twenty one"),
        ("the file has 20 lines", "the file has twenty lines"),
        ("my account balance is 250", "my account balance is two hundred fifty"),
        ("The account total is 4821.", "The account total is four thousand eight hundred twenty one."),
        ("The invoice amount is 1500.", "The invoice amount is one thousand five hundred."),
        ("the order total is 45", "the order total is forty five"),
        ("the status is 42", "the status is forty two"),
        ("my phone has 12 apps", "my phone has twelve apps"),
        ("I'll call you back in 10", "I'll call you back in ten"),
        ("my flight is at 10", "my flight is at ten"),
        ("the card has 500 points", "the card has five hundred points"),
        ("room 12", "room twelve"),
        # "lot" left the lexicon: "a lot of" is everyday, lot 707 is a rare label.
        ("a lot of 500", "a lot of five hundred"),
        ("the lot has 40 cars", "the lot has forty cars"),
        # …while the label readings the lexicon exists for still fire.
        ("room 101", "room one zero one"),
        ("the account ending in 4821", "the account ending in four eight two one"),
        ("Your PIN is 4829", "Your PIN is four eight two nine"),
        ("Jenny's number ends in 5309", "Jenny's number ends in five three zero nine"),
        ("the code has 6 digits: 482913", "the code has six digits: four eight two nine one three"),
        ("my phone with the number 5551234",
         "my phone with the number five five five one two three four"),
        # …including the standard id phrasings, "X of NNNN" and "X has been …", which a
        # position-free veto disarmed.
        ("Use a PIN of 1234.", "Use a PIN of one two three four."),
        ("Your verification code has been sent: 482913.",
         "Your verification code has been sent: four eight two nine one three."),
        ("Enter the zip code of 90210.", "Enter the zip code of nine zero two one zero."),
        ("with a confirmation number of 778899.",
         "with a confirmation number of seven seven eight eight nine nine."),
        ("an extension of 4021.", "an extension of four zero two one."),
        ("with an ID of 4242.", "with an ID of four two four two."),
        ("a case number of 12345.", "a case number of one two three four five."),
        ("the port of 8080.", "the port of eight zero eight zero."),
        ("a status of 404.", "a status of four zero four."),
        ("an OTP of 482913.", "an OTP of four eight two nine one three."),
        ("Your PIN has changed to 4829.", "Your PIN has changed to four eight two nine."),
        ("The code I had was 482913.", "The code I had was four eight two nine one three."),
        ("The code you have is 482913.", "The code you have is four eight two nine one three."),
        ("The extension has changed to 4021.", "The extension has changed to four zero two one."),
        ("The ticket has been assigned 12345.",
         "The ticket has been assigned one two three four five."),
        ("The area code of 415 is San Francisco.",
         "The area code of four one five is San Francisco."),
    ],
)
def test_trigger_nouns_do_not_shred_everyday_quantities(text, expected):
    assert verbalize_numbers_en(text) == expected


def test_bare_huge_runs_keep_the_digit_fallback():
    # Only a GROUPED amount is explicit quantity notation; an ungrouped 13-digit run
    # behind a unit is still read out (the id-territory rule of the bare cardinal pass).
    assert verbalize_numbers_en("1000000000000 residents") == (
        "one zero zero zero zero zero zero zero zero zero zero zero zero residents"
    )


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
    # English dates render as full words: with digits left behind, a trigger word
    # ("ticket") would make the sequence pass re-shred the date's own rendering.
    assert space_digit_sequences("on 2026-08-19") == "on August nineteenth, twenty twenty six"
    assert space_digit_sequences("on 2008-05-03") == "on May third, two thousand eight"
    assert space_digit_sequences("ticket 2026-08-19") == "ticket August nineteenth, twenty twenty six"
    assert space_digit_sequences("from 2020-2024") == "from 20 20 to 20 24"  # a range, not a date
    # Written-out dates and decades render on this path too — the trailing "s" of
    # "1990s" is letter glue, which would digit-shred the year.
    assert space_digit_sequences("Today is August 20, 2026.") == \
        "Today is August twentieth, twenty twenty six."
    assert space_digit_sequences("the 1990s were wild") == "the nineteen nineties were wild"
    # Currency and degree policy hold on this path too (espeak alone says "yen").
    assert space_digit_sequences("about ¥199") == "about 199 yuan"
    assert space_digit_sequences("it is 25°C") == "it is 25 degrees Celsius"
    assert space_digit_sequences("$5.00 total") == "5 dollars total"
    assert space_digit_sequences("it is -3.5°C") == "it is minus 3.5 degrees Celsius"
    # Range connectives too: espeak names a bare hyphen ("twenty dash thirty"), and a
    # hyphen pair with a stray "%" behind it was digit-shredded ("5, 1 0%").
    assert space_digit_sequences("20-30°C") == "20 to 30 degrees Celsius"
    assert space_digit_sequences("5-10%") == "5 to 10%"
    assert space_digit_sequences("Growth of 5-10%.") == "Growth of 5 to 10%."
    assert space_digit_sequences("at 9:00am") == "at nine am"
    assert space_digit_sequences("1,000km") == "one thousand km"
    assert space_digit_sequences("1,000s of users") == "thousands of users"
    assert space_digit_sequences("the 1,000th user") == "the one thousandth user"
    assert space_digit_sequences("the 21st user") == "the 21st user"  # plain: engine-native
    assert space_digit_sequences("$100k") == "$100k"  # no glue rule fires: espeak's own cardinal
    # Any other language keeps the language-neutral digit spacing — month names
    # are English words a German or Japanese voice cannot say.
    assert space_digit_sequences("Termin am 2026-08-19", "de") == "Termin am 2 0 2 6, 0 8, 1 9"


def test_espeak_path_renders_clocks_and_grouped_amounts_to_words():
    from nanobot_channel_voice.tts.text_frontend import space_digit_sequences

    # Regression: with neither pass in front of the sequence rules, the leading-zero
    # rule shredded the "00" of a clock and every "000" group of a grouped amount —
    # espeak then read "12:0 0" as "twelve COLON zero zero" and "1,0 0 0,0 0 0" as
    # seven separate digits instead of "one million".
    assert space_digit_sequences("at 12:00") == "at twelve o'clock"
    assert space_digit_sequences("meeting at 9:00 am") == "meeting at nine am"
    assert space_digit_sequences("the time is 08:30") == "the time is eight thirty"
    assert space_digit_sequences("it was 1,000,000") == "it was one million"
    assert space_digit_sequences("3,000 and 40,000") == "three thousand and forty thousand"
    assert space_digit_sequences("1,020,300") == "one million twenty thousand three hundred"
    # A grouped decimal goes whole (grouped alone stranded ".56" as "dot five six");
    # a plain decimal stays engine-native, as every other bare number here.
    assert space_digit_sequences("It costs $1,234.56.") == (
        "It costs one thousand two hundred thirty four point five six dollars."
    )
    assert space_digit_sequences("Rate 1,000.25 per unit.") == (
        "Rate one thousand point two five per unit."
    )
    assert space_digit_sequences("about 3.14 here") == "about 3.14 here"
    # h:mm:ss is read whole; half-read, espeak voiced the remaining colon.
    assert space_digit_sequences("Duration 1:30:45.") == "Duration one thirty forty five."
    assert space_digit_sequences("at 12:00:00 UTC") == "at twelve zero zero UTC"
    # Other languages have no word table on this path and keep the neutral spacing.
    assert space_digit_sequences("um 12:00", "de") == "um 12:0 0"


def test_mms_overflow_splits_in_proportion_to_the_predicted_overrun():
    pytest.importorskip("numpy")
    import numpy as np

    from nanobot_channel_voice.tts.mms import MmsTtsAdapter

    # The budget bounds the ENCODER input (2 ids per char + 1 closer); the decoder window
    # is fixed, so the ~1 in 8 sentence that overruns it is re-cut where the prediction
    # says it fits, not blindly halved (a smaller budget cost every sentence a pass).
    tts = MmsTtsAdapter.__new__(MmsTtsAdapter)
    tts._max_length = 200
    assert tts._piece_budget() == 99
    tts.output_rate = 16000
    tts._join_gap_s = 0.06
    seen: list[str] = []
    tts._synthesize_piece = lambda text: (seen.append(text), np.ones(160, np.float32))[1]
    text = "one two three four five six seven eight nine ten"
    tts._halve_and_retry(text, frac=0.8)
    assert seen == ["one two three four five six seven eight", "nine ten"]
    seen.clear()
    tts._halve_and_retry(text)  # default: the midpoint
    assert seen == ["one two three four five", "six seven eight nine ten"]


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
