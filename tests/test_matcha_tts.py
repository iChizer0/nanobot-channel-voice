"""Matcha-TTS front-ends and host-side DSP, hermetic: token/lexicon parsing, the two
tokenizers (espeak phonemization mocked), the numpy ISTFT against a forward STFT, the
zh number verbalizer, and the registry fallback. Model math lives in
``test_ondevice_real.py``.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from nanobot_channel_voice.tts.matcha import (  # noqa: E402
    EspeakFrontend,
    LexiconFrontend,
    MatchaTtsAdapter,
    add_blank,
    fold_punct_aliases,
    istft,
    load_lexicon,
    official_token2id,
    read_tokens,
)
from nanobot_channel_voice.tts.text_frontend import verbalize_numbers_zh  # noqa: E402

# ---- file formats ----------------------------------------------------------


def test_read_tokens_handles_the_bare_id_space_line(tmp_path):
    p = tmp_path / "tokens.txt"
    p.write_text("_ 0\n^ 1\n$ 2\n 3\na 4\nni3 5\n", encoding="utf-8")
    tokens = read_tokens(str(p))
    assert tokens[" "] == 3          # a lone id means the symbol is the space char
    assert tokens["_"] == 0
    assert tokens["ni3"] == 5        # zh tokens are multi-char strings


def test_fold_punct_aliases_bridges_half_and_full_width():
    en = fold_punct_aliases({",": 8, ".": 10, "!": 4})
    assert en["，"] == 8 and en["。"] == 10 and en["！"] == 4
    assert en["…"] == 10             # ellipsis reads as a full stop
    zh = fold_punct_aliases({"，": 2, "。": 3})
    assert zh[","] == 2 and zh["."] == 3 and zh["、"] == 2


def test_load_lexicon_first_spelling_wins_and_oov_phones_drop(tmp_path):
    p = tmp_path / "lexicon.txt"
    p.write_text("你 n i2\n你 WRONG\n好 h ao3\n你好 n i2 h ao3\n坏 q x\n", encoding="utf-8")
    tokens = {"n": 1, "i2": 2, "h": 3, "ao3": 4}
    lex = load_lexicon(str(p), tokens)
    assert lex["你"] == [1, 2]        # duplicate line ignored
    assert lex["你好"] == [1, 2, 3, 4]
    assert "坏" not in lex            # unknown phones drop the word (sherpa behaviour)


def test_add_blank_frames_every_token():
    assert add_blank([5, 6], pad_id=1) == [1, 5, 1, 6, 1]
    assert add_blank([], pad_id=0) == [0]


# ---- the lexicon front-end -------------------------------------------------


def _zh_frontend():
    tokens = fold_punct_aliases({"，": 2, "。": 3, "n": 10, "i2": 11, "h": 12, "ao3": 13,
                                 "shi4": 14, "jie4": 15})
    lex = {"你": [10, 11], "好": [12, 13], "你好": [10, 11, 12, 13],
           "世": [14], "界": [15], "世界": [14, 15]}
    return LexiconFrontend(lex, tokens)


def test_lexicon_greedy_longest_match_prefers_the_phrase():
    fe = _zh_frontend()
    (seq,) = fe.sentences("你好")
    assert seq == [10, 11, 12, 13]   # matched as 你好, not 你 + 好


def test_lexicon_keeps_punctuation_and_splits_sentences():
    fe = _zh_frontend()
    seqs = fe.sentences("你好，世界。你好。")
    assert seqs == [[10, 11, 12, 13, 2, 14, 15, 3], [10, 11, 12, 13, 3]]


def test_lexicon_drops_oov_and_reports_via_can_speak():
    fe = _zh_frontend()
    (seq,) = fe.sentences("你x好")
    assert seq == [10, 11, 12, 13]   # OOV latin dropped from the id stream
    assert fe.can_speak("你") and not fe.can_speak("x")


# ---- the espeak front-end --------------------------------------------------


def _en_frontend(ipa_by_clause: dict[str, str]):
    tokens = fold_punct_aliases(
        {"_": 0, "^": 1, "$": 2, " ": 3, ",": 8, ".": 10, "!": 4,
         "h": 20, "ə": 59, "l": 24, "ˈ": 120, "o": 27, "ʊ": 100, "w": 35, "d": 17}
    )
    # Newline-batched like real espeak: one output line per input clause.
    fe = EspeakFrontend(
        tokens,
        phonemize=lambda text: "\n".join(ipa_by_clause[c] for c in text.split("\n")),
    )
    return fe, tokens


def test_espeak_ids_skip_unknown_phonemes_and_join_clauses():
    fe, t = _en_frontend({"hello": "həlˈoʊ", "world": "wˈɜːld"})
    (seq,) = fe.sentences("hello, world!")
    # ɜ and ː are not in this token table: skipped like sherpa, the rest survives.
    assert seq == [t["h"], t["ə"], t["l"], t["ˈ"], t["o"], t["ʊ"], t[","],
                   t[" "], t["w"], t["ˈ"], t["l"], t["d"], t["!"]]


def test_espeak_sentences_split_on_final_punctuation():
    fe, t = _en_frontend({"hello": "h", "world": "w"})
    assert fe.sentences("hello. world.") == [[t["h"], t["."]], [t["w"], t["."]]]


def test_sentence_and_clause_splits_leave_numbers_intact():
    # An intra-token '.'/':'/',' is content, not punctuation: "3.14" must reach
    # espeak as one clause (it reads the decimal), not as two sentences.
    fe, t = _en_frontend({"hello 3.14 world": "h", "hello": "h"})
    assert fe.sentences("hello 3.14 world.") == [[t["h"], t["."]]]
    fe2, t2 = _en_frontend({"hello 3:30 or 1,234 world": "w"})
    assert fe2.sentences("hello 3:30 or 1,234 world!") == [[t2["w"], t2["!"]]]


def test_clause_batch_falls_back_on_line_count_mismatch():
    # A phonemizer that re-splits the batch (extra newline) must not misalign
    # clauses; the frontend re-runs them one at a time.
    tokens = {"a": 1, "b": 2, " ": 0, ",": 8, ".": 10}
    calls = []

    def phonemize(text):
        calls.append(text)
        if "\n" in text:
            return "a\nb\nb"  # 3 lines for 2 clauses: bogus batch
        return {"one": "a", "two": "b"}[text]

    fe = EspeakFrontend(tokens, phonemize=phonemize)
    assert fe.sentences("one, two.") == [[1, 8, 0, 2, 10]]
    assert calls == ["one\ntwo", "one", "two"]


def test_espeak_language_switch_flags_are_stripped():
    # z/h/e/n are real phoneme symbols: an unstripped "(zh)" flag would be voiced.
    fe = EspeakFrontend({"z": 5, "h": 6, "e": 7, "n": 8, "a": 1, " ": 0},
                        phonemize=str)
    assert fe._ipa_ids("(zh)a(en)a") == [1, 1]


def test_espeak_can_speak_allows_latin_only():
    fe = EspeakFrontend({}, phonemize=str)
    assert fe.can_speak("a") and fe.can_speak("ü") and fe.can_speak("3")
    assert not fe.can_speak("好") and not fe.can_speak("こ") and not fe.can_speak("한")
    # Non-CJK foreign scripts must ALSO be refused: espeak would language-switch
    # and the English phoneme table would voice the survivors as gibberish.
    assert not fe.can_speak("б") and not fe.can_speak("ω") and not fe.can_speak("ب")


# ---- the official symbol table ---------------------------------------------


def test_official_symbol_table_matches_upstream_layout():
    t = fold_punct_aliases(official_token2id())
    # matcha/text/symbols.py: ["_"] + punctuation(16, space last) + letters(52) + IPA
    assert t["_"] == 0
    assert t[";"] == 1
    assert t[" "] == 16
    assert t["A"] == 17 and t["Z"] == 42 and t["a"] == 43 and t["z"] == 68
    assert t["ɑ"] == 69          # first IPA symbol
    assert "ˈ" in t and "ː" in t  # stress + length marks survive verbatim
    assert t["。"] == t["."]      # full-width folding still applies on top


# ---- ISTFT ------------------------------------------------------------------


def test_istft_inverts_a_forward_stft():
    n_fft, hop = 1024, 256
    rng = np.random.default_rng(0)
    signal = rng.standard_normal(hop * 40).astype(np.float32) * 0.3

    # torch.stft-compatible forward pass: reflect pad (center), periodic hann, rfft.
    padded = np.pad(signal, n_fft // 2, mode="reflect")
    win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))
    n_frames = (len(padded) - n_fft) // hop + 1
    frames = np.stack(
        [padded[i * hop : i * hop + n_fft] * win for i in range(n_frames)], axis=1
    )
    spec = np.fft.rfft(frames, axis=0)
    mag = np.abs(spec).astype(np.float32) + 1e-12
    cos = (spec.real / mag).astype(np.float32)
    sin = (spec.imag / mag).astype(np.float32)

    out = istft(mag, cos, sin, n_fft=n_fft, hop_length=hop, center=True)
    n = min(len(out), len(signal))
    assert np.allclose(out[:n], signal[:n], atol=1e-4)


# ---- adapter verdicts and normalization ------------------------------------


def test_matcha_normalize_verbalizes_digits_for_zh_only():
    adapter = MatchaTtsAdapter.__new__(MatchaTtsAdapter)  # no models needed
    adapter.spoken_language = "zh"
    assert adapter._normalize("现在7点") == "现在七点"
    adapter.spoken_language = "en"
    assert adapter._normalize("at 7 pm") == "at 7 pm"  # espeak reads digits itself


def test_verbalize_numbers_zh_inside_cjk_text():
    # \b never fires between a CJK char and a digit (both are \w); the zh patterns
    # anchor on digit lookarounds instead, or every real sentence bypasses them.
    assert verbalize_numbers_zh("现在是7:45。") == "现在是七点四十五分。"
    assert verbalize_numbers_zh("时间7：45") == "时间七点四十五分"  # full-width colon
    assert verbalize_numbers_zh("圆周率是3.14") == "圆周率是三点一四"
    assert verbalize_numbers_zh("增长50%") == "增长百分之五十"
    assert verbalize_numbers_zh("增长3.5%") == "增长百分之三点五"  # percent before decimal
    assert verbalize_numbers_zh("费率0.5％") == "费率百分之零点五"  # full-width percent
    assert verbalize_numbers_zh("共1,234元") == "共一千二百三十四元"


def test_verbalize_numbers_zh():
    assert verbalize_numbers_zh("0") == "零"
    assert verbalize_numbers_zh("12") == "十二"
    assert verbalize_numbers_zh("112") == "一百一十二"
    assert verbalize_numbers_zh("1005") == "一千零五"
    assert verbalize_numbers_zh("10000") == "一万"
    assert verbalize_numbers_zh("100200") == "十万零二百"
    assert verbalize_numbers_zh("100000001") == "一亿零一"
    assert verbalize_numbers_zh("1,234") == "一千二百三十四"
    assert verbalize_numbers_zh("7:45") == "七点四十五分"
    assert verbalize_numbers_zh("8:00") == "八点"
    assert verbalize_numbers_zh("9:05") == "九点零五分"
    assert verbalize_numbers_zh("50%") == "百分之五十"
    assert verbalize_numbers_zh("3.14") == "三点一四"


# ---- the espeak resolution ladder ------------------------------------------


def test_espeak_ladder_prefers_the_explicit_binary(monkeypatch, tmp_path):
    from nanobot_channel_voice.tts import espeak

    exe = tmp_path / "espeak-ng"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    calls = []
    monkeypatch.setattr(espeak.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or
                        type("P", (), {"returncode": 0, "stdout": "ɪpə", "stderr": ""})())
    fn = espeak.make_ipa_phonemizer("en-us", espeak_path=str(exe))
    assert fn("hi") == "ɪpə"
    # The build-time probe ran first, and the config path wins over $PATH.
    assert [c[0] for c in calls] == [str(exe), str(exe)]
    assert calls[0][-1] == "Okay."


def test_espeak_ladder_rejects_a_bad_explicit_path(tmp_path):
    from nanobot_channel_voice.tts import espeak

    with pytest.raises(RuntimeError) as exc:
        espeak.make_ipa_phonemizer("en-us", espeak_path=str(tmp_path / "nope"))
    # Fails at BUILD time (registry falls back to system TTS), naming the setting.
    assert "espeakPath" in str(exc.value)


def test_espeak_ladder_names_every_fix_when_nothing_is_found(monkeypatch):
    from nanobot_channel_voice.tts import espeak

    def no_loader():
        raise ImportError("espeakng_loader")

    monkeypatch.setattr(espeak.shutil, "which", lambda _: None)
    monkeypatch.setattr(espeak, "_load_library", no_loader)
    with pytest.raises(RuntimeError) as exc:
        espeak.make_ipa_phonemizer("en-us")  # no binary AND no espeakng-loader
    msg = str(exc.value)
    assert "espeakPath" in msg and "[espeak]" in msg and "apt install" in msg


def test_espeak_bundled_library_phonemizes():
    # Runs only where the [espeak] extra is installed (e.g. the dev container, which
    # deliberately has no espeak binary): the ctypes route must produce real IPA.
    pytest.importorskip("espeakng_loader")
    from nanobot_channel_voice.tts import espeak

    lib = espeak._load_library()
    ipa = espeak._library_phonemizer(lib, "en-us")("Hello world")
    assert "ə" in ipa or "ˈ" in ipa, ipa


# ---- registry ---------------------------------------------------------------


def test_make_tts_falls_back_to_system_without_model_paths():
    from nanobot_channel_voice.config import TtsConfig
    from nanobot_channel_voice.tts import make_tts

    tts = make_tts(TtsConfig.model_validate({"provider": "matcha"}))
    assert type(tts).__name__ == "SystemTtsAdapter"
