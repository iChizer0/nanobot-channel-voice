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
    VocoderSpec,
    add_blank,
    denoise,
    denoiser_bias,
    fold_punct_aliases,
    istft,
    load_lexicon,
    official_token2id,
    read_tokens,
    stft,
)
from nanobot_channel_voice.tts.text_frontend import verbalize_numbers_zh  # noqa: E402

# ---- file formats -----------------------------------------------------------


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


def test_lexicon_overrides_layer_over_the_model_lexicon(tmp_path):
    from nanobot_channel_voice.tts.matcha import _load_lexicons

    base = tmp_path / "lexicon.txt"
    base.write_text("空 kong1\n抽 chou1\n", encoding="utf-8")
    over = tmp_path / "overrides.txt"
    over.write_text("抽空 chou1 kong4\n空 kong9\n", encoding="utf-8")
    tokens = {"kong1": 1, "kong4": 2, "chou1": 3}
    lex = _load_lexicons(str(base), str(over), tokens)
    assert lex["抽空"] == [3, 2]      # new phrase entry wins greedy longest-match
    assert lex["空"] == [1]           # bad override phone drops; the model entry stays


def test_lexicon_overrides_require_a_lexicon():
    from nanobot_channel_voice.config import MatchaTtsConfig

    with pytest.raises(ValueError, match="lexiconOverridesPath"):
        MatchaTtsConfig.model_validate({"lexiconOverridesPath": "overrides.txt"})


def test_add_blank_frames_every_token():
    assert add_blank([5, 6], pad_id=1) == [1, 5, 1, 6, 1]
    assert add_blank([], pad_id=0) == [0]


# ---- the lexicon front-end --------------------------------------------------


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


# ---- the espeak front-end ---------------------------------------------------


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


# ---- the official symbol table ----------------------------------------------


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


# ---- the bilingual zh-en dialect --------------------------------------------


def test_english_to_ipa_folds_to_the_trained_alphabet():
    from nanobot_channel_voice.tts.matcha import EnglishToIpa

    tokens = {"h": 1, "ə": 2, "l": 3, "O": 4, "ˈ": 5, "ɛ": 6, "ɡ": 7, "ɹ": 8,
              "ʧ": 9, "A": 10, "t": 11, "i": 12, "d": 13}
    ipa = {"hello": "həlˈoʊ", "great": "ɡɹˈeɪt", "teach": "tˈiːtʃ", "red": "rˈɛd", "e": "e"}
    e2i = EnglishToIpa(tokens, phonemize=lambda w: ipa[w])
    assert e2i.word_ids("hello") == [1, 2, 3, 5, 4]        # oʊ -> O
    assert e2i.word_ids("great") == [7, 8, 5, 10, 11]      # eɪ -> A; g/r fold to ɡ/ɹ
    assert e2i.word_ids("teach") == [11, 5, 12, 9]         # ː stripped, tʃ -> ʧ
    assert e2i.word_ids("red") == [8, 5, 6, 13]
    assert e2i.word_ids("e") == [6]                        # bare e never trained: -> ɛ


def test_english_to_ipa_drops_the_word_on_espeak_failure():
    from nanobot_channel_voice.tts.matcha import EnglishToIpa

    def boom(_word: str) -> str:
        raise OSError("espeak died")

    assert EnglishToIpa({"h": 1}, phonemize=boom).word_ids("hello") == []


def test_english_words_prime_in_one_espeak_call():
    """Subprocess espeak spawns per call: a fresh English clause must batch."""
    from nanobot_channel_voice.tts.matcha import EnglishToIpa

    tokens = {"h": 1, "i": 2, "O": 3, " ": 9, "。": 8}
    ipa = {"hi": "hi", "ho": "hoʊ"}
    calls: list[str] = []

    def phonemize(text: str) -> str:
        calls.append(text)
        return "\n".join(ipa[w] for w in text.split("\n"))

    fe = LexiconFrontend({}, tokens, english=EnglishToIpa(tokens, phonemize),
                         latin_space_id=9)
    (seq,) = fe.sentences("hi ho。")
    assert seq == [1, 2, 9, 1, 3, 9, 8]
    assert calls == ["hi\nho"]          # one batch, no per-word spawns
    fe.sentences("ho hi。")
    assert calls == ["hi\nho"]          # everything served from the cache


def test_prime_recovers_per_word_when_espeak_reclauses():
    from nanobot_channel_voice.tts.matcha import EnglishToIpa

    calls: list[str] = []

    def phonemize(text: str) -> str:
        calls.append(text)
        if "\n" in text:
            return "hi"  # espeak merged the batch: line count lies
        return {"hi": "hi", "ho": "hoʊ"}[text]

    e2i = EnglishToIpa({"h": 1, "i": 2, "O": 3}, phonemize)
    e2i.prime(["hi", "ho"])
    assert e2i.word_ids("hi") == [1, 2] and e2i.word_ids("ho") == [1, 3]
    assert calls == ["hi\nho", "hi", "ho"]  # batch rejected, per-word correct


def test_lexicon_inserts_the_latin_space_like_sherpa():
    """A space id follows every voiced Latin word — before the next English word,
    a zh run, or punctuation alike (sherpa MatchaTtsLexicon parity)."""
    from nanobot_channel_voice.tts.matcha import EnglishToIpa

    tokens = {"。": 44, " ": 1, "h": 20, "i": 21}
    lex = {"你": [10], "好": [11], "你好": [10, 11]}
    fe = LexiconFrontend(
        lex, tokens,
        english=EnglishToIpa(tokens, phonemize=lambda _w: "hi"),
        latin_space_id=1,
    )
    (seq,) = fe.sentences("hi hi你好 hi。")
    assert seq == [20, 21, 1, 20, 21, 1, 10, 11, 20, 21, 1, 44]


def test_frame_ids_interleave_off_keeps_plain_sequences():
    from nanobot_channel_voice.tts.matcha import frame_ids

    class _Fe:
        def sentences(self, _text):
            return [[5, 6], [7]]

    assert frame_ids(_Fe(), "x", bos_id=None, eos_id=None, pad_id=0,
                     interleave=False) == [5, 6, 7]
    assert frame_ids(_Fe(), "x", bos_id=1, eos_id=2, pad_id=0,
                     interleave=False) == [1, 5, 6, 2, 1, 7, 2]


def test_from_config_detects_the_bilingual_zh_en_export(monkeypatch, tmp_path):
    """voice: "zh en-us" metadata (no jieba/has_espeak keys) selects the hybrid
    frontend, kills the interleave, and declares both languages."""
    from nanobot_channel_voice.config import MatchaTtsConfig
    from nanobot_channel_voice.tts import matcha

    (tmp_path / "tokens.txt").write_text(
        " 1\n, 4\n。 44\nh 20\ni 21\nni3 30\nhao3 31\n", encoding="utf-8"
    )
    (tmp_path / "lexicon.txt").write_text(
        "你 ni3\n好 hao3\n你好 ni3 hao3\n", encoding="utf-8"
    )

    class FakeModel:
        def __init__(self, path, **_kw):
            self.path = path
            self.calls = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def release(self):
            pass

        def input_specs(self):
            if "acoustic" in self.path:
                return [("x", 0, 0), ("x_length", 0, 0),
                        ("noise_scale", 0, 0), ("length_scale", 0, 0)]
            return [("mels", 0, 0)]

        def metadata(self):
            if "acoustic" in self.path:
                # the real model's exact keys: use_eos_bos lies (no ^/$ in tokens)
                return {"voice": "zh en-us", "language": "chinese English",
                        "sample_rate": "16000", "pad_id": "0", "use_eos_bos": "1"}
            return {}

        def output_names(self):
            return ["mel"] if "acoustic" in self.path else ["mag", "x", "y"]

        def input_shape(self, _name):
            return None

        def run(self, inputs):
            self.calls.append(inputs)
            if "acoustic" in self.path:
                return [np.zeros((1, 80, 4), np.float32)]
            return [np.ones((1, 513, 4), np.float32)] * 3

    monkeypatch.setattr(matcha, "OnDeviceModel", FakeModel)
    voices: list[str] = []
    monkeypatch.setattr(
        matcha, "make_ipa_phonemizer",
        lambda voice, **_k: voices.append(voice) or (lambda _w: "hi"),
    )
    base = {
        "acousticModelPath": str(tmp_path / "acoustic.onnx"),
        "vocoderPath": str(tmp_path / "vocos.onnx"),
        "tokensPath": str(tmp_path / "tokens.txt"),
    }
    with pytest.raises(ValueError, match="lexiconPath"):
        matcha.MatchaTtsAdapter.from_config(MatchaTtsConfig.model_validate(base))

    cfg = MatchaTtsConfig.model_validate(
        base | {"lexiconPath": str(tmp_path / "lexicon.txt"), "espeakVoice": "de"}
    )
    tts = matcha.MatchaTtsAdapter.from_config(cfg)
    assert voices == ["en-us"]  # the fold table is en-us-trained: espeakVoice ignored
    assert tts.spoken_language == "zh" and tts.spoken_languages == ("zh", "en")
    assert tts.output_rate == 16000
    assert tts._bos_id is None and tts._eos_id is None
    # no vocos metadata -> sherpa's own defaults (verified against its C++ source)
    assert tts._vocoder.stft == {"n_fft": 1024, "hop_length": 256, "center": True}

    wav = tts._synthesize_piece("hi你好。")
    assert wav.size > 0
    x = tts._acoustic.calls[0][0][1]
    # h i _sp ni3 hao3 。 — hybrid English + latin space, and NO blank interleave
    assert x.tolist() == [[20, 21, 1, 30, 31, 44]]


# ---- adapter verdicts and normalization -------------------------------------


def test_matcha_normalize_verbalizes_digits_for_zh_only():
    adapter = MatchaTtsAdapter.__new__(MatchaTtsAdapter)  # no models needed
    adapter.spoken_language = "zh"
    assert adapter._normalize("现在7点") == "现在七点"
    adapter.spoken_language = "en"
    assert adapter._normalize("at 7 pm") == "at 7 pm"  # espeak reads quantities itself
    # …but its grammar is cardinal-biased, so sequences are re-spaced for it.
    assert adapter._normalize("zip 94105") == "zip 9 4 1 0 5"


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
    assert verbalize_numbers_zh("100000001个") == "一亿零一个"  # a counter keeps it a quantity
    assert verbalize_numbers_zh("1,234") == "一千二百三十四"
    assert verbalize_numbers_zh("7:45") == "七点四十五分"
    assert verbalize_numbers_zh("8:00") == "八点"
    assert verbalize_numbers_zh("9:05") == "九点零五分"
    assert verbalize_numbers_zh("50%") == "百分之五十"
    assert verbalize_numbers_zh("3.14") == "三点一四"


def test_verbalize_numbers_zh_reads_sequences_digit_wise():
    # Mandarin years are digit-read; the cardinal reading is simply wrong.
    assert verbalize_numbers_zh("2026年") == "二零二六年"
    assert verbalize_numbers_zh("1949年10月1日") == "一九四九年十月一日"
    assert verbalize_numbers_zh("3年") == "三年"        # a duration, not a year
    assert verbalize_numbers_zh("10000年") == "一万年"   # 4-digit runs only
    # A leading zero marks an identifier; the cardinal reading deletes it.
    assert verbalize_numbers_zh("第007号") == "第零零七号"
    assert verbalize_numbers_zh("订单号0755") == "订单号零七五五"
    # Multi-dot runs kept a literal "." the lexicon cannot voice.
    assert verbalize_numbers_zh("IP是192.168.1.1") == "IP是一九二点一六八点一点一"
    assert verbalize_numbers_zh("版本3.14.2") == "版本三点十四点二"


def test_verbalize_numbers_zh_dates_currency_degrees():
    assert verbalize_numbers_zh("会议定在2026-08-19") == "会议定在二零二六年八月十九日"
    assert verbalize_numbers_zh("2026/8/1发布") == "二零二六年八月一日发布"
    assert verbalize_numbers_zh("08月09日截止") == "八月九日截止"  # date pad, not identifier
    assert verbalize_numbers_zh("三点05分") == "三点零五分"        # minutes keep their 零
    assert verbalize_numbers_zh("订单08号") == "订单零八号"        # 号 ids keep their pad
    assert verbalize_numbers_zh("价格$5.99") == "价格五点九九美元"
    assert verbalize_numbers_zh("¥199起") == "一百九十九元起"
    assert verbalize_numbers_zh("罚款￥1,000") == "罚款一千元"
    # A scale suffix rides in front of the relocated unit; ambiguous or already-suffixed
    # shapes fall back to the plain number reading (the symbol drops as OOV downstream).
    assert verbalize_numbers_zh("¥200万") == "二百万元"
    assert verbalize_numbers_zh("大概¥1.5万") == "大概一点五万元"
    assert verbalize_numbers_zh("¥199元起") == "¥一百九十九元起"
    assert verbalize_numbers_zh("¥200多万") == "¥二百多万"
    assert verbalize_numbers_zh("今天25°C") == "今天二十五摄氏度"
    assert verbalize_numbers_zh("烤箱调到200℃") == "烤箱调到二百摄氏度"
    assert verbalize_numbers_zh("零下3℉") == "零下三华氏度"
    # An invalid month or an impossible day is not a date; the id keeps its
    # digit-wise reading.
    assert verbalize_numbers_zh("单号1234-56-78") == "单号一二三四五六七八"
    assert verbalize_numbers_zh("编号2026-02-30") == "编号二零二六零二三零"


def test_verbalize_numbers_zh_sequences():
    assert verbalize_numbers_zh("请拨打13800138000") == "请拨打一三八零零一三八零零零"
    assert verbalize_numbers_zh("邮编100084") == "邮编一零零零八四"
    assert verbalize_numbers_zh("验证码是482913") == "验证码是四八二九一三"
    assert verbalize_numbers_zh("分机8021") == "分机八零二一"
    assert verbalize_numbers_zh("车牌京A88888") == "车牌京A八八八八八"
    assert verbalize_numbers_zh("2020-2024年") == "二零二零到二零二四年"
    # A unit or counter after the run keeps it a quantity.
    assert verbalize_numbers_zh("账单总额1234元") == "账单总额一千二百三十四元"
    assert verbalize_numbers_zh("一共1234个文件") == "一共一千二百三十四个文件"
    assert verbalize_numbers_zh("大概需要45分钟") == "大概需要四十五分钟"
    # Ungrouped and long is an identifier: a quantity that size carries a counter.
    assert verbalize_numbers_zh("100000001") == "一零零零零零零零一"
    assert verbalize_numbers_zh("总额 10000000 元") == "总额 一千万 元"  # counter behind a space


# ---- the espeak resolution ladder -------------------------------------------


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

    def no_loader(_root=None):
        raise ImportError("espeakng_loader")

    monkeypatch.setattr(espeak.shutil, "which", lambda _: None)
    monkeypatch.setattr(espeak, "_load_library", no_loader)
    with pytest.raises(RuntimeError) as exc:
        espeak.make_ipa_phonemizer("en-us")  # no binary AND no espeakng-loader
    msg = str(exc.value)
    assert "espeakPath" in msg and "[espeak]" in msg and "apt install" in msg


def test_split_path_refuses_undeclared_zh_en_artifacts(tmp_path):
    """Framed like every other dialect (blank interleave, pinyin English), zh-en
    artifacts synthesize fluent rhythm over wrong sounds, so an UNDECLARED zh-en
    table must refuse — the dialect needs the exporter's word (meta.json
    {"frontend": "zh-en-lexicon"}), not a heuristic's. Both halves of the signature are
    load-bearing: an espeak table carries the capitals alone, a zh-only table the
    tonal pinyin alone."""
    import string

    from nanobot_channel_voice.config import MatchaTtsConfig
    from nanobot_channel_voice.tts.matcha import SplitMatchaTtsAdapter, is_zh_en_tokens

    assert is_zh_en_tokens(dict.fromkeys([*"AIOWY", "zhong1"], 1))
    assert not is_zh_en_tokens({"zhong1": 1, "ong1": 2, "zh": 3})  # zh-only export
    # A character-level English table holds every capital (see _OFFICIAL_LETTERS).
    assert not is_zh_en_tokens(dict.fromkeys(string.ascii_letters, 1))
    tokens = tmp_path / "tokens.txt"
    tokens.write_text(
        "\n".join(f"{t} {i}" for i, t in enumerate(["_", *"AIOWY", "zhong1"])), encoding="utf-8"
    )
    cfg = MatchaTtsConfig.model_validate({
        "encoderPath": str(tmp_path / "enc.onnx"), "decoderPath": str(tmp_path / "dec.onnx"),
        "vocoderPath": str(tmp_path / "voc.onnx"), "tokensPath": str(tokens),
    })
    with pytest.raises(ValueError, match="frontend.*zh-en-lexicon"):
        SplitMatchaTtsAdapter.from_config(cfg)
    # An unknown declaration is a loud config error, not a silent fallback.
    (tmp_path / "meta.json").write_text('{"frontend": "bilingual"}', encoding="utf-8")
    cfg = cfg.model_copy(update={"meta_path": str(tmp_path / "meta.json")})
    with pytest.raises(ValueError, match="expected zh-en-lexicon"):
        SplitMatchaTtsAdapter.from_config(cfg)


class _SplitFakeModel:
    """The OnDeviceModel protocol as the split path exercises it: RKNN-like (no
    introspectable geometry), encoder/decoder keyed on path suffix, Vocos vocoder.
    ``made`` collects construction order; clear it per test."""

    made: list["_SplitFakeModel"] = []

    def __init__(self, path, **_kw):
        self.path = path
        self.calls = []
        self.made.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def release(self):
        pass

    def input_shape(self, _name):
        return None  # RKNN-like: geometry not introspectable

    def metadata(self):
        return {}

    def run(self, inputs):
        self.calls.append(inputs)
        if self.path.endswith("encoder.rknn"):
            # One mel frame per input token makes the static contract visible.
            return [np.zeros((1, 80, 200), np.float32), np.zeros((1, 1, 200), np.float32)]
        if self.path.endswith("decoder.rknn"):
            return [np.zeros((1, 80, 16), np.float32)]
        return [
            np.ones((1, 513, 16), np.float32),
            np.ones((1, 513, 16), np.float32),
            np.zeros((1, 513, 16), np.float32),
        ]


def test_split_builds_declared_zh_en_dialect(monkeypatch, tmp_path):
    """meta.json {"frontend": "zh-en-lexicon"} is the exporter's word: the split builds the
    bilingual frontend with the dialect's own framing — no blank interleave, espeak
    English, the Latin-run space id — instead of refusing on the token heuristic."""
    from nanobot_channel_voice.config import MatchaTtsConfig
    from nanobot_channel_voice.tts import matcha

    (tmp_path / "tokens.txt").write_text("1\nni3 2\nh 3\nI 4\n. 5\n", encoding="utf-8")
    (tmp_path / "lexicon.txt").write_text("你 ni3\n", encoding="utf-8")
    (tmp_path / "meta.json").write_text(
        '{"frontend": "zh-en-lexicon", "encoder_len": 200, "mel_len": 16, '
        '"mel_scale": 1.0, "mel_bias": 0.0, "sample_rate": 16000}', encoding="utf-8"
    )
    _SplitFakeModel.made.clear()
    monkeypatch.setattr(matcha, "OnDeviceModel", _SplitFakeModel)
    monkeypatch.setattr(matcha, "make_ipa_phonemizer", lambda *_a, **_k: lambda _t: "haɪ")
    tts = matcha.SplitMatchaTtsAdapter.from_config(MatchaTtsConfig.model_validate({
        "encoderPath": str(tmp_path / "encoder.rknn"),
        "decoderPath": str(tmp_path / "decoder.rknn"),
        "vocoderPath": str(tmp_path / "vocoder.rknn"),
        "tokensPath": str(tmp_path / "tokens.txt"),
        "lexiconPath": str(tmp_path / "lexicon.txt"),
    }))
    assert tts.spoken_language == "zh" and tts.spoken_languages == ("zh", "en")
    assert tts.output_rate == 16000  # the sidecar's rate, not the 22.05 kHz default
    # 你 via lexicon, "hi" via espeak IPA (haɪ folds to hI), the Latin-run space id,
    # then the period — RAW ids: the dialect trains without the blank interleave.
    assert tts._ids("你hi.") == [2, 3, 4, 1, 5]
    assert tts._synthesize_piece("你hi.").dtype == np.float32  # host bridge runs


def test_split_sidecar_declares_framing_and_a_stale_sidecar_contradicts(monkeypatch, tmp_path):
    """pad_id/use_eos_bos come from the sidecar when declared (the token-table
    conventions are only the fallback), and a lexicon/espeak declaration beside a
    zh-en token table is refused as a wrong sidecar."""
    from nanobot_channel_voice.config import MatchaTtsConfig
    from nanobot_channel_voice.tts import matcha

    (tmp_path / "tokens.txt").write_text("_ 0\n^ 1\n$ 2\n 3\n. 4\nh 5\ni 6\n", encoding="utf-8")
    (tmp_path / "meta.json").write_text(
        '{"use_eos_bos": 0, "pad_id": 7, "mel_len": 16}', encoding="utf-8"
    )
    _SplitFakeModel.made.clear()
    monkeypatch.setattr(matcha, "OnDeviceModel", _SplitFakeModel)
    monkeypatch.setattr(matcha, "make_ipa_phonemizer", lambda *_a, **_k: lambda _t: "hi")
    cfg = MatchaTtsConfig.model_validate({
        "encoderPath": str(tmp_path / "encoder.rknn"),
        "decoderPath": str(tmp_path / "decoder.rknn"),
        "vocoderPath": str(tmp_path / "vocoder.rknn"),
        "tokensPath": str(tmp_path / "tokens.txt"),
    })
    tts = matcha.SplitMatchaTtsAdapter.from_config(cfg)
    assert tts._pad_id == 7
    # use_eos_bos=0 suppresses the ^/$ framing the table alone would have implied.
    assert tts._ids("hi.") == [7, 5, 7, 6, 7, 4, 7]

    zh_en_tokens = tmp_path / "zh-en-tokens.txt"
    zh_en_tokens.write_text(
        "\n".join(f"{t} {i}" for i, t in enumerate(["_", *"AIOWY", "zhong1"])), encoding="utf-8"
    )
    (tmp_path / "meta.json").write_text('{"frontend": "lexicon"}', encoding="utf-8")
    with pytest.raises(ValueError, match="wrong sidecar"):
        matcha.SplitMatchaTtsAdapter.from_config(
            cfg.model_copy(update={"tokens_path": str(zh_en_tokens)})
        )


def test_espeak_data_dir_pins_the_models_own_voice_pack(monkeypatch, tmp_path):
    """IPA spellings drift between espeak releases (en-us FORCE: oː -> ɔː), and a
    drifted vowel lands on a different embedding id, so the phonemizer must run against
    the data the model was trained with — passed as espeak's parent-of-data --path."""
    from nanobot_channel_voice.tts import espeak

    (tmp_path / "espeak-ng-data").mkdir()
    argv: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = "ˈoːɹ\n"
        stderr = ""

    monkeypatch.setattr(espeak.shutil, "which", lambda _: "/usr/bin/espeak-ng")
    monkeypatch.setattr(espeak.subprocess, "run", lambda cmd, **kw: argv.append(cmd) or _Proc())
    espeak.make_ipa_phonemizer("en-us", data_dir=str(tmp_path / "espeak-ng-data"))("more")
    assert f"--path={tmp_path}" in argv[-1]  # the PARENT, per espeak's convention
    # Either spelling of the same directory resolves identically...
    espeak.make_ipa_phonemizer("en-us", data_dir=str(tmp_path))("more")
    assert f"--path={tmp_path}" in argv[-1]
    # ...and no data dir leaves the installed espeak's own data in charge.
    espeak.make_ipa_phonemizer("en-us")("more")
    assert not any(a.startswith("--path") for a in argv[-1])
    with pytest.raises(RuntimeError, match="espeakDataDir"):
        espeak.make_ipa_phonemizer("en-us", data_dir=str(tmp_path / "nope"))


def test_espeak_data_dir_defaults_to_a_sibling_of_the_model(tmp_path):
    from nanobot_channel_voice.config import MatchaTtsConfig
    from nanobot_channel_voice.tts.matcha import _espeak_data_dir

    (tmp_path / "espeak-ng-data").mkdir()
    (tmp_path / "model.onnx").touch()
    cfg = MatchaTtsConfig.model_validate({"acousticModelPath": str(tmp_path / "model.onnx")})
    assert _espeak_data_dir(cfg) == str(tmp_path / "espeak-ng-data")
    # An explicit setting always wins, and a model without a pack claims none.
    assert _espeak_data_dir(
        MatchaTtsConfig.model_validate({
            "acousticModelPath": str(tmp_path / "model.onnx"), "espeakDataDir": "/opt/pack",
        })
    ) == "/opt/pack"
    assert _espeak_data_dir(
        MatchaTtsConfig.model_validate({"acousticModelPath": str(tmp_path / "sub/m.onnx")})
    ) is None


def test_lexicon_mismatch_against_the_token_table_is_reported(tmp_path):
    """Wrong-model file pairing keeps every id in range but points it at another
    phone: fluent rhythm over wrong sounds, which no exception would ever surface."""
    from loguru import logger as loguru_logger

    from nanobot_channel_voice.tts.matcha import load_lexicon

    lex = tmp_path / "lexicon.txt"
    lex.write_text("\n".join(["你好 ni3 hao3", *(f"字{i} zi{i}" for i in range(9))]), encoding="utf-8")
    messages: list[str] = []
    sink = loguru_logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        word2ids = load_lexicon(str(lex), {"ni3": 1, "hao3": 2})
    finally:
        loguru_logger.remove(sink)
    assert word2ids == {"你好": [1, 2]}
    assert any("same model" in m for m in messages)


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


# ---- the static split -------------------------------------------------------


def test_matcha_split_requires_the_complete_static_contract():
    from nanobot_channel_voice.config import MatchaTtsConfig
    from nanobot_channel_voice.tts.matcha import SplitMatchaTtsAdapter

    with pytest.raises(ValueError, match="encoderPath"):
        SplitMatchaTtsAdapter.from_config(MatchaTtsConfig())


def test_matcha_registry_routes_dynamic_first(monkeypatch):
    from nanobot_channel_voice.config import TtsConfig
    from nanobot_channel_voice.tts import _build_matcha, matcha

    monkeypatch.setattr(
        matcha.SplitMatchaTtsAdapter, "from_config", classmethod(lambda _c, m: ("static", m))
    )
    monkeypatch.setattr(
        matcha.MatchaTtsAdapter, "from_config",
        classmethod(lambda _c, m, **_kw: ("dynamic", m)),
    )
    enc = TtsConfig.model_validate({"provider": "matcha", "matcha": {"encoderPath": "e.rknn"}})
    assert _build_matcha(enc)[0] == "static"
    # vocoderPath is ALSO the dynamic mel-export field: with an acoustic graph it is
    # dynamic; alone it is an error naming the missing contract, never "static".
    dyn = TtsConfig.model_validate({
        "provider": "matcha",
        "matcha": {"acousticModelPath": "a.onnx", "vocoderPath": "v.onnx"},
    })
    assert _build_matcha(dyn)[0] == "dynamic"
    with pytest.raises(ValueError, match="acousticModelPath"):
        _build_matcha(TtsConfig.model_validate(
            {"provider": "matcha", "matcha": {"vocoderPath": "v.onnx"}}
        ))


def test_matcha_config_rejects_both_contracts_at_parse_time():
    from nanobot_channel_voice.config import MatchaTtsConfig

    with pytest.raises(ValueError, match="mutually exclusive"):
        MatchaTtsConfig.model_validate(
            {"acousticModelPath": "a.onnx", "encoderPath": "e.rknn"}
        )


def test_matcha_split_host_bridge_tiles_decoder_and_edges_vocos(monkeypatch, tmp_path):
    """The adapter's host-side glue follows the static split contract."""
    from nanobot_channel_voice.config import MatchaTtsConfig
    from nanobot_channel_voice.tts import matcha

    tokens = tmp_path / "tokens.txt"
    # A minimal valid icefall-style English table. The fake phonemizer below only emits h/i.
    tokens.write_text("_ 0\n^ 1\n$ 2\n 3\n. 4\nh 5\ni 6\n", encoding="utf-8")

    made = _SplitFakeModel.made
    made.clear()
    monkeypatch.setattr(matcha, "OnDeviceModel", _SplitFakeModel)
    monkeypatch.setattr(matcha, "make_ipa_phonemizer", lambda *_args, **_kw: lambda _text: "hi")
    cfg = MatchaTtsConfig.model_validate({
        "encoderPath": str(tmp_path / "encoder.rknn"),
        "decoderPath": str(tmp_path / "decoder.rknn"),
        "vocoderPath": str(tmp_path / "vocoder.rknn"),
        "tokensPath": str(tokens),
        "encoderLen": 200,
        "melLen": 16,
    })
    tts = matcha.SplitMatchaTtsAdapter.from_config(cfg)
    wav = tts._synthesize_piece("hi.")

    # [blank, BOS, blank, h, blank, i, blank, '.', blank, EOS, blank] => 11 frames.
    assert wav.dtype == np.float32 and wav.size == 11 * 256
    decoder_inputs = dict(made[1].calls[0])
    assert decoder_inputs["mask"].shape == (1, 1, 16)
    assert np.all(decoder_inputs["mask"] == 1)
    # mu and z tile with ONE period: work_len = 4*ceil(11/4) = 12 for both.
    assert np.array_equal(decoder_inputs["z"][:, :, 12:], decoder_inputs["z"][:, :, :4])
    assert np.array_equal(
        decoder_inputs["mu_up"][:, :, 12:], decoder_inputs["mu_up"][:, :, :4]
    )
    # calls[0] is the build-time mode/bias probe (zero mel); calls[-1] the synth.
    assert not dict(made[2].calls[0])["mels"].any()
    vocoder_inputs = dict(made[2].calls[-1])
    assert vocoder_inputs["mels"].shape == (1, 80, 16)
    tts.release()


def test_matcha_split_meta_sidecar_fills_unset_fields_only(monkeypatch, tmp_path):
    """Precedence: explicit config > meta.json sidecar > en_US-ljspeech defaults.
    The sidecar is how a store key ships zh-baker's OWN mel statistics - the wrong
    pair is not an error anywhere, just audibly wrong audio."""
    import json as jsonlib

    from nanobot_channel_voice.config import MatchaTtsConfig
    from nanobot_channel_voice.tts import matcha

    tokens = tmp_path / "tokens.txt"
    tokens.write_text("_ 0\n^ 1\n$ 2\n 3\n", encoding="utf-8")
    meta = tmp_path / "meta.json"
    meta.write_text(jsonlib.dumps({
        "mel_scale": 2.7628188, "mel_bias": -5.9870973, "encoder_len": 100, "mel_len": 400,
    }), encoding="utf-8")

    class FakeModel:
        def __init__(self, path, **_kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def release(self):
            pass

        def input_shape(self, _name):
            return None  # RKNN-like: geometry not introspectable

        def metadata(self):
            return {}

        def run(self, inputs):  # only the build-time vocoder probe lands here
            ((_, mel),) = inputs
            t = mel.shape[2]
            return [np.ones((1, 513, t), np.float32)] * 3

    monkeypatch.setattr(matcha, "OnDeviceModel", FakeModel)
    monkeypatch.setattr(matcha, "make_ipa_phonemizer", lambda *_a, **_k: str)
    base = {
        "encoderPath": str(tmp_path / "encoder.rknn"),
        "decoderPath": str(tmp_path / "decoder.rknn"),
        "vocoderPath": str(tmp_path / "vocoder.rknn"),
        "tokensPath": str(tokens),
    }
    # No metaPath needed: the sidecar beside encoderPath is auto-discovered (the
    # documented hand-wired route).
    tts = matcha.SplitMatchaTtsAdapter.from_config(MatchaTtsConfig.model_validate(base))
    assert (tts._mel_scale, tts._mel_bias) == (2.7628188, -5.9870973)
    assert (tts._encoder_len, tts._mel_len) == (100, 400)

    tts = matcha.SplitMatchaTtsAdapter.from_config(
        MatchaTtsConfig.model_validate({**base, "melScale": 9.0, "melLen": 8})
    )
    assert (tts._mel_scale, tts._mel_bias) == (9.0, -5.9870973)  # explicit beats sidecar
    assert (tts._encoder_len, tts._mel_len) == (100, 8)

    meta.unlink()  # no sidecar anywhere -> the en_US-ljspeech defaults
    tts = matcha.SplitMatchaTtsAdapter.from_config(MatchaTtsConfig.model_validate(base))
    assert (tts._mel_scale, tts._mel_bias) == (2.0661438, -5.5238085)
    assert (tts._encoder_len, tts._mel_len) == (200, 800)


def test_split_length_regulator_matches_the_dense_alignment():
    """The searchsorted gather must equal the reference one-hot alignment matmul."""
    from nanobot_channel_voice.tts.matcha import SplitMatchaTtsAdapter

    adapter = SplitMatchaTtsAdapter.__new__(SplitMatchaTtsAdapter)  # no models needed
    adapter._speed = 1.0
    rng = np.random.default_rng(2)
    token_len, table_len = 7, 12
    mu = rng.standard_normal((1, 4, table_len)).astype(np.float32)
    logw = rng.standard_normal((1, 1, table_len)).astype(np.float32)

    mu_up, total = adapter._length_regulator(mu, logw, token_len)

    durations = np.ceil(np.exp(logw[0, 0, :token_len]))
    assert total == int(durations.sum())
    work_len = int(4 * np.ceil(total / 4))
    ends = np.cumsum(durations)
    spans = (np.arange(work_len)[None, :] < ends[:, None]).astype(np.float32)
    spans[1:] -= spans[:-1]
    dense = np.einsum("bcl,lm->bcm", mu[:, :, :token_len], spans)
    assert mu_up.shape == (1, 4, work_len)
    assert np.allclose(mu_up, dense, atol=1e-6)


def test_matcha_split_uses_lexicon_frontend_for_zh(monkeypatch, tmp_path):
    """A static icefall zh-baker split has no ^/$ framing or espeak dependency."""
    from nanobot_channel_voice.config import MatchaTtsConfig
    from nanobot_channel_voice.tts import matcha

    tokens = tmp_path / "tokens.txt"
    tokens.write_text("  0\n_ 1\n。 2\nni3 3\nhao3 4\nqi1 5\n", encoding="utf-8")
    lexicon = tmp_path / "lexicon.txt"
    lexicon.write_text("你 ni3\n好 hao3\n七 qi1\n", encoding="utf-8")
    made = _SplitFakeModel.made
    made.clear()
    monkeypatch.setattr(matcha, "OnDeviceModel", _SplitFakeModel)
    # The zh TOKENIZATION never needs espeak; the optional English-fallback tier
    # builds one at from_config, so hand it a benign stand-in.
    monkeypatch.setattr(matcha, "make_ipa_phonemizer", lambda *_args, **_kw: str)
    cfg = MatchaTtsConfig.model_validate({
        "encoderPath": str(tmp_path / "encoder.rknn"),
        "decoderPath": str(tmp_path / "decoder.rknn"),
        "vocoderPath": str(tmp_path / "vocoder.rknn"),
        "tokensPath": str(tokens),
        "lexiconPath": str(lexicon),
        "encoderLen": 200,
        "melLen": 16,
    })
    tts = matcha.SplitMatchaTtsAdapter.from_config(cfg)
    assert tts.spoken_language == "zh"
    assert tts._normalize("你好7。") == "你好七。"
    # No BOS/EOS: [pad, ni3, pad, hao3, pad, qi1, pad, 。, pad].
    assert tts._ids("你好七。") == [1, 3, 1, 4, 1, 5, 1, 2, 1]
    wav = tts._synthesize_piece("你好七。")
    assert wav.size == 9 * 256
    encoder_inputs = dict(made[0].calls[0])
    assert encoder_inputs["x_length"].tolist() == [9]
    assert encoder_inputs["x"][0, :9].tolist() == [1, 3, 1, 4, 1, 5, 1, 2, 1]
    # The tail pads with the TABLE's pad id (zh "_" is 1, not 0 - id 0 is a real token).
    assert set(encoder_inputs["x"][0, 9:].tolist()) == {1}
    tts.release()


# ---- denoiser and edge fades ------------------------------------------------


def test_stft_round_trips_through_istft():
    rng = np.random.default_rng(3)
    sig = rng.standard_normal(5000).astype(np.float32) * 0.3
    mag, cos, sin = stft(sig, n_fft=1024, hop_length=256, center=True)
    assert mag.shape[0] == 513
    out = istft(mag, cos, sin, n_fft=1024, hop_length=256, center=True)
    n = min(out.size, sig.size)
    assert np.allclose(out[:n], sig[:n], atol=1e-4)


def test_denoise_preserves_length_and_a_huge_bias_silences():
    rng = np.random.default_rng(4)
    wav = rng.standard_normal(5000).astype(np.float32) * 0.2  # not a hop multiple
    out = denoise(wav, np.zeros((513, 1), np.float32))
    assert out.size == wav.size and np.allclose(out, wav, atol=1e-4)
    silenced = denoise(wav, np.full((513, 1), 100.0, np.float32))
    assert silenced.size == wav.size and np.abs(silenced).max() < 1e-6
    tiny = np.ones(300, np.float32)
    assert denoise(tiny, np.zeros((513, 1), np.float32)) is tiny  # too short to pad


def test_denoiser_bias_probes_the_vocoder_at_its_own_geometry():
    calls = []

    class Voc:
        shape = None  # dynamic graph

        def input_shape(self, _name):
            return self.shape

        def run(self, inputs):
            calls.append(inputs)
            ((_, mel),) = inputs
            return [np.full((1, mel.shape[2] * 256), 0.01, np.float32)]

    spec = VocoderSpec(Voc(), "mels", None)
    bias = denoiser_bias(spec, 0.00025)
    ((name, mel),) = calls[0]
    assert name == "mels" and mel.shape == (1, 80, 88) and not mel.any()
    assert bias.shape == (513, 1) and (bias >= 0).all()

    fixed = Voc()
    fixed.shape = (1, 80, 800)  # RKNN-style fixed bucket: probed as-is, never 88
    assert denoiser_bias(VocoderSpec(fixed, "mels", None), 0.00025) is not None
    assert calls[-1][0][1].shape == (1, 80, 800)

    # Vocos, strength 0, and a failing probe all disable rather than build or raise.
    assert denoiser_bias(VocoderSpec(Voc(), "mels", {"n_fft": 1024}), 0.00025) is None
    assert denoiser_bias(spec, 0.0) is None

    class Broken:
        def input_shape(self, _name):
            return None

        def run(self, inputs):
            raise RuntimeError("bad graph")

    assert denoiser_bias(VocoderSpec(Broken(), "mels", None), 0.00025) is None


def test_edge_fade_ramps_piece_boundaries():
    adapter = MatchaTtsAdapter.__new__(MatchaTtsAdapter)
    adapter.output_rate = 22050
    out = adapter._edge_fade(np.ones(2000, np.float32))
    n = int(0.005 * 22050)
    assert out[0] == 0.0 and out[-1] == 0.0
    assert out[n : 2000 - n].min() == 1.0  # only the edges are touched
    assert adapter._edge_fade(np.ones(1, np.float32))[0] == 1.0  # too short: untouched


def _waveform_adapter(vocoder_model, bias):
    class Acoustic:
        def run(self, inputs):
            return [np.zeros((1, 80, 8), np.float32)]

    class Frontend:
        def sentences(self, text):
            return [[5, 6]]

        def can_speak(self, ch):
            return True

    return MatchaTtsAdapter(
        acoustic=Acoustic(), vocoder=VocoderSpec(vocoder_model, "mels", None),
        frontend=Frontend(), official=False, length_input="x_length", interleave=True,
        sample_rate=22050, pad_id=0, bos_id=None, eos_id=None, spk_input=None,
        speaker_id=0, noise_scale=0.667, speed=1.0, max_len=300, language="en",
        denoise_bias=bias,
    )


def test_waveform_vocoder_output_is_denoised_and_faded():
    class Voc:
        def run(self, inputs):
            rng = np.random.default_rng(1)
            return [rng.standard_normal((1, 4096)).astype(np.float32) * 0.1]

    plain = _waveform_adapter(Voc(), None)._synthesize_piece("hi")
    huge = np.full((513, 1), 10.0, np.float32)
    denoised = _waveform_adapter(Voc(), huge)._synthesize_piece("hi")
    assert plain.size == denoised.size == 4096
    assert plain[0] == 0.0 and plain[-1] == 0.0  # edge fade applied
    assert np.abs(denoised).max() < np.abs(plain).max() * 0.01


def test_matcha_split_waveform_vocoder_is_probed_and_denoised(monkeypatch, tmp_path):
    """A 1-output vocoder flips the split to waveform mode: factor from the probe,
    denoiser bias from the same run, wav sliced to the real frames."""
    from nanobot_channel_voice.config import MatchaTtsConfig
    from nanobot_channel_voice.tts import matcha

    tokens = tmp_path / "tokens.txt"
    tokens.write_text("_ 0\n^ 1\n$ 2\n 3\n. 4\nh 5\ni 6\n", encoding="utf-8")
    made = []

    class FakeModel:
        def __init__(self, path, **_kw):
            self.path = path
            self.calls = []
            made.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def release(self):
            pass

        def input_shape(self, _name):
            return None

        def metadata(self):
            return {}

        def run(self, inputs):
            self.calls.append(inputs)
            if self.path.endswith("encoder.rknn"):
                return [np.zeros((1, 80, 200), np.float32), np.zeros((1, 1, 200), np.float32)]
            if self.path.endswith("decoder.rknn"):
                return [np.zeros((1, 80, 16), np.float32)]
            rng = np.random.default_rng(2)
            return [rng.standard_normal((1, 16 * 256)).astype(np.float32) * 0.05]

    monkeypatch.setattr(matcha, "OnDeviceModel", FakeModel)
    monkeypatch.setattr(matcha, "make_ipa_phonemizer", lambda *_args, **_kw: lambda _text: "hi")
    base = {
        "encoderPath": str(tmp_path / "encoder.rknn"),
        "decoderPath": str(tmp_path / "decoder.rknn"),
        "vocoderPath": str(tmp_path / "vocoder.rknn"),
        "tokensPath": str(tokens),
        "encoderLen": 200,
        "melLen": 16,
    }
    tts = matcha.SplitMatchaTtsAdapter.from_config(MatchaTtsConfig.model_validate(base))
    assert tts._stft is None and tts._voc_factor == 256
    assert tts._denoise_bias is not None  # the default strength rides the probe
    wav = tts._synthesize_piece("hi.")
    assert wav.size == 11 * 256 and wav.dtype == np.float32
    assert wav[0] == 0.0 and wav[-1] == 0.0  # faded like the dynamic path
    tts.release()

    off = matcha.SplitMatchaTtsAdapter.from_config(
        MatchaTtsConfig.model_validate({**base, "denoiserStrength": 0})
    )
    assert off._denoise_bias is None
    off.release()


# ---- English-into-pinyin fallback -------------------------------------------


def test_english_letters_spell_with_token_validation():
    from nanobot_channel_voice.tts.pinyin_english import EnglishToPinyin

    tokens = {s: i for i, s in enumerate(["ei1", "si1", "bi4", "you1", "ai4"])}
    eng = EnglishToPinyin(tokens)
    assert bool(eng)
    # U-S-B: the Mandarin letter readings, each syllable a live token id.
    assert eng.word_ids("USB") == [
        tokens["you1"], tokens["ai4"], tokens["si1"], tokens["bi4"],
    ]
    # A letter whose syllables are absent drops silently ('c' needs xi1).
    assert eng.word_ids("UC") == [tokens["you1"]]
    # The tone ladder falls back to other tones / the toneless twin.
    bare = EnglishToPinyin({"bi": 7})
    assert bare.word_ids("B") == [7]
    # No resolvable letter at all => falsy => the frontend disables the tier.
    assert not EnglishToPinyin({"zzz": 0})


def test_english_words_transliterate_via_espeak_ipa():
    import pytest as _pytest

    from nanobot_channel_voice.tts.pinyin_english import EnglishToPinyin

    tokens = {s: i for i, s in enumerate(
        ["he1", "lou1", "ban1", "ei1", "ai4", "you1", "si1", "bi4"]
    )}
    ipa = {"hello": "həlˈoʊ", "ban": "bˈæn"}
    eng = EnglishToPinyin(tokens, phonemize=lambda w: ipa[w])
    assert eng.word_ids("hello") == [tokens["he1"], tokens["lou1"]]
    assert eng.word_ids("ban") == [tokens["ban1"]]  # nasal coda folds into the syllable
    # Acronyms never consult espeak (a raising phonemizer proves the routing).
    strict = EnglishToPinyin(
        tokens, phonemize=lambda w: _pytest.fail("acronym reached espeak")
    )
    assert strict.word_ids("USB") == [tokens["you1"], tokens["ai4"], tokens["si1"], tokens["bi4"]]
    # A word the mapping cannot carry falls back to spelling.
    sparse = EnglishToPinyin(tokens, phonemize=lambda w: "ʘ")
    assert sparse.word_ids("ab") == [tokens["ei1"], tokens["bi4"]]


def test_lexicon_frontend_voices_latin_runs_via_the_fallback():
    from nanobot_channel_voice.tts.pinyin_english import EnglishToPinyin

    tokens = fold_punct_aliases(
        {"。": 3, "ni3": 10, "hao3": 11, "you1": 20, "ai4": 21, "si1": 22,
         "bi4": 23, "ei1": 24, "o": 25}
    )
    lex = {"你": [10], "好": [11]}
    fe = LexiconFrontend(lex, tokens, english=EnglishToPinyin(tokens))
    assert fe.can_speak("w") and fe.can_speak("Z")
    (seq,) = fe.sentences("你好USB。")
    assert seq == [10, 11, 20, 21, 22, 23, 3]
    # The run is consumed WHOLE: 'o' must not leak as the pinyin vowel token.
    (seq2,) = fe.sentences("好ok。")
    assert 25 not in seq2
    # Without the fallback, behavior is unchanged: Latin drops and reports.
    fe0 = LexiconFrontend(lex, tokens)
    assert not fe0.can_speak("w")
    assert fe0.sentences("你好USB。") == [[10, 11, 3]]
