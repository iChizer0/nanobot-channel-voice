"""On-device Matcha-TTS (``tts.provider="matcha"``), export dialects detected from
the graph: OFFICIAL (``scales`` input, in-code symbol table, optional embedded
vocoder -> direct wav) and icefall/sherpa (metadata-driven front-end, tokens.txt
[+ lexicon.txt for zh], mel out) — including the bilingual zh-en flavour
(``voice: "zh en-us"``: lexicon zh + espeak English, no blank interleave). Mel needs
``vocoderPath``: Vocos (mag/cos/sin -> host ISTFT) or any single-output waveform
vocoder. Imported lazily by make_tts.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import NamedTuple

import numpy as np
from loguru import logger

from nanobot_channel_voice.config import MatchaTtsConfig
from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
from nanobot_channel_voice.tts.espeak import make_ipa_phonemizer
from nanobot_channel_voice.tts.ondevice_base import OnDeviceTtsAdapter
from nanobot_channel_voice.tts.pinyin_english import EnglishToPinyin
from nanobot_channel_voice.tts.text_frontend import (
    space_digit_sequences,
    verbalize_numbers_zh,
)

_JOIN_GAP_S = 0.1
_SAMPLE_RATE_DEFAULT = 22050  # every published matcha vocoder (HiFi-GAN, Vocos univ)

# metadata "language" -> ISO 639-1; unknown names leave tts.language in charge
_LANG_NAMES = {"english": "en", "chinese": "zh", "german": "de", "japanese": "ja"}


# CJK terminators split anywhere; ASCII only before whitespace/end ("3.14" survives)
_SENTENCE_SPLIT_RE = re.compile(r"(?:(?<=[。！？…])(?![。！？…])|(?<=[.!?])(?=\s|$))")
_CLAUSE_SPLIT_RE = re.compile(r"([、，；：]|[,;:](?=\s|$))")
_SENTENCE_PUNCT = ".!?…。！？"
_LANG_SWITCH_RE = re.compile(r"\([a-z0-9-]+\)")  # espeak "(zh)" language-switch flags
_LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")  # = LexiconFrontend's Latin runs

# Verbatim official symbol table; list position IS the embedding id
_OFFICIAL_PUNCTUATION = ';:,.!?¡¿—…"«»“” '
_OFFICIAL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_OFFICIAL_LETTERS_IPA = (
    "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
)


def official_token2id() -> dict[str, int]:
    symbols = ["_", *_OFFICIAL_PUNCTUATION, *_OFFICIAL_LETTERS, *_OFFICIAL_LETTERS_IPA]
    return {s: i for i, s in enumerate(symbols)}  # dup "'" keeps the last id, as upstream


def read_tokens(path: str) -> dict[str, int]:
    """sherpa-onnx ``tokens.txt``: ``<sym> <id>`` per line; a lone id means the
    symbol is the space character."""
    token2id: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            sym, tid = (" ", parts[0]) if len(parts) == 1 else (parts[0], parts[1])
            token2id[sym] = int(tid)
    if not token2id:
        # empty table => healthy-looking mute TTS
        raise ValueError(f"no tokens parsed from {path} (wrong format?)")
    return token2id


# Half/full-width twins: each table carries one spelling, agent text uses both
_PUNCT_TWINS = (
    (",", "，"), (".", "。"), ("!", "！"), ("?", "？"), (":", "："),
    ('"', "“"), ('"', "”"), ("'", "‘"), ("'", "’"), (";", "；"),
)


def fold_punct_aliases(token2id: dict[str, int]) -> dict[str, int]:
    for half, full in _PUNCT_TWINS:
        if half in token2id and full not in token2id:
            token2id[full] = token2id[half]
        elif full in token2id and half not in token2id:
            token2id[half] = token2id[full]
    if "，" in token2id:
        token2id.setdefault("、", token2id["，"])
    if "," in token2id:
        token2id.setdefault(";", token2id[","])
    if "." in token2id:
        token2id.setdefault("…", token2id["."])
    return token2id


def _parse_lexicon(path: str, token2id: dict[str, int]) -> tuple[dict[str, list[int]], list[str]]:
    """``lexicon.txt``: ``<word> <phone>...`` per line, first spelling wins; a word
    with any unknown phone is dropped (sherpa behaviour) and returned by name."""
    word2ids: dict[str, list[int]] = {}
    dropped: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0].lower()
            if word in word2ids:
                continue
            try:
                word2ids[word] = [token2id[p] for p in parts[1:]]
            except KeyError:
                dropped.append(word)
    return word2ids, dropped


def load_lexicon(path: str, token2id: dict[str, int]) -> dict[str, list[int]]:
    word2ids, dropped = _parse_lexicon(path, token2id)
    # Heuristic, deliberately sensitive: a correct pairing drops ~0 (measured 0 of 68k
    # for zh-en, 4 of 66k for baker) while a cross-model one drops 2% (same pinyin
    # inventory, different ids) to ~100% (different alphabet). It only ever costs a log
    # line, and the failure it names is otherwise silent — right rhythm, wrong sounds.
    if len(dropped) * 100 > len(dropped) + len(word2ids):
        logger.warning(
            "voice: matcha lexicon '{}' dropped {} of {} entries whose phones are "
            "absent from the token table — are lexiconPath and tokensPath from the "
            "same model?", path, len(dropped), len(dropped) + len(word2ids),
        )
    return word2ids


def _load_lexicons(
    path: str, overrides_path: str | None, token2id: dict[str, int]
) -> dict[str, list[int]]:
    """Override entries win over the model lexicon."""
    word2ids = load_lexicon(path, token2id)
    if overrides_path:
        overrides, dropped = _parse_lexicon(overrides_path, token2id)
        if dropped:  # every override line is hand-authored: a drop is a typo, name it
            logger.warning(
                "voice: matcha lexicon overrides '{}' dropped {} entries whose phones "
                "are absent from the token table: {}",
                overrides_path, len(dropped), " ".join(dropped[:8]),
            )
        word2ids |= overrides
    return word2ids


def add_blank(ids: list[int], pad_id: int) -> list[int]:
    """[pad, t0, pad, t1, ..., pad]: the training-time interleave."""
    out = [pad_id] * (2 * len(ids) + 1)
    out[1::2] = ids
    return out


def frame_ids(
    frontend: EspeakFrontend | LexiconFrontend, text: str, *,
    bos_id: int | None, eos_id: int | None, pad_id: int, interleave: bool = True,
) -> list[int]:
    """Per-sentence bos/eos framing + blank interleave (shared by both adapters);
    the zh-en dialect trains without the interleave."""
    ids: list[int] = []
    for seq in frontend.sentences(text):
        if bos_id is not None:
            seq = [bos_id, *seq]
        if eos_id is not None:
            seq = [*seq, eos_id]
        ids.extend(add_blank(seq, pad_id) if interleave else seq)
    return ids


def _hann(n_fft: int) -> np.ndarray:
    return (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))).astype(np.float32)


def istft(mag: np.ndarray, cos: np.ndarray, sin: np.ndarray, *,
          n_fft: int, hop_length: int, center: bool) -> np.ndarray:
    """Vocos (bins, frames) mag/cos/sin -> waveform; torch.istft-compatible."""
    # in-place complex fill: the naive product allocates four full-size temporaries
    spec = np.empty(mag.shape, dtype=np.complex64)
    np.multiply(mag, cos, out=spec.real)
    np.multiply(mag, sin, out=spec.imag)
    frames = np.fft.irfft(spec, n=n_fft, axis=0).astype(np.float32)  # (n_fft, F)
    win = _hann(n_fft)
    frames *= win[:, None]
    n_frames = frames.shape[1]
    total = n_fft + hop_length * (n_frames - 1)
    out = np.zeros(total, dtype=np.float32)
    wsum = np.zeros(total, dtype=np.float32)
    w2 = win * win
    if n_fft % hop_length == 0:
        # n_fft/hop strided lane-adds; within a lane the frame targets never overlap
        for k in range(n_fft // hop_length):
            s = k * hop_length
            out[s : s + n_frames * hop_length].reshape(n_frames, hop_length)[...] += (
                frames[s : s + hop_length, :].T
            )
            wsum[s : s + n_frames * hop_length].reshape(n_frames, hop_length)[...] += (
                w2[s : s + hop_length][None, :]
            )
    else:
        for i in range(n_frames):
            s = i * hop_length
            out[s : s + n_fft] += frames[:, i]
            wsum[s : s + n_fft] += w2
    out /= np.maximum(wsum, 1e-11)
    return out[n_fft // 2 : -(n_fft // 2)] if center else out


def stft(wav: np.ndarray, *, n_fft: int, hop_length: int, center: bool
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Waveform -> (bins, frames) mag/cos/sin; the exact forward of :func:`istft`."""
    if center:
        wav = np.pad(wav, n_fft // 2, mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(wav, n_fft)[::hop_length].T
    spec = np.fft.rfft(frames * _hann(n_fft)[:, None], axis=0)
    mag = np.abs(spec).astype(np.float32)
    safe = np.maximum(mag, 1e-12)
    return mag, (spec.real / safe).astype(np.float32), (spec.imag / safe).astype(np.float32)


def _is_latin(ch: str) -> bool:
    """Non-Latin would be voiced as garbage IPA after espeak language-switches;
    refusing it lets the speakability guard skip + warn instead."""
    cp = ord(ch)
    return cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF  # ASCII..IPA block; Latin ext additional


def _sentences(text: str) -> list[tuple[str, str]]:
    """(body, trailing sentence punctuation) chunks; the model voices the pause."""
    ans = []
    for piece in _SENTENCE_SPLIT_RE.split(text):
        piece = piece.strip()
        if not piece:
            continue
        body = piece.rstrip(_SENTENCE_PUNCT)
        ans.append((body, piece[len(body) :]))
    return ans


class EspeakFrontend:
    """Text -> per-sentence token ids via espeak IPA; clauses phonemize separately
    because espeak drops the punctuation the model needs as pause tokens."""

    def __init__(self, token2id: dict[str, int], *, phonemize: Callable[[str], str]):
        self._token2id = token2id
        self._phonemize = phonemize
        self._space_id = token2id.get(" ")

    def can_speak(self, ch: str) -> bool:
        return _is_latin(ch)  # per-char, only the wrong-script failure is answerable

    def _ipa_ids(self, ipa: str) -> list[int]:
        # strip "(zh)" switch flags first: their letters are valid phoneme symbols
        ipa = _LANG_SWITCH_RE.sub("", ipa)
        ids: list[int] = []
        for ch in ipa:
            if ch.isspace():
                if self._space_id is not None and ids and ids[-1] != self._space_id:
                    ids.append(self._space_id)
            elif ch in self._token2id:
                ids.append(self._token2id[ch])
        while ids and ids[-1] == self._space_id:
            ids.pop()
        return ids

    def _phonemize_clauses(self, clauses: list[str]) -> list[str]:
        """One call per batch (espeak emits one clause per line); a line-count
        mismatch falls back to per-clause calls."""
        if len(clauses) > 1:
            lines = self._phonemize("\n".join(clauses)).splitlines()
            if len(lines) == len(clauses):
                return lines
        return [self._phonemize(c) for c in clauses]

    def sentences(self, text: str) -> list[list[int]]:
        chunks: list[tuple[list[str], list[str]]] = []  # (clauses, their punctuation)
        all_clauses: list[str] = []
        for body, final_punct in _sentences(text):
            parts = _CLAUSE_SPLIT_RE.split(body)
            clauses, puncts = [], []
            for clause, punct in zip(parts[::2], parts[1::2] + [final_punct[-1:]]):
                clauses.append(clause.strip())
                puncts.append(punct)
            chunks.append((clauses, puncts))
            all_clauses.extend(c for c in clauses if c)
        ipa = iter(self._phonemize_clauses(all_clauses)) if all_clauses else iter(())

        ans = []
        for clauses, puncts in chunks:
            seq: list[int] = []
            for clause, punct in zip(clauses, puncts):
                if clause:
                    if seq and self._space_id is not None:
                        seq.append(self._space_id)
                    seq.extend(self._ipa_ids(next(ipa)))
                if seq and punct in self._token2id:
                    seq.append(self._token2id[punct])
            if seq:
                ans.append(seq)
        return ans


# sherpa-onnx kReplacements (PR #2853): the zh-en model trains English on a folded
# espeak alphabet — diphthongs are single symbols, bare e/g/r never occur.
_ZH_EN_IPA_FOLD = (
    ("ɝ", "ɜɹ"), ("ɚ", "əɹ"),
    ("eɪ", "A"), ("aɪ", "I"), ("ɔɪ", "Y"), ("oʊ", "O"), ("əʊ", "O"), ("aʊ", "W"),
    ("tʃ", "ʧ"), ("dʒ", "ʤ"), ("ː", ""), ("g", "ɡ"), ("r", "ɹ"), ("e", "ɛ"),
)
_IPA_CACHE_CAP = 4096
_ZH_EN_FOLDED = frozenset("AIOWY")
_PINYIN_SYLLABLE = re.compile(r"[a-z]{1,6}[1-5]\Z")


def is_zh_en_tokens(token2id: dict[str, int]) -> bool:
    """Only the zh-en table carries BOTH halves: folded English diphthongs and tonal
    pinyin. A zh-only export has the pinyin alone, a character-level English table the
    capitals alone. For the dynamic path the graph metadata settles the dialect; the
    split path has no metadata and only this."""
    return _ZH_EN_FOLDED <= token2id.keys() and any(
        _PINYIN_SYLLABLE.match(token) for token in token2id
    )


class EnglishToIpa:
    """Word -> token ids for the bilingual zh-en table: espeak IPA folded to the
    trained alphabet. Same shape as EnglishToPinyin, so LexiconFrontend cannot
    tell native English from transliteration."""

    def __init__(self, token2id: dict[str, int], phonemize: Callable[[str], str]):
        self._token2id = token2id
        self._phonemize = phonemize
        self._cache: dict[str, list[int]] = {}

    def _fold_ids(self, ipa: str) -> list[int]:
        ipa = _LANG_SWITCH_RE.sub("", ipa)
        for src, dst in _ZH_EN_IPA_FOLD:
            ipa = ipa.replace(src, dst)
        return [self._token2id[ch] for ch in ipa if ch in self._token2id]

    def _put(self, word: str, ids: list[int]) -> list[int]:
        if len(self._cache) >= _IPA_CACHE_CAP:
            self._cache.clear()
        self._cache[word] = ids
        return ids

    def prime(self, words: list[str]) -> None:
        """Batch-phonemize uncached words in ONE espeak call — subprocess espeak
        spawns per call, so a fresh English clause would otherwise spawn per word."""
        fresh = list(dict.fromkeys(w for w in words if w not in self._cache))
        if len(fresh) < 2:
            return
        try:
            lines = self._phonemize("\n".join(fresh)).splitlines()
        except Exception:  # noqa: BLE001 - the per-word path reports instead
            return
        if len(lines) != len(fresh):
            return  # espeak re-claused the batch: per-word calls stay correct
        for word, ipa in zip(fresh, lines):
            self._put(word, self._fold_ids(ipa))

    def word_ids(self, word: str) -> list[int]:
        ids = self._cache.get(word)
        if ids is None:
            try:
                ids = self._put(word, self._fold_ids(self._phonemize(word)))
            except Exception:  # noqa: BLE001 - espeak hiccup drops the word, not the chunk
                return []
        return ids


class LexiconFrontend:
    """Text -> per-sentence token ids by greedy longest lexicon match, single chars
    falling back to the token table; OOV drops, the speakability guard reports.
    With an ``english`` resolver, Latin word runs voice through it instead of
    dropping: pinyin transliteration (see pinyin_english) or the zh-en model's
    real espeak English. ``latin_space_id`` mirrors sherpa's zh-en frontend: the
    id emitted after every voiced Latin word, before whatever follows."""

    def __init__(
        self,
        word2ids: dict[str, list[int]],
        token2id: dict[str, int],
        english: EnglishToPinyin | EnglishToIpa | None = None,
        latin_space_id: int | None = None,
    ):
        self._word2ids = word2ids
        self._token2id = token2id
        self._english = english if english else None  # falsy = no resolvable letters
        self._latin_space_id = latin_space_id
        # longest key per first char: one long entry must not slow every position
        self._max_by_first: dict[str, int] = {}
        for word in word2ids:
            if len(word) > self._max_by_first.get(word[0], 0):
                self._max_by_first[word[0]] = len(word)

    def can_speak(self, ch: str) -> bool:
        ch = ch.lower()
        if self._english is not None and "a" <= ch <= "z":
            return True
        return ch in self._word2ids or ch in self._token2id

    def _tokenize(self, text: str) -> list[int]:
        low = text.lower()
        if len(low) != len(text):
            text = low  # exotic case-fold expansion: keep alignment, forfeit case
        ids: list[int] = []
        after_latin = False

        def emit(seq: list[int], latin: bool = False) -> None:
            nonlocal after_latin
            if not seq:
                return
            if after_latin and self._latin_space_id is not None:
                ids.append(self._latin_space_id)
            ids.extend(seq)
            after_latin = latin

        i = 0
        while i < len(low):
            if self._english is not None and "a" <= low[i] <= "z":
                # Whole Latin run at once, ORIGINAL case (all-caps = acronym): a
                # per-char walk would leak letters that coincide with pinyin
                # syllables ("o" -> 哦) into the ids.
                j = i + 1
                while j < len(low) and ("a" <= low[j] <= "z" or low[j] == "'"):
                    j += 1
                emit(self._english.word_ids(text[i:j]), latin=True)
                i = j
                continue
            longest = self._max_by_first.get(low[i], 0)
            for ln in range(min(longest, len(low) - i), 0, -1):
                cand = low[i : i + ln]
                if cand in self._word2ids:
                    emit(self._word2ids[cand])
                    i += ln
                    break
            else:
                ch = low[i]
                if ch in self._token2id and not ch.isspace():
                    emit([self._token2id[ch]])
                i += 1
        return ids

    def sentences(self, text: str) -> list[list[int]]:
        prime = getattr(self._english, "prime", None)
        if prime is not None:
            prime(_LATIN_WORD_RE.findall(text))
        return [ids for body, punct in _sentences(text) if (ids := self._tokenize(body + punct))]


class VocoderSpec(NamedTuple):
    """Mel-consuming vocoder; ``stft`` set for Vocos (host ISTFT), None for HiFi-GAN."""

    model: OnDeviceModel
    input_name: str
    stft: dict | None


# Grad-TTS/WaveGlow denoiser: upstream's CLI subtracts the vocoder's zero-mel bias
# spectrum from every HiFi-GAN output, but the ONNX export only clamps — without this
# the export carries a constant hiss the demo doesn't. Vocos has no such bias, and an
# embedded vocoder can't be probed alone, so neither is denoised.
_DENOISE_STFT = {"n_fft": 1024, "hop_length": 256, "center": True}


def _bias_from_wav(wav: np.ndarray, strength: float) -> np.ndarray | None:
    """Strength-scaled bias magnitude (bins, 1) from a zero-mel probe waveform."""
    if strength <= 0:
        return None
    try:
        mag, _, _ = stft(wav, **_DENOISE_STFT)
    except Exception as exc:  # noqa: BLE001 - an enhancement must not fail the build
        logger.warning("voice: matcha denoiser disabled (bias probe failed: {})", exc)
        return None
    return mag[:, :1] * strength  # frame 0, as upstream


def denoiser_bias(vocoder: VocoderSpec, strength: float) -> np.ndarray | None:
    """Probe a waveform vocoder for its bias; None when denoising doesn't apply."""
    if vocoder.stft is not None or strength <= 0:
        return None
    shape = vocoder.model.input_shape(vocoder.input_name)
    frames = shape[2] if shape is not None and len(shape) == 3 else 88  # fixed graphs probe as-is
    try:
        wav = np.asarray(
            vocoder.model.run([(vocoder.input_name, np.zeros((1, 80, frames), np.float32))])[0],
            dtype=np.float32,
        ).reshape(-1)
    except Exception as exc:  # noqa: BLE001
        logger.warning("voice: matcha denoiser disabled (bias probe failed: {})", exc)
        return None
    return _bias_from_wav(wav, strength)


def denoise(wav: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Subtract ``bias`` in the magnitude domain, keeping the phase and the length."""
    hop = _DENOISE_STFT["hop_length"]
    if wav.size <= _DENOISE_STFT["n_fft"] // 2:  # reflect pad needs len > n_fft/2
        return wav
    # pad to a hop multiple: the center-ISTFT round trip yields floor(n/hop)*hop samples
    padded = np.pad(np.clip(wav, -1.0, 1.0), (0, -wav.size % hop))
    mag, cos, sin = stft(padded, **_DENOISE_STFT)
    out = istft(np.maximum(mag - bias, 0.0), cos, sin, **_DENOISE_STFT)
    return out[: wav.size]


def _espeak_data_dir(cfg: MatchaTtsConfig) -> str | None:
    """The voice pack this model was trained against: config, else the ``espeak-ng-data``
    beside the model files (every sherpa espeak export ships one). Absent it, the
    installed espeak's release picks the phonemes — en-us FORCE drifted oː -> ɔː, which
    lands English on a different embedding."""
    if cfg.espeak_data_dir:
        return cfg.espeak_data_dir
    for path in (cfg.acoustic_model_path, cfg.tokens_path, cfg.lexicon_path):
        if path and (data := Path(path).expanduser().parent / "espeak-ng-data").is_dir():
            return str(data)
    return None


def _english_fallback(cfg: MatchaTtsConfig, token2id: dict[str, int]) -> EnglishToPinyin:
    """The zh lexicon models' English tier; letters-only when no espeak resolves."""
    try:
        return EnglishToPinyin(
            token2id,
            make_ipa_phonemizer(
                "en-us", espeak_path=cfg.espeak_path, data_dir=_espeak_data_dir(cfg)
            ),
        )
    except Exception as exc:  # noqa: BLE001 - an optional tier, never a build error
        logger.info(
            "voice: matcha zh English fallback is letters-only (espeak unavailable: {})",
            exc,
        )
        return EnglishToPinyin(token2id)


def _espeak_frontend(
    cfg: MatchaTtsConfig, token2id: dict[str, int], default_voice: str = "en-us",
) -> tuple[EspeakFrontend, str]:
    """(frontend, language): the espeak voice IS the spoken language."""
    voice = cfg.espeak_voice or default_voice
    frontend = EspeakFrontend(
        token2id,
        phonemize=make_ipa_phonemizer(
            voice, espeak_path=cfg.espeak_path, data_dir=_espeak_data_dir(cfg)
        ),
    )
    prefix = voice.partition("-")[0].lower()
    return frontend, (prefix if len(prefix) == 2 else _LANG_NAMES.get(prefix, "en"))


class _MatchaCommon:
    """Frontend hooks shared by both adapters (needs _frontend/_max_len/spoken_language)."""

    def _piece_budget(self) -> int:
        return self._max_len  # type: ignore[attr-defined]

    def _normalize(self, text: str) -> str:
        # The zh lexicon has 零..九 but no "0".."9". espeak reads digits itself, but its
        # grammar is cardinal-biased, so only the sequences are re-spaced for it to name.
        if self.spoken_language == "zh":  # type: ignore[attr-defined]
            return verbalize_numbers_zh(text)
        return space_digit_sequences(text, self.spoken_language)  # type: ignore[attr-defined]

    def _can_speak(self, ch: str) -> bool:
        return self._frontend.can_speak(ch)  # type: ignore[attr-defined]

    def _edge_fade(self, wav: np.ndarray) -> np.ndarray:
        # 5 ms cosine ramps: pieces butt-join against silence, and a non-zero edge clicks
        n = min(int(0.005 * self.output_rate), wav.size // 2)  # type: ignore[attr-defined]
        if n > 0:
            ramp = (0.5 * (1.0 - np.cos(np.pi * np.arange(n) / n))).astype(np.float32)
            wav[:n] *= ramp
            wav[-n:] *= ramp[::-1]
        return wav


class MatchaTtsAdapter(_MatchaCommon, OnDeviceTtsAdapter):
    _label = "Matcha"
    _join_gap_s = _JOIN_GAP_S

    def __init__(
        self,
        *,
        acoustic: OnDeviceModel,
        vocoder: VocoderSpec | None,     # None => the acoustic graph outputs wav
        frontend: EspeakFrontend | LexiconFrontend,
        official: bool,                  # official export (scales input) vs icefall
        length_input: str,               # "x_lengths" (official) / "x_length" (icefall)
        interleave: bool,                # blank-interleave ids (all dialects but zh-en)
        sample_rate: int,
        pad_id: int,
        bos_id: int | None,              # None => no bos/eos framing
        eos_id: int | None,
        spk_input: str | None,           # "spks"/"sid" when the graph wants one
        speaker_id: int,
        noise_scale: float,
        speed: float,
        max_len: int,
        language: str | None,
        denoise_bias: np.ndarray | None,
    ):
        super().__init__()
        self._acoustic = acoustic
        self._vocoder = vocoder
        self._denoise_bias = denoise_bias
        self._frontend = frontend
        self._official = official
        self._length_input = length_input
        self._interleave = interleave
        self._pad_id = pad_id
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._spk_input = spk_input
        self._speaker_id = speaker_id
        self._noise_scale = noise_scale
        self._speed = speed
        self._max_len = max_len
        self.output_rate = sample_rate
        self.spoken_language = language
        self._log = logger.bind(component="tts-matcha")

    @classmethod
    def from_config(
        cls, cfg: MatchaTtsConfig, *, vocoder_share: VocoderSpec | None = None
    ) -> MatchaTtsAdapter:
        """``vocoder_share``: a sibling engine's already-loaded vocoder (bilingual
        router, same vocoderPath) — used instead of loading a second session, and
        deliberately NOT registered with this build's cleanup stack."""
        if not cfg.acoustic_model_path:
            raise ValueError("matcha dynamic export needs tts.matcha.acousticModelPath")
        model_kw = dict(
            core_mask=cfg.core_mask, target=cfg.target, device_id=cfg.device_id,
            providers=cfg.execution_providers, provider_options=cfg.provider_options,
        )
        with ExitStack() as models:
            acoustic = models.enter_context(
                OnDeviceModel(cfg.acoustic_model_path, **model_kw)  # type: ignore[arg-type]
            )
            input_names = [name for name, _, _ in acoustic.input_specs()]
            if not input_names:
                raise ValueError(
                    "matcha needs an introspectable ONNX export; .rknn is not supported"
                )
            official = "scales" in input_names
            # resolve inputs at build: unknown layouts must degrade here, not per chunk
            length_input = next((n for n in input_names if n in ("x_lengths", "x_length")), None)
            needed = {"x", "scales"} if official else {"x", "noise_scale", "length_scale"}
            if length_input is None or not needed <= set(input_names):
                raise ValueError(f"unrecognized matcha acoustic inputs: {input_names}")
            meta = acoustic.metadata()

            frontend: EspeakFrontend | LexiconFrontend
            interleave = True
            languages: tuple[str, ...] | None = None
            if official:
                # the official table is code; a supplied file cannot match the embedding
                if cfg.tokens_path:
                    logger.warning(
                        "voice: tts.matcha.tokensPath is ignored for official matcha "
                        "exports (their symbol table is fixed upstream)"
                    )
                token2id = fold_punct_aliases(official_token2id())
                frontend, language = _espeak_frontend(cfg, token2id)
                pad_id, bos_id, eos_id = 0, None, None
            else:
                if not meta:
                    raise ValueError(
                        "unrecognized matcha export: no 'scales' input and no metadata "
                        "(need the official or the icefall/sherpa-onnx .onnx)"
                    )
                if not cfg.tokens_path:
                    raise ValueError("icefall/sherpa matcha exports need tts.matcha.tokensPath")
                token2id = fold_punct_aliases(read_tokens(cfg.tokens_path))
                if meta.get("voice") == "zh en-us":
                    # dengcunqin bilingual (sherpa is_zh_en): zh via lexicon, English
                    # via espeak IPA folded to the trained alphabet, no interleave.
                    # espeak is mandatory here — English is half the model.
                    if not cfg.lexicon_path:
                        raise ValueError("this matcha model needs tts.matcha.lexiconPath")
                    if cfg.espeak_voice:
                        logger.warning(
                            "voice: tts.matcha.espeakVoice is ignored for the zh-en "
                            "model (its English is trained on en-us phonemes)"
                        )
                    data_dir = _espeak_data_dir(cfg)
                    if data_dir is None:
                        logger.warning(
                            "voice: no espeak-ng-data beside this zh-en matcha model, so "
                            "its English phonemes come from the installed espeak-ng and "
                            "may not match training (set tts.matcha.espeakDataDir)"
                        )
                    frontend = LexiconFrontend(
                        _load_lexicons(cfg.lexicon_path, cfg.lexicon_overrides_path, token2id), token2id,
                        english=EnglishToIpa(
                            token2id,
                            make_ipa_phonemizer(
                                "en-us", espeak_path=cfg.espeak_path, data_dir=data_dir
                            ),
                        ),
                        latin_space_id=token2id.get(" "),
                    )
                    interleave = False
                    languages = ("zh", "en")  # zh-primary (numbers, prologue), truly both
                elif int(meta.get("has_espeak", "0")):
                    frontend, _ = _espeak_frontend(cfg, token2id, meta.get("voice", "en-us"))
                elif int(meta.get("jieba", "0")):
                    if not cfg.lexicon_path:
                        raise ValueError("this matcha model needs tts.matcha.lexiconPath")
                    frontend = LexiconFrontend(
                        _load_lexicons(cfg.lexicon_path, cfg.lexicon_overrides_path, token2id), token2id,
                        english=_english_fallback(cfg, token2id),
                    )
                else:
                    raise ValueError(
                        "unsupported matcha export: metadata has neither has_espeak nor jieba"
                    )
                pad_id = int(meta.get("pad_id", "0"))
                use_eos_bos = bool(int(meta.get("use_eos_bos", "0")))
                bos_id = token2id.get("^") if use_eos_bos else None
                eos_id = token2id.get("$") if use_eos_bos else None
                language = _LANG_NAMES.get(meta.get("language", "").lower())
                if language is None and isinstance(frontend, LexiconFrontend):
                    language = "zh"  # fail closed: lexicon exports are all Chinese

            sample_rate = int(meta.get("sample_rate", str(_SAMPLE_RATE_DEFAULT)))
            out_names = acoustic.output_names()
            wav_direct = bool(out_names) and out_names[0] in ("wav", "audio_output")

            vocoder = None
            if not wav_direct and vocoder_share is not None:
                vocoder = vocoder_share
            elif not wav_direct:
                if not cfg.vocoder_path:
                    raise ValueError(
                        "this matcha export outputs mel; set tts.matcha.vocoderPath "
                        "(or re-export with the vocoder embedded)"
                    )
                model = models.enter_context(
                    OnDeviceModel(cfg.vocoder_path, **model_kw)  # type: ignore[arg-type]
                )
                vmeta = model.metadata()
                voc_rate = int(vmeta.get("sample_rate", str(sample_rate)))
                if voc_rate != sample_rate:
                    raise ValueError(
                        f"vocoder sample rate {voc_rate} != acoustic model {sample_rate}"
                    )
                stft_params = None
                if len(model.output_names()) == 3:  # Vocos: mag/cos/sin STFT frames
                    if int(vmeta.get("normalized", "0")):
                        raise ValueError("normalized-STFT vocos exports are not supported")
                    stft_params = {
                        "n_fft": int(vmeta.get("n_fft", "1024")),
                        "hop_length": int(vmeta.get("hop_length", "256")),
                        "center": bool(int(vmeta.get("center", "1"))),
                    }
                specs = model.input_specs()
                vocoder = VocoderSpec(model, specs[0][0] if specs else "mels", stft_params)

            adapter = cls(
                acoustic=acoustic,
                vocoder=vocoder,
                frontend=frontend,
                official=official,
                length_input=length_input,
                interleave=interleave,
                sample_rate=sample_rate,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
                spk_input=next((n for n in input_names if n in ("spks", "sid")), None),
                speaker_id=cfg.speaker_id,
                noise_scale=cfg.noise_scale,
                speed=cfg.speed,
                max_len=cfg.max_len
                or (120 if isinstance(frontend, LexiconFrontend) else 300),
                language=language,
                denoise_bias=(
                    denoiser_bias(vocoder, cfg.denoiser_strength) if vocoder else None
                ),
            )
            if languages:
                adapter.spoken_languages = languages
            models.pop_all()  # success: the adapter owns the models now
            return adapter

    def release(self) -> None:
        self._acoustic.release()
        if self._vocoder is not None:
            self._vocoder.model.release()

    def _synthesize_piece(self, text: str) -> np.ndarray:
        ids = frame_ids(
            self._frontend, text,
            bos_id=self._bos_id, eos_id=self._eos_id, pad_id=self._pad_id,
            interleave=self._interleave,
        )
        if not ids:
            return np.zeros(0, dtype=np.float32)

        x = np.array([ids], dtype=np.int64)
        length = np.array([x.shape[1]], dtype=np.int64)
        inputs = [("x", x), (self._length_input, length)]
        if self._official:
            inputs.append(
                ("scales", np.array([self._noise_scale, 1.0 / self._speed], dtype=np.float32))
            )
        else:
            inputs.append(("noise_scale", np.array([self._noise_scale], dtype=np.float32)))
            inputs.append(("length_scale", np.array([1.0 / self._speed], dtype=np.float32)))
        if self._spk_input:
            inputs.append((self._spk_input, np.array([self._speaker_id], dtype=np.int64)))
        outs = self._acoustic.run(inputs)

        if self._vocoder is None:  # embedded vocoder: the graph already emitted wav
            return self._edge_fade(np.asarray(outs[0], dtype=np.float32).reshape(-1))
        mel = np.asarray(outs[0], dtype=np.float32)
        vouts = self._vocoder.model.run([(self._vocoder.input_name, mel)])
        if self._vocoder.stft is not None:
            return self._edge_fade(istft(
                np.asarray(vouts[0])[0], np.asarray(vouts[1])[0],
                np.asarray(vouts[2])[0], **self._vocoder.stft,
            ))
        wav = np.asarray(vouts[0], dtype=np.float32).reshape(-1)
        if self._denoise_bias is not None:
            wav = denoise(wav, self._denoise_bias)
        return self._edge_fade(wav)


class SplitMatchaTtsAdapter(_MatchaCommon, OnDeviceTtsAdapter):
    """Static three-graph icefall Matcha: the host bridges durations -> alignment ->
    tiling -> denorm -> vocoding between fixed buckets (RKNN cannot express them).
    The decoder needs a PERIODIC tail - its attention has no time mask. Extension
    picks the runtime, so an .onnx triple validates the split off-board. The vocoder
    is Vocos (3 outputs, host ISTFT) or a waveform HiFi-GAN (1 output, denoised),
    classified by a build-time probe."""

    output_rate = _SAMPLE_RATE_DEFAULT
    _label = "Matcha-split"

    def __init__(
        self,
        *,
        encoder: OnDeviceModel,
        decoder: OnDeviceModel,
        vocoder: OnDeviceModel,
        frontend: EspeakFrontend | LexiconFrontend,
        pad_id: int,
        bos_id: int | None,
        eos_id: int | None,
        language: str,
        encoder_len: int,
        mel_len: int,
        mel_scale: float,
        mel_bias: float,
        stft: dict | None,               # None => single-output waveform vocoder
        voc_factor: int,                 # waveform mode: samples per mel frame
        denoise_bias: np.ndarray | None,
        noise_scale: float,
        speed: float,
        max_len: int,
    ):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._vocoder = vocoder
        self._frontend = frontend
        self._pad_id = pad_id
        self._bos_id = bos_id
        self._eos_id = eos_id
        self._encoder_len = encoder_len
        self._mel_len = mel_len
        self._mel_scale = mel_scale
        self._mel_bias = mel_bias
        self._stft = stft
        self._voc_factor = voc_factor
        self._denoise_bias = denoise_bias
        self._noise_scale = noise_scale
        self._speed = speed
        self._max_len = max_len
        self._rng = np.random.default_rng()
        self._mask = np.ones((1, 1, mel_len), dtype=np.float32)  # decoder sees a full bucket
        self.spoken_language = language
        self._log = logger.bind(component="tts-matcha-split")

    @classmethod
    def from_config(cls, cfg: MatchaTtsConfig) -> SplitMatchaTtsAdapter:
        missing = [
            name for name, value in (
                ("encoderPath", cfg.encoder_path),
                ("decoderPath", cfg.decoder_path),
                ("vocoderPath", cfg.vocoder_path),
                ("tokensPath", cfg.tokens_path),
            ) if not value
        ]
        if missing:
            raise ValueError(
                "matcha static split needs tts.matcha." + ", tts.matcha.".join(missing)
            )
        # explicit config > meta.json (named or beside the encoder) > ljspeech defaults;
        # a wrong mel pair is never an error, just audibly wrong audio
        meta_path = cfg.meta_path or (
            p if (p := Path(cfg.encoder_path).with_name("meta.json")).is_file() else None  # type: ignore[arg-type]
        )
        side: dict = {"encoder_len": 200, "mel_len": 800,
                      "mel_scale": 2.0661438, "mel_bias": -5.5238085}
        if meta_path:
            side.update(json.loads(Path(meta_path).read_text(encoding="utf-8")))
        encoder_len = cfg.encoder_len or int(side["encoder_len"])
        mel_len = cfg.mel_len or int(side["mel_len"])
        mel_scale = cfg.mel_scale if cfg.mel_scale is not None else float(side["mel_scale"])
        mel_bias = cfg.mel_bias if cfg.mel_bias is not None else float(side["mel_bias"])
        if mel_len % 4:
            raise ValueError(
                "tts.matcha.melLen must be a multiple of 4 (the decoder U-Net downsamples twice)"
            )

        token2id = fold_punct_aliases(read_tokens(cfg.tokens_path))  # type: ignore[arg-type]
        if is_zh_en_tokens(token2id):
            # Refusing beats synthesizing: this path frames ids as every other dialect
            # wants them, which for zh-en artifacts is fluent rhythm over wrong sounds.
            raise ValueError(
                "the bilingual zh-en matcha dialect needs the dynamic export "
                "(tts.matcha.acousticModelPath): the static split cannot express its "
                "no-blank-interleave framing or its espeak English"
            )
        # no metadata here: side files pick the frontend; lexicon exports are all zh
        if cfg.lexicon_path:
            frontend: EspeakFrontend | LexiconFrontend = LexiconFrontend(
                _load_lexicons(cfg.lexicon_path, cfg.lexicon_overrides_path, token2id), token2id,
                english=_english_fallback(cfg, token2id),
            )
            language = "zh"
        else:
            frontend, language = _espeak_frontend(cfg, token2id)
        pad_id = token2id.get("_", 0)
        bos_id = token2id.get("^")
        eos_id = token2id.get("$")
        if (bos_id is None) != (eos_id is None):
            raise ValueError("matcha split tokensPath must contain both '^' and '$', or neither")
        model_kw = dict(
            core_mask=cfg.core_mask, target=cfg.target, device_id=cfg.device_id,
            providers=cfg.execution_providers, provider_options=cfg.provider_options,
        )
        with ExitStack() as models:
            encoder, decoder, vocoder = (
                models.enter_context(OnDeviceModel(path, **model_kw))  # type: ignore[arg-type]
                for path in (cfg.encoder_path, cfg.decoder_path, cfg.vocoder_path)
            )
            # ONNX exposes static shapes (.rknn does not): catch geometry mismatch at build
            for model, name, want in (
                (encoder, "x", (1, encoder_len)),
                (decoder, "mu_up", (1, 80, mel_len)),
                (vocoder, "mels", (1, 80, mel_len)),
            ):
                shape = model.input_shape(name)
                if shape is not None and tuple(shape) != want:
                    raise ValueError(
                        f"matcha split: graph input {name} is {list(shape)}, "
                        f"configured geometry wants {list(want)}"
                    )
            # One zero-mel probe classifies the vocoder without graph introspection
            # (RKNN has none): 1 output = waveform (HiFi-GAN), 3 = Vocos. It also
            # yields the upsample factor and the denoiser bias, and a graph that
            # cannot run its own bucket fails HERE, not mid-conversation.
            probe = vocoder.run([("mels", np.zeros((1, 80, mel_len), np.float32))])
            stft_params, voc_factor, bias = None, 0, None
            if len(probe) == 1:
                wav = np.asarray(probe[0], dtype=np.float32).reshape(-1)
                voc_factor, rem = divmod(wav.size, mel_len)
                if voc_factor < 1 or rem:
                    raise ValueError(
                        f"matcha split: waveform vocoder emits {wav.size} samples for "
                        f"a {mel_len}-frame mel (not an integer upsample factor)"
                    )
                bias = _bias_from_wav(wav, cfg.denoiser_strength)
            else:
                vmeta = vocoder.metadata()
                if int(vmeta.get("normalized", "0")):
                    raise ValueError("normalized-STFT vocos exports are not supported")
                stft_params = {
                    "n_fft": int(vmeta.get("n_fft", "1024")),
                    "hop_length": int(vmeta.get("hop_length", "256")),
                    "center": bool(int(vmeta.get("center", "1"))),
                }
            adapter = cls(
                encoder=encoder,
                decoder=decoder,
                vocoder=vocoder,
                frontend=frontend,
                pad_id=pad_id,
                bos_id=bos_id,
                eos_id=eos_id,
                language=language,
                encoder_len=encoder_len,
                mel_len=mel_len,
                mel_scale=mel_scale,
                mel_bias=mel_bias,
                stft=stft_params,
                voc_factor=voc_factor,
                denoise_bias=bias,
                noise_scale=cfg.noise_scale,
                speed=cfg.speed,
                max_len=cfg.max_len or (40 if language == "zh" else 80),
            )
            models.pop_all()
            return adapter

    def release(self) -> None:
        for model in (self._encoder, self._decoder, self._vocoder):
            model.release()

    def _ids(self, text: str) -> list[int]:
        return frame_ids(
            self._frontend, text,
            bos_id=self._bos_id, eos_id=self._eos_id, pad_id=self._pad_id,
        )

    def _synthesize_piece(self, text: str) -> np.ndarray:
        ids = self._ids(text)
        if not ids:
            return np.zeros(0, dtype=np.float32)
        if len(ids) > self._encoder_len:
            # IPA expansion can outrun the char budget; halve, never crop mid-word
            return self._overflow_retry(
                text, f"needs {len(ids)} tokens > encoderLen {self._encoder_len}"
            )
        x = np.full((1, self._encoder_len), self._pad_id, dtype=np.int64)
        x[0, :len(ids)] = ids
        x_length = np.array([len(ids)], dtype=np.int64)
        mu, logw = self._encoder.run([("x", x), ("x_length", x_length)])
        mu_up, total = self._length_regulator(mu, logw, len(ids))
        if total <= 0:
            return np.zeros(0, dtype=np.float32)
        if total > self._mel_len:
            return self._overflow_retry(
                text, f"predicts {total} mel frames > melLen {self._mel_len}"
            )
        # fresh noise at mu_up's length: mu and z tile with ONE period
        z = (
            self._rng.standard_normal((1, 80, mu_up.shape[2])).astype(np.float32)
            * self._noise_scale
        )
        mu_in = self._tile_to_bucket(mu_up, self._mel_len)
        z_in = self._tile_to_bucket(z, self._mel_len)
        mel_raw = self._decoder.run(
            [("mu_up", mu_in), ("mask", self._mask), ("z", z_in)]
        )[0]
        mel = (
            np.asarray(mel_raw, dtype=np.float32)[:, :, :total] * self._mel_scale
            + self._mel_bias
        )
        # edge-replicate: zero is a loud, valid log-mel value
        mel_in = self._edge_to_bucket(mel, self._mel_len)
        if self._stft is None:  # waveform vocoder: bucket wav, keep the real frames
            wav = np.asarray(self._vocoder.run([("mels", mel_in)])[0], dtype=np.float32)
            wav = wav.reshape(-1)[: total * self._voc_factor]
            if self._denoise_bias is not None:
                wav = denoise(wav, self._denoise_bias)
            return self._edge_fade(wav)
        mag, cos, sin = self._vocoder.run([("mels", mel_in)])
        n_fft, hop = self._stft["n_fft"], self._stft["hop_length"]
        # frames past total + n_fft/hop cannot touch a kept sample
        keep = min(self._mel_len, total + max(1, n_fft // hop))
        wav = istft(
            np.asarray(mag, dtype=np.float32)[0, :, :keep],
            np.asarray(cos, dtype=np.float32)[0, :, :keep],
            np.asarray(sin, dtype=np.float32)[0, :, :keep],
            **self._stft,
        )
        return self._edge_fade(wav[: total * hop])

    def _overflow_retry(self, text: str, reason: str) -> np.ndarray:
        if len(text.strip()) <= 1:
            self._log.warning("Matcha-split: unsplittable piece {}", reason)
            return np.zeros(0, dtype=np.float32)
        return self._halve_and_retry(text)

    def _length_regulator(self, mu: np.ndarray, logw: np.ndarray, token_len: int) -> tuple[np.ndarray, int]:
        mu = np.asarray(mu, dtype=np.float32)
        logw = np.asarray(logw, dtype=np.float32)
        # ceil then scale, matching the dynamic graph
        durations = np.ceil(np.exp(logw[0, 0, :token_len])) * (1.0 / self._speed)
        total = int(durations.sum())
        work_len = int(4 * np.ceil(total / 4))
        if work_len <= 0:
            return np.zeros((1, 80, 0), dtype=np.float32), 0
        # the alignment is one-hot per frame: gather, not a dense matmul
        ends = np.cumsum(durations)
        idx = np.searchsorted(ends, np.arange(total), side="right")
        mu_up = np.zeros((1, mu.shape[1], work_len), dtype=np.float32)
        mu_up[:, :, :total] = mu[:, :, np.minimum(idx, token_len - 1)]
        return mu_up, total

    @staticmethod
    def _tile_to_bucket(arr: np.ndarray, bucket: int) -> np.ndarray:
        if arr.shape[2] <= 0:
            raise ValueError("cannot pad an empty Matcha sequence")
        reps = int(np.ceil(bucket / arr.shape[2]))
        return np.tile(arr, (1, 1, reps))[:, :, :bucket].astype(np.float32, copy=False)

    @staticmethod
    def _edge_to_bucket(arr: np.ndarray, bucket: int) -> np.ndarray:
        if arr.shape[2] > bucket:
            raise ValueError(f"mel length {arr.shape[2]} exceeds bucket {bucket}")
        if arr.shape[2] == bucket:
            return arr.astype(np.float32, copy=False)
        return np.concatenate(
            [arr, np.repeat(arr[:, :, -1:], bucket - arr.shape[2], axis=2)], axis=2
        ).astype(np.float32, copy=False)
