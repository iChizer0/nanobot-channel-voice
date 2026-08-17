"""On-device Matcha-TTS (``tts.provider="matcha"``): the KTH flow-matching acoustic
model over ONNX, speaking BOTH export dialects, detected from the graph itself:

* the OFFICIAL export (``python -m matcha.onnx.export``, the preferred source):
  ``scales=[temperature, length_scale]`` input, no bos/eos, the fixed symbol table
  vendored below (upstream bakes it in code, there is no tokens.txt), and - when the
  vocoder was embedded at export - a direct ``wav`` output, no vocoder file at all;
* the icefall/sherpa-onnx export (``matcha-icefall-*``): separate noise/length inputs,
  front-end contract in the ONNX metadata (``has_espeak``/``jieba``/``use_eos_bos``),
  ``tokens.txt`` [+ ``lexicon.txt`` for zh], and a mel output for a vocoder file.

A mel-emitting model needs ``vocoderPath``: a Vocos graph (three outputs: STFT
magnitude/cos/sin; the inverse STFT runs host-side in numpy, mirroring sherpa's
knf::IStft) or any single-output waveform vocoder (HiFi-GAN). English phonemizes
through espeak-ng - the system binary or an explicit ``espeakPath``. Imported lazily
by :func:`make_tts`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from contextlib import ExitStack
from typing import NamedTuple

import numpy as np
from loguru import logger

from nanobot_channel_voice.config import MatchaTtsConfig
from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
from nanobot_channel_voice.tts.espeak import make_ipa_phonemizer
from nanobot_channel_voice.tts.ondevice_base import OnDeviceTtsAdapter
from nanobot_channel_voice.tts.text_frontend import verbalize_numbers_zh

_JOIN_GAP_S = 0.1
_SAMPLE_RATE_DEFAULT = 22050  # every published matcha vocoder (HiFi-GAN, Vocos univ)

# metadata "language" -> the ISO 639-1 code spoken_language wants. Only names seen on
# published matcha exports; an unknown name leaves the config's tts.language in charge.
_LANG_NAMES = {"english": "en", "chinese": "zh", "german": "de", "japanese": "ja"}


# CJK terminators split anywhere; ASCII ones only before whitespace/end, so "3.14",
# "3:30" and "e.g." survive intact (the chunker's _primary_cut applies the same rule).
_SENTENCE_SPLIT_RE = re.compile(r"(?:(?<=[。！？…])(?![。！？…])|(?<=[.!?])(?=\s|$))")
_CLAUSE_SPLIT_RE = re.compile(r"([、，；：]|[,;:](?=\s|$))")
_SENTENCE_PUNCT = ".!?…。！？"
_LANG_SWITCH_RE = re.compile(r"\([a-z0-9-]+\)")  # espeak "(zh)" language-switch flags

# The official symbol table (matcha/text/symbols.py, from keithito/tacotron): ids are
# list positions. Copied VERBATIM - character count and order define the embedding.
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
        # An empty table maps every synthesis to zero ids: a mute TTS that looks healthy.
        raise ValueError(f"no tokens parsed from {path} (wrong format?)")
    return token2id


# Half-width/full-width punctuation twins (sherpa's CharacterLexicon::InitTokens): each
# model's symbol table carries one spelling, agent text uses both.
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


def load_lexicon(path: str, token2id: dict[str, int]) -> dict[str, list[int]]:
    """``lexicon.txt``: ``<word> <phone>...`` per line, first spelling wins; a word
    with any unknown phone is dropped (sherpa behaviour)."""
    word2ids: dict[str, list[int]] = {}
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
                continue
    return word2ids


def add_blank(ids: list[int], pad_id: int) -> list[int]:
    """[pad, t0, pad, t1, ..., pad]: the training-time interleave (upstream
    ``intersperse``, sherpa ``AddBlank``)."""
    out = [pad_id] * (2 * len(ids) + 1)
    out[1::2] = ids
    return out


def istft(mag: np.ndarray, cos: np.ndarray, sin: np.ndarray, *,
          n_fft: int, hop_length: int, center: bool) -> np.ndarray:
    """Vocos (bins, frames) magnitude/cos/sin -> waveform: periodic-hann overlap-add
    with window-square normalization, torch.istft-compatible (= sherpa's knf::IStft)."""
    # Fill the complex halves in place: mag.astype(complex) * (cos + 1j*sin) would
    # allocate four full-size temporaries on a memory-bandwidth-bound path.
    spec = np.empty(mag.shape, dtype=np.complex64)
    np.multiply(mag, cos, out=spec.real)
    np.multiply(mag, sin, out=spec.imag)
    frames = np.fft.irfft(spec, n=n_fft, axis=0).astype(np.float32)  # (n_fft, F)
    win = (0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n_fft) / n_fft))).astype(np.float32)
    frames *= win[:, None]
    n_frames = frames.shape[1]
    total = n_fft + hop_length * (n_frames - 1)
    out = np.zeros(total, dtype=np.float32)
    wsum = np.zeros(total, dtype=np.float32)
    w2 = win * win
    if n_fft % hop_length == 0:
        # Overlap-add as n_fft/hop strided lane-adds (4 for Vocos) instead of a
        # per-frame Python loop: within one lane the frame targets never overlap.
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


def _is_latin(ch: str) -> bool:
    """Scripts the published espeak-fronted matcha voices (all Latin-written) can
    plausibly speak. Everything else - CJK, Cyrillic, Greek, Arabic, ... - would be
    voiced as garbage IPA after espeak switches language, so refusing it here lets
    the shell's speakability guard skip and WARN instead (mms/supertonic parity)."""
    cp = ord(ch)
    return cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF  # ASCII..IPA-block start; Latin ext additional


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
    """Text -> per-sentence token-id sequences via espeak-ng IPA output, one
    phonemizer call per clause so clause punctuation survives as its own token
    (espeak never emits punctuation in IPA output). Unknown IPA codepoints are
    skipped like the references; ties/affricate joiners fall out the same way."""

    def __init__(self, token2id: dict[str, int], *, phonemize: Callable[[str], str]):
        self._token2id = token2id
        self._phonemize = phonemize
        self._space_id = token2id.get(" ")

    def can_speak(self, ch: str) -> bool:
        # Phoneme coverage is decided post-G2P, so per-char only the wrong-script
        # failure mode is answerable: non-Latin never survives these voices.
        return _is_latin(ch)

    def _ipa_ids(self, ipa: str) -> list[int]:
        # Strip "(zh)"-style switch flags BEFORE the per-codepoint map: their letters
        # are valid phoneme symbols and would otherwise be voiced.
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
        """One phonemizer call for the whole batch: espeak emits one clause per line,
        so newline-joined input maps back by line - saving a subprocess spawn per
        clause on the hot path. A line-count mismatch (espeak re-split something)
        falls back to per-clause calls, whose multi-line output _ipa_ids tolerates."""
        if len(clauses) > 1:
            lines = self._phonemize("\n".join(clauses)).splitlines()
            if len(lines) == len(clauses):
                return lines
        return [self._phonemize(c) for c in clauses]

    def sentences(self, text: str) -> list[list[int]]:
        # Collect every clause first so the whole piece phonemizes in one call.
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


class LexiconFrontend:
    """Text -> per-sentence token-id sequences by greedy longest match against
    lexicon.txt (multi-char entries make this word- not char-level; sherpa's
    PhraseMatcher without jieba), single chars falling back to the token table
    (folded punctuation included). OOV is dropped; the shell's speakability guard
    reports it."""

    def __init__(self, word2ids: dict[str, list[int]], token2id: dict[str, int]):
        self._word2ids = word2ids
        self._token2id = token2id
        # Longest key PER FIRST CHAR: one long entry anywhere in a user lexicon must
        # not multiply the match scan for every position in every text.
        self._max_by_first: dict[str, int] = {}
        for word in word2ids:
            if len(word) > self._max_by_first.get(word[0], 0):
                self._max_by_first[word[0]] = len(word)

    def can_speak(self, ch: str) -> bool:
        ch = ch.lower()
        return ch in self._word2ids or ch in self._token2id

    def _tokenize(self, text: str) -> list[int]:
        text = text.lower()
        ids: list[int] = []
        i = 0
        while i < len(text):
            longest = self._max_by_first.get(text[i], 0)
            for ln in range(min(longest, len(text) - i), 0, -1):
                cand = text[i : i + ln]
                if cand in self._word2ids:
                    ids.extend(self._word2ids[cand])
                    i += ln
                    break
            else:
                ch = text[i]
                if ch in self._token2id and not ch.isspace():
                    ids.append(self._token2id[ch])
                i += 1
        return ids

    def sentences(self, text: str) -> list[list[int]]:
        return [ids for body, punct in _sentences(text) if (ids := self._tokenize(body + punct))]


class VocoderSpec(NamedTuple):
    """A mel-consuming vocoder graph; ``stft`` is set for Vocos (3-output STFT
    frames needing the host ISTFT), None for direct-waveform vocoders (HiFi-GAN)."""

    model: OnDeviceModel
    input_name: str
    stft: dict | None


class MatchaTtsAdapter(OnDeviceTtsAdapter):
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
    ):
        super().__init__()
        self._acoustic = acoustic
        self._vocoder = vocoder
        self._frontend = frontend
        self._official = official
        self._length_input = length_input
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
    def from_config(cls, cfg: MatchaTtsConfig) -> MatchaTtsAdapter:
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
            # Resolve every graph input at BUILD time: an unrecognized layout must
            # fall back to system TTS here, not fail mutely per chunk in the shell.
            length_input = next((n for n in input_names if n in ("x_lengths", "x_length")), None)
            needed = {"x", "scales"} if official else {"x", "noise_scale", "length_scale"}
            if length_input is None or not needed <= set(input_names):
                raise ValueError(f"unrecognized matcha acoustic inputs: {input_names}")
            meta = acoustic.metadata()

            frontend: EspeakFrontend | LexiconFrontend
            if official:
                # No tokens.txt in the official world: the symbol table is code, and a
                # supplied file cannot match the baked-in embedding.
                if cfg.tokens_path:
                    logger.warning(
                        "voice: tts.matcha.tokensPath is ignored for official matcha "
                        "exports (their symbol table is fixed upstream)"
                    )
                token2id = fold_punct_aliases(official_token2id())
                voice = cfg.espeak_voice or "en-us"
                phonemize = make_ipa_phonemizer(voice, espeak_path=cfg.espeak_path)
                frontend = EspeakFrontend(token2id, phonemize=phonemize)
                pad_id, bos_id, eos_id = 0, None, None
                # The voice IS the language: "de"/"de-x-..." speaks German even
                # through an en-trained model, and the agent must be told which.
                prefix = voice.partition("-")[0].lower()
                language = prefix if len(prefix) == 2 else _LANG_NAMES.get(prefix, "en")
            else:
                if not meta:
                    raise ValueError(
                        "unrecognized matcha export: no 'scales' input and no metadata "
                        "(need the official or the icefall/sherpa-onnx .onnx)"
                    )
                if not cfg.tokens_path:
                    raise ValueError("icefall/sherpa matcha exports need tts.matcha.tokensPath")
                token2id = fold_punct_aliases(read_tokens(cfg.tokens_path))
                if int(meta.get("has_espeak", "0")):
                    voice = cfg.espeak_voice or meta.get("voice", "en-us")
                    frontend = EspeakFrontend(
                        token2id,
                        phonemize=make_ipa_phonemizer(voice, espeak_path=cfg.espeak_path),
                    )
                elif int(meta.get("jieba", "0")):
                    if not cfg.lexicon_path:
                        raise ValueError("this matcha model needs tts.matcha.lexiconPath")
                    frontend = LexiconFrontend(
                        load_lexicon(cfg.lexicon_path, token2id), token2id
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
                    # Every published lexicon/jieba matcha export is Chinese; failing
                    # open on a re-export's metadata spelling would silently skip the
                    # digit verbalizer AND the agent's reply-in-zh context.
                    language = "zh"

            sample_rate = int(meta.get("sample_rate", str(_SAMPLE_RATE_DEFAULT)))
            out_names = acoustic.output_names()
            wav_direct = bool(out_names) and out_names[0] in ("wav", "audio_output")

            vocoder = None
            if not wav_direct:
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
                stft = None
                if len(model.output_names()) == 3:  # Vocos: mag/cos/sin STFT frames
                    if int(vmeta.get("normalized", "0")):
                        raise ValueError("normalized-STFT vocos exports are not supported")
                    stft = {
                        "n_fft": int(vmeta.get("n_fft", "1024")),
                        "hop_length": int(vmeta.get("hop_length", "256")),
                        "center": bool(int(vmeta.get("center", "1"))),
                    }
                specs = model.input_specs()
                vocoder = VocoderSpec(model, specs[0][0] if specs else "mels", stft)

            adapter = cls(
                acoustic=acoustic,
                vocoder=vocoder,
                frontend=frontend,
                official=official,
                length_input=length_input,
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
            )
            models.pop_all()  # success: the adapter owns the models now
            return adapter

    def release(self) -> None:
        self._acoustic.release()
        if self._vocoder is not None:
            self._vocoder.model.release()

    def _piece_budget(self) -> int:
        return self._max_len

    def _normalize(self, text: str) -> str:
        # espeak verbalizes digits itself; the zh lexicon has 零..九 but no "0".."9".
        return verbalize_numbers_zh(text) if self.spoken_language == "zh" else text

    def _can_speak(self, ch: str) -> bool:
        return self._frontend.can_speak(ch)

    def _synthesize_piece(self, text: str) -> np.ndarray:
        ids: list[int] = []
        for seq in self._frontend.sentences(text):
            if self._bos_id is not None:
                seq = [self._bos_id, *seq]
            if self._eos_id is not None:
                seq = [*seq, self._eos_id]
            ids.extend(add_blank(seq, self._pad_id))
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
            return np.asarray(outs[0], dtype=np.float32).reshape(-1)
        mel = np.asarray(outs[0], dtype=np.float32)
        vouts = self._vocoder.model.run([(self._vocoder.input_name, mel)])
        if self._vocoder.stft is not None:
            return istft(
                np.asarray(vouts[0])[0], np.asarray(vouts[1])[0],
                np.asarray(vouts[2])[0], **self._vocoder.stft,
            )
        return np.asarray(vouts[0], dtype=np.float32).reshape(-1)
