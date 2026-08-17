"""Pure-Python Whisper tokenizer helpers: language tokens, detection, detokenization.

Carries **no numpy**, unlike :mod:`.whisper`. The language is the special token at
decoder-prompt position 1; the export is the multilingual ``base`` checkpoint (not
``base.en``), confirmed by the 51865-entry vocabulary (English-only is 51864) carrying
all 99 ``<|xx|>`` tokens at 50259..50357, so adding a language needs NO re-export.

**Language ID is free.** Our decoder is causal with no KV cache and pins SOT at window
position 0, so ``logits[0][0]`` is already the distribution Whisper's
``detect_language`` argmaxes on every decode step; :func:`detect_language` restricts
that argmax to the caller's candidates, which is what makes it accurate on a small
model.

**Detokenization is the byte-level BPE inverse mapping** and nothing else: concatenate
the per-id strings over the GPT-2 reversible byte<->unicode alphabet (space -> ``Ġ``),
map each char back to its byte, UTF-8-decode once at the end, so a CJK char split
across merges reassembles by construction. :func:`read_vocab` never re-encodes; it
normalizes ORIGINAL artifacts to that alphabet, rejecting the Rockchip demo's
per-group-base64 ``vocab_zh.txt``, which modified the encoding instead of inverting it.
"""

from __future__ import annotations

import base64
import json
import math
import unicodedata
from collections.abc import Sequence

# Whisper multilingual language order (base/tiny/medium family, n_vocab 51865); the
# token id is ``<|startoftranscript|>(50258) + 1 + index``: en=50259, ja=50266.
LANGUAGES: tuple[str, ...] = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
    "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
    "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
    "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
    "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
    "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
)

_LANG_TOKEN_BASE = 50259  # == <|startoftranscript|> + 1 == <|en|>
# First special token. Everything at or above it (<|endoftext|>, <|startoftranscript|>,
# the 99 languages, <|transcribe|>, <|notimestamps|>, the 1501 timestamps) is structural
# and must never be suppressed: blocking EOT alone would stop decoding terminating.
_EOT = 50257


def language_token(code: str) -> int:
    """The Whisper language token id for an ISO code (``en``->50259, ``ja``->50266)."""
    try:
        return _LANG_TOKEN_BASE + LANGUAGES.index(code)
    except ValueError:
        raise ValueError(
            f"unknown Whisper language '{code}'; expected one of {', '.join(LANGUAGES)}"
        ) from None


def language_tokens(codes: Sequence[str]) -> dict[int, str]:
    """``{token_id: code}`` for the enabled languages (order-preserving, deduped);
    an unknown code raises so a typo in ``stt.languages`` fails at construction (a
    clean registry fallback) instead of silently narrowing detection."""
    return {language_token(c): c for c in dict.fromkeys(c.lower() for c in codes)}


def detect_language(
    logits: Sequence[float], candidates: dict[int, str], *, min_confidence: float = 0.0
) -> int | None:
    """Whisper language ID: the highest-scoring language token among ``candidates``.

    ``logits`` is the decoder row at window position 0 (the one holding SOT).
    ``min_confidence`` (0..1) is a softmax over the CANDIDATE SET only, meaningful
    with 2-4 candidates where one over all 51865 logits would not be; below it,
    return ``None`` and the caller keeps its configured default.
    """
    if not candidates:
        return None
    top_id, top_score = None, 0.0
    scores: list[float] = []
    for token in candidates:
        score = float(logits[token])
        scores.append(score)
        if top_id is None or score > top_score:
            top_id, top_score = token, score
    if min_confidence > 0.0:
        # softmax(top) == 1 / sum(exp(s - top)); shift-invariant, so no overflow.
        total = sum(math.exp(s - top_score) for s in scores)
        if total <= 0.0 or 1.0 / total < min_confidence:
            return None
    return top_id


# ---- decodable-language vocabulary mask -------------------------------------
# The DECODABLE set (``stt.languages``) is a guarantee on what may be EMITTED,
# enforced by suppressing tokens whose script no enabled language uses. Latin is
# allowed unconditionally: it carries the digits, names and loanwords that appear in
# every language's real transcripts, and is ~80% of the vocabulary.
_ALWAYS_ALLOWED_SCRIPTS = frozenset({"latin"})

# Scripts per Whisper language, non-Latin ones only; every other code defaults to
# Latin. Languages written in two scripts list both.
_LANGUAGE_SCRIPTS: dict[str, frozenset[str]] = {
    "zh": frozenset({"han"}),
    "ja": frozenset({"han", "hiragana", "katakana"}),
    "ko": frozenset({"hangul", "han"}),
    "ru": frozenset({"cyrillic"}), "uk": frozenset({"cyrillic"}),
    "be": frozenset({"cyrillic"}), "bg": frozenset({"cyrillic"}),
    "mk": frozenset({"cyrillic"}), "kk": frozenset({"cyrillic"}),
    "mn": frozenset({"cyrillic"}), "tg": frozenset({"cyrillic"}),
    "tt": frozenset({"cyrillic"}), "ba": frozenset({"cyrillic"}),
    "sr": frozenset({"cyrillic"}), "uz": frozenset({"cyrillic"}),
    "az": frozenset({"cyrillic"}),
    "ar": frozenset({"arabic"}), "fa": frozenset({"arabic"}),
    "ur": frozenset({"arabic"}), "ps": frozenset({"arabic"}),
    "sd": frozenset({"arabic"}),
    "he": frozenset({"hebrew"}), "yi": frozenset({"hebrew"}),
    "hi": frozenset({"devanagari"}), "mr": frozenset({"devanagari"}),
    "ne": frozenset({"devanagari"}), "sa": frozenset({"devanagari"}),
    "bn": frozenset({"bengali"}), "as": frozenset({"bengali"}),
    "pa": frozenset({"gurmukhi"}), "gu": frozenset({"gujarati"}),
    "ta": frozenset({"tamil"}), "te": frozenset({"telugu"}),
    "kn": frozenset({"kannada"}), "ml": frozenset({"malayalam"}),
    "si": frozenset({"sinhala"}), "th": frozenset({"thai"}),
    "lo": frozenset({"lao"}), "bo": frozenset({"tibetan"}),
    "my": frozenset({"myanmar"}), "km": frozenset({"khmer"}),
    "el": frozenset({"greek"}), "hy": frozenset({"armenian"}),
    "ka": frozenset({"georgian"}), "am": frozenset({"ethiopic"}),
}

# Contiguous Unicode ranges -> script bucket, ordered by first code point; only the
# scripts Whisper's vocabulary contains. Allowed to be incomplete: an unmapped letter
# is script-neutral, so a gap costs a missed suppression, never a blocked token.
_SCRIPT_RANGES: tuple[tuple[int, int, str], ...] = (
    (0x0000, 0x007F, "latin"), (0x0080, 0x024F, "latin"),
    (0x0370, 0x03FF, "greek"), (0x0400, 0x052F, "cyrillic"),
    (0x0530, 0x058F, "armenian"), (0x0590, 0x05FF, "hebrew"),
    (0x0600, 0x06FF, "arabic"), (0x0750, 0x077F, "arabic"),
    (0x0900, 0x097F, "devanagari"), (0x0980, 0x09FF, "bengali"),
    (0x0A00, 0x0A7F, "gurmukhi"), (0x0A80, 0x0AFF, "gujarati"),
    (0x0B80, 0x0BFF, "tamil"), (0x0C00, 0x0C7F, "telugu"),
    (0x0C80, 0x0CFF, "kannada"), (0x0D00, 0x0D7F, "malayalam"),
    (0x0D80, 0x0DFF, "sinhala"), (0x0E00, 0x0E7F, "thai"),
    (0x0E80, 0x0EFF, "lao"), (0x0F00, 0x0FFF, "tibetan"),
    (0x1000, 0x109F, "myanmar"), (0x10A0, 0x10FF, "georgian"),
    (0x1100, 0x11FF, "hangul"), (0x1200, 0x137F, "ethiopic"),
    (0x1780, 0x17FF, "khmer"), (0x1E00, 0x1EFF, "latin"),
    (0x1F00, 0x1FFF, "greek"), (0x3040, 0x309F, "hiragana"),
    (0x30A0, 0x30FF, "katakana"), (0x3130, 0x318F, "hangul"),
    (0x31F0, 0x31FF, "katakana"), (0x3400, 0x4DBF, "han"),
    (0x4E00, 0x9FFF, "han"), (0xAC00, 0xD7AF, "hangul"),
    (0xF900, 0xFAFF, "han"), (0xFB50, 0xFDFF, "arabic"),
    # Fullwidth ASCII (ｆｕｌｌ) and halfwidth katakana (ﾃﾞ) are common in ja/zh web text.
    (0xFF01, 0xFF5E, "latin"), (0xFF61, 0xFF9F, "katakana"),
    (0xFFA0, 0xFFDC, "hangul"),
)


def _char_script(ch: str) -> str | None:
    """Script bucket for one char, or ``None`` when script-NEUTRAL: digits,
    whitespace, punctuation, symbols (incl. CJK punctuation) and combining marks occur
    in every language, and a letter outside :data:`_SCRIPT_RANGES` fails open: a
    table gap must not block the token for EVERY multi-language set."""
    if ch.isspace() or ch.isdigit():
        return None
    if unicodedata.category(ch)[0] in ("M", "P", "S", "C", "Z"):
        return None
    o = ord(ch)
    for lo, hi, script in _SCRIPT_RANGES:
        if lo <= o <= hi:
            return script
    return None


def token_scripts(text: str) -> frozenset[str]:
    """The scripts a decoded token draws on (empty = script-neutral). A token that
    does not decode to valid UTF-8 is a **partial byte-level fragment** (a multi-byte
    char split across merges) and returns empty: suppressing fragments would break
    the encoding of legitimate text."""
    if "�" in text:  # U+FFFD from byte_level_decode(errors="replace")
        return frozenset()
    return frozenset(s for s in (_char_script(c) for c in text) if s)


def suppressed_token_ids(vocab: dict[str, str], codes: Sequence[str]) -> tuple[int, ...]:
    """Token ids to block so output stays inside the DECODABLE language set.

    ``()`` when ``codes`` is empty (feature off): the caller keeps the plain
    full-vocabulary argmax and pays nothing. Never suppresses specials
    (``>= <|endoftext|>``, including EOT and the timestamps the decode loop relies on),
    script-neutral tokens, byte-level fragments or Latin; ``["en","zh","ja"]`` blocks a
    few thousand of the 51865, mostly Cyrillic/Hangul/Arabic/Greek/Hebrew/Thai."""
    if not codes:
        return ()
    allowed = set(_ALWAYS_ALLOWED_SCRIPTS)
    for code in codes:
        allowed |= _LANGUAGE_SCRIPTS.get(code.lower(), _ALWAYS_ALLOWED_SCRIPTS)

    blocked: list[int] = []
    for key, token in vocab.items():
        try:
            token_id = int(key)
        except ValueError:
            continue
        if token_id >= _EOT or not token:
            continue
        scripts = token_scripts(byte_level_decode(token))
        if scripts and not scripts <= allowed:
            blocked.append(token_id)
    return tuple(sorted(blocked))


def resolve_languages(
    language: str,
    languages: Sequence[str] | None,
) -> tuple[str, tuple[str, ...]]:
    """Resolve ``(preferred_language, decodable_languages)`` from config.

    ``language`` (**preferred**, exactly one) selects the decoder prompt token and is
    the fallback for a low-confidence detection. ``languages`` (**decodable**) bounds
    what may be OUTPUT and supplies the detection candidates: the prompt carries
    exactly ONE language token, so a list can never mean "decode all at once". A list
    of one is just a fixed language, so both effects switch off (zero cost). Preferring
    a non-decodable language is contradictory, so preferred is pulled back into the set
    (the caller warns when the preference was explicit) rather than widening the decodable
    guarantee.
    """
    preferred = (language or "en").lower()
    codes = tuple(dict.fromkeys(c.lower() for c in (languages or []) if c))
    if len(codes) == 1:
        preferred, codes = codes[0], ()
    elif codes and preferred not in codes:
        preferred = codes[0]
    return preferred, codes


# The 64 base64 symbols plus padding, used only to REJECT the legacy re-encoded vocab.
# A byte-level token can coincidentally be drawn from this alphabet ("abcd"), so the
# check also demands the 4-char grouping the re-encoding forced on every subword ("!"
# would have become "IQ==").
_B64_ALPHABET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)


def _reject_reencoded_vocab(vocab: dict[str, str], path: str) -> None:
    checked = 0
    for i in range(256):  # low ids are single bytes in every true encoding
        token = vocab.get(str(i))
        if not token:
            continue
        checked += 1
        if len(token) % 4 != 0 or not _B64_ALPHABET.issuperset(token):
            return
    if checked:
        raise ValueError(
            f"{path} re-encodes the vocabulary per-subword (the Rockchip demo's "
            "vocab_zh.txt scheme), which is not supported: Whisper detokenization is "
            "the byte-level BPE inverse mapping over the ORIGINAL vocabulary. Use "
            "multilingual.tiktoken (openai/whisper assets), vocab.json (HF export), "
            "or a byte-level flat '<id> <token>' file."
        )


def read_vocab(vocab_path: str) -> dict[str, str]:
    """Load ``{id: byte_level_token}`` from an ORIGINAL tokenizer artifact, normalized
    to the byte-level alphabet: ``*.tiktoken`` (OpenAI's whisper assets,
    ``base64(token_bytes) rank`` per line), HF ``vocab.json`` (``{token_string: id}``,
    already byte-level), or flat byte-level ``<id> <token>`` text (the Rockchip demo's
    ``vocab_en.txt``: despite the name, the full multilingual vocabulary)."""
    if vocab_path.endswith(".tiktoken"):
        vocab: dict[str, str] = {}
        with open(vocab_path, encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 2:
                    continue
                token_bytes = base64.b64decode(parts[0])
                vocab[parts[1]] = "".join(_BYTE_ENCODER[b] for b in token_bytes)
        return vocab
    if vocab_path.endswith(".json"):
        with open(vocab_path, encoding="utf-8") as f:
            raw = json.load(f)
        return {str(i): token for token, i in raw.items()}
    vocab = {}
    with open(vocab_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            vocab[parts[0]] = parts[1] if len(parts) >= 2 else ""
    _reject_reencoded_vocab(vocab, vocab_path)
    return vocab


# ---- byte-level BPE (GPT-2 / tiktoken) --------------------------------------

def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 reversible byte<->unicode map: every one of 256 bytes -> a printable char."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


_BYTE_ENCODER = _bytes_to_unicode()
_BYTE_DECODER = {ch: b for b, ch in _BYTE_ENCODER.items()}


def byte_level_encode(text: str) -> str:
    """Encode text to its byte-level representation (inverse of the decoder; for tests)."""
    return "".join(_BYTE_ENCODER[b] for b in text.encode("utf-8"))


def byte_level_decode(s: str) -> str:
    """Byte-level string back to real text (``Ġthe`` -> `` the``); transcript callers
    concatenate all per-token strings first so multi-byte chars split across merges
    reassemble before the single UTF-8 decode, while ``errors="replace"`` lets script
    analysis decode a LONE token and detect fragments by the U+FFFD."""
    buf = bytes(_BYTE_DECODER[ch] for ch in s if ch in _BYTE_DECODER)
    return buf.decode("utf-8", errors="replace")
