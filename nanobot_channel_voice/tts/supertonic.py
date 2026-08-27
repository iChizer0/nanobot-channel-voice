"""On-device Supertonic-3 TTS (flow matching) over RKNN/ONNX.

Four graphs from the ORIGINAL release (huggingface.co/Supertone/supertonic-3, MIT):
duration_predictor -> text_encoder -> vector_estimator (Euler flow loop) -> vocoder.
One 44.1 kHz model covers 31 languages (NO zh), selected by wrapping the text in
``<lang>...</lang>`` tags indexed like any other characters. The front-end is a
per-codepoint lookup into ``unicode_indexer.json`` (65536 entries) after NFKD
decomposition — only decomposed characters are known (precomposed ă/č/한 map to -1).
Math mirrors the reference (supertone-inc/supertonic py/helper.py, sherpa-onnx's C++
port); side files keep the original JSON formats so the fp32 graphs double as the
int8/RKNN conversion source. Imported lazily, so the plugin imports without numpy.
"""

from __future__ import annotations

import json
import math
import unicodedata
from contextlib import ExitStack

import numpy as np
from loguru import logger

from nanobot_channel_voice.config import SupertonicTtsConfig
from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
from nanobot_channel_voice.tts.ondevice_base import OnDeviceTtsAdapter

_JOIN_GAP_S = 0.3       # silence between budget-split pieces (reference default)
_MIN_DURATION_S = 0.1   # floor after speed scaling (reference kMinDuration)
_MAX_LATENT_LEN = 10000  # OOM guard (reference kMaxLatentLen)

LANGUAGES = frozenset(
    "en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl "
    "pt ro ru sk sl sv tr uk vi".split()
)

# PreprocessText replacement table, applied in order (reference).
_REPLACEMENTS = (
    ("–", "-"), ("‑", "-"), ("—", "-"), ("_", " "),
    ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"),
    ("´", "'"), ("`", "'"),
    ("[", " "), ("]", " "), ("|", " "), ("/", " "), ("#", " "),
    ("→", " "), ("←", " "),
    ("♥", ""), ("☆", ""), ("♡", ""), ("©", ""), ("\\", ""),
    ("@", " at "),
    ("e.g.,", "for example, "), ("i.e.,", "that is, "),
)

_SPACE = " \t\n\r\f\v"          # reference uses ASCII isspace only
_PUNCT_AFTER_SPACE = ",.!?;:'"
_ENDING_PUNCT = set(".!?;:,'\")]}>") | set("…。」』】〉》›»“”‘’")


def preprocess_text(text: str, lang: str) -> str:
    """Reference ``PreprocessText``: normalize punctuation/whitespace, ensure terminal
    punctuation, wrap in ``<lang>...</lang>``."""
    for src, dst in _REPLACEMENTS:
        text = text.replace(src, dst)
    text = "".join(c for c in text if not 0x1F000 <= ord(c) <= 0x1FFFF)

    out: list[str] = []
    i = 0
    while i < len(text):  # " ," -> ","
        if text[i] == " " and i + 1 < len(text) and text[i + 1] in _PUNCT_AFTER_SPACE:
            out.append(text[i + 1])
            i += 2
        else:
            out.append(text[i])
            i += 1
    text = "".join(out)
    text = text.replace('""', '"').replace("''", "'")

    out = []
    last_space = False
    for c in text:  # collapse ASCII whitespace runs
        if c in _SPACE:
            if not last_space:
                out.append(" ")
            last_space = True
        else:
            out.append(c)
            last_space = False
    text = "".join(out).strip()

    if text and text[-1] not in _ENDING_PUNCT:
        text += "."
    return f"<{lang}>{text}</{lang}>" if text else ""


def decompose(text: str) -> list[int]:
    """Per-codepoint NFKD decomposition to BMP values (reference ``TextToUnicodeValues``):
    no cross-character reordering, decompositions leaving the BMP keep the original
    codepoint, non-BMP is dropped."""
    values: list[int] = []
    for ch in text:
        cp = ord(ch)
        if cp > 0xFFFF:
            continue
        dec = unicodedata.normalize("NFKD", ch)
        if dec != ch and all(ord(c) <= 0xFFFF for c in dec):
            values.extend(ord(c) for c in dec)
        else:
            values.append(cp)
    return values


def load_indexer(path: str) -> np.ndarray:
    """``unicode_indexer.json``: a JSON array of 65536 codepoint->id int32s."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list) or len(raw) != 65536:
        # BMP-complete or bust: a truncated file maps out-of-range codepoints to a real
        # character id — garbage AUDIO, not a load-time error.
        got = len(raw) if isinstance(raw, list) else type(raw).__name__
        raise ValueError(f"unicode indexer must be a JSON array of 65536 ids, got {got}: {path}")
    return np.asarray(raw, dtype=np.int32)


def load_voice_style(path: str) -> tuple[np.ndarray, np.ndarray]:
    """``voice_styles/*.json``: ``style_ttl``/``style_dp`` as ``{"dims", "data"}``
    (dims ``[1, 50, 256]`` / ``[1, 8, 16]``)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    styles = []
    for key in ("style_ttl", "style_dp"):
        entry = raw.get(key)
        if not isinstance(entry, dict) or "dims" not in entry or "data" not in entry:
            raise ValueError(f"voice style {path} missing '{key}'.dims/.data")
        dims = tuple(int(d) for d in entry["dims"])
        arr = np.asarray(entry["data"], dtype=np.float32).reshape(dims)
        if arr.ndim != 3 or dims[0] != 1:
            raise ValueError(f"voice style {path}: {key} dims must be [1, d1, d2], got {dims}")
        styles.append(arr)
    return styles[0], styles[1]


class SupertonicTtsAdapter(OnDeviceTtsAdapter):
    _label = "Supertonic"
    _join_gap_s = _JOIN_GAP_S

    def __init__(
        self,
        *,
        text_encoder: OnDeviceModel,
        duration_predictor: OnDeviceModel,
        vector_estimator: OnDeviceModel,
        vocoder: OnDeviceModel,
        indexer: np.ndarray,
        style_ttl: np.ndarray,
        style_dp: np.ndarray,
        sample_rate: int,
        latent_chunk: int,   # waveform samples per latent frame
        latent_dim: int,     # vector-estimator channel dim
        language: str,
        num_steps: int,
        speed: float,
        max_len: int,
    ):
        super().__init__()
        self._text_encoder = text_encoder
        self._duration_predictor = duration_predictor
        self._vector_estimator = vector_estimator
        self._vocoder = vocoder
        self._indexer = indexer
        self._style_ttl = style_ttl
        self._style_dp = style_dp
        self._sample_rate = sample_rate
        self._latent_chunk = latent_chunk
        self._latent_dim = latent_dim
        self._language = language
        self.spoken_language = language  # fixed at load: the <lang> tag wrapping every piece
        self._num_steps = num_steps
        self._speed = speed
        self._max_len = max_len
        self._rng = np.random.default_rng()
        self.output_rate = sample_rate
        self._log = logger.bind(component="tts-supertonic")

    @classmethod
    def from_config(cls, cfg: SupertonicTtsConfig) -> SupertonicTtsAdapter:
        sc = cfg
        if sc.language not in LANGUAGES:
            raise ValueError(
                f"unsupported supertonic language '{sc.language}' "
                f"(note: zh is NOT in the model; supported: {sorted(LANGUAGES)})"
            )
        # ALL side files before any model load: a malformed one must fail before four
        # expensive graph loads.
        with open(sc.tts_json_path, encoding="utf-8") as f:  # type: ignore[arg-type]
            tts_json = json.load(f)
        ae, ttl = tts_json["ae"], tts_json["ttl"]
        compress = int(ttl["chunk_compress_factor"])
        sample_rate = int(ae["sample_rate"])
        latent_chunk = int(ae["base_chunk_size"]) * compress
        latent_dim = int(ttl["latent_dim"]) * compress
        indexer = load_indexer(sc.unicode_indexer_path)  # type: ignore[arg-type]
        style_ttl, style_dp = load_voice_style(sc.voice_style_path)  # type: ignore[arg-type]

        model_kw = dict(
            core_mask=sc.core_mask, target=sc.target, device_id=sc.device_id,
            providers=sc.execution_providers, provider_options=sc.provider_options,
            # prepack stays: int8 graphs, and synth speed is JIT-deadline-critical
            profile="bulk", prepack=True,
        )
        with ExitStack() as models:
            graphs = [
                models.enter_context(OnDeviceModel(path, **model_kw))  # type: ignore[arg-type]
                for path in (
                    sc.text_encoder_path, sc.duration_predictor_path,
                    sc.vector_estimator_path, sc.vocoder_path,
                )
            ]
            adapter = cls(
                text_encoder=graphs[0],
                duration_predictor=graphs[1],
                vector_estimator=graphs[2],
                vocoder=graphs[3],
                indexer=indexer,
                style_ttl=style_ttl,
                style_dp=style_dp,
                sample_rate=sample_rate,
                latent_chunk=latent_chunk,
                latent_dim=latent_dim,
                language=sc.language,
                num_steps=sc.num_steps,
                speed=sc.speed,
                max_len=sc.max_len or (120 if sc.language in ("ko", "ja") else 300),
            )
            models.pop_all()  # success: the adapter owns the models now
            return adapter

    def release(self) -> None:
        for model in (self._text_encoder, self._duration_predictor,
                      self._vector_estimator, self._vocoder):
            model.release()

    def _piece_budget(self) -> int:
        return self._max_len

    def _can_speak(self, ch: str) -> bool:
        """Does every value ``ch`` decomposes to index into the embedding table?

        The indexer IS the front-end, so a ``-1`` gathers noise from the end of the
        table. NFKD is what keeps precomposed letters and Hangul speakable, so a ``-1``
        after decomposition is genuinely unmapped (most CJK ideographs: ~7.5k of 21k
        are mapped). Non-BMP decomposes to nothing, which is silence."""
        values = decompose(ch)
        return bool(values) and all(self._indexer[v] >= 0 for v in values)

    def _synthesize_piece(self, text: str) -> np.ndarray:
        processed = preprocess_text(text, self._language)
        if not processed:
            return np.zeros(0, dtype=np.float32)
        values = decompose(processed)
        # Unknown codepoints map to -1, which the embedding Gather resolves from the end
        # of the table (reference does too). Indexing is total: decompose emits only BMP
        # and load_indexer guarantees all 65536 entries.
        ids = self._indexer[np.asarray(values, dtype=np.int64)].astype(np.int64)[None, :]
        text_mask = np.ones((1, 1, ids.shape[1]), dtype=np.float32)

        duration = float(
            np.asarray(
                self._duration_predictor.run(
                    [("text_ids", ids), ("style_dp", self._style_dp), ("text_mask", text_mask)]
                )[0]
            ).reshape(-1)[0]
        )
        duration = max(duration / self._speed, _MIN_DURATION_S)

        text_emb = np.asarray(
            self._text_encoder.run(
                [("text_ids", ids), ("style_ttl", self._style_ttl), ("text_mask", text_mask)]
            )[0],
            dtype=np.float32,
        )

        wav_len = max(int(duration * self._sample_rate), 1)
        latent_len = min(math.ceil(wav_len / self._latent_chunk), _MAX_LATENT_LEN)
        latent = self._rng.standard_normal(
            (1, self._latent_dim, latent_len), dtype=np.float32
        )
        latent_mask = np.ones((1, 1, latent_len), dtype=np.float32)
        total_step = np.array([self._num_steps], dtype=np.float32)

        # The graph does the Euler update internally: each output IS the next latent.
        for step in range(self._num_steps):
            latent = np.asarray(
                self._vector_estimator.run(
                    [
                        ("noisy_latent", latent),
                        ("text_emb", text_emb),
                        ("style_ttl", self._style_ttl),
                        ("latent_mask", latent_mask),
                        ("text_mask", text_mask),
                        ("current_step", np.array([step], dtype=np.float32)),
                        ("total_step", total_step),
                    ]
                )[0],
                dtype=np.float32,
            )

        waveform = self._vocoder.run([("latent", latent)])[0]
        return np.asarray(waveform, dtype=np.float32).reshape(-1)[:wav_len]
