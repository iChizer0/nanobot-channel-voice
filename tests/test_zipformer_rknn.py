"""Zipformer .rknn static contract: meta.json sidecar loading, zero-state init,
declared input ordering, host-side cached_len increments (the rv1126b port drops
the int64 new_cached_len_* outputs), the load-time contract probe, and the
registry's loud-degrade paths — all against fake on-device models with tiny dims
(the adapter never hardcodes 512/6254)."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("kaldi_native_fbank")

from nanobot_channel_voice.config import SttConfig, ZipformerSttConfig  # noqa: E402
from nanobot_channel_voice.stt import make_stt  # noqa: E402
from nanobot_channel_voice.stt import zipformer as zf_mod  # noqa: E402
from nanobot_channel_voice.stt.zipformer import ZipformerOnDeviceStt  # noqa: E402

_VOCAB = 20
_DIM = 8
# Declared encoder IO, tiny but shape-faithful: x + 3 states, one of them the
# int64 cached_len the port increments host-side (no new_cached_len_0 output).
# cached_key_0 is the declared NHWC feed; the model returns it NCHW [1, 8, 4, 1].
_INPUTS = [
    ["x", [1, 39, 80], "float32"],
    ["cached_len_0", [2, 1], "int64"],
    ["cached_avg_0", [1, 1, _DIM], "float32"],
    ["cached_key_0", [1, 4, 1, _DIM], "float32"],
]
_OUTPUTS = ["encoder_out", "new_cached_avg_0", "new_cached_key_0"]


class FakeModel:
    """One fake per path suffix: encoder scripts outputs in declared order and
    records input order; the joiner emits blanks until a test scripts tokens
    (the construction probe must decode nothing)."""

    def __init__(self, path: str):
        self.path = path
        self.calls: list[list[str]] = []
        self.released = False
        self.joiner_script: list[int] = []  # tokens for successive calls, then blanks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def input_specs(self):
        return []  # .rknn has no introspection: the adapter must use the sidecar

    def output_names(self):
        return []

    def metadata(self):
        return {}

    def run(self, inputs):
        self.calls.append([n for n, _ in inputs])
        if self.path.endswith("encoder.rknn"):
            return [
                np.zeros((1, 2, _DIM), dtype=np.float32),  # 2 output frames per chunk
                np.ones((1, 1, _DIM), dtype=np.float32),   # new_cached_avg_0 = ones
                np.ones((1, _DIM, 4, 1), dtype=np.float32),  # new_cached_key_0, NCHW
            ]
        if self.path.endswith("decoder.rknn"):
            return [np.zeros((1, _DIM), dtype=np.float32)]
        tok = self.joiner_script.pop(0) if self.joiner_script else 0
        logit = np.full((1, _VOCAB), -10.0, dtype=np.float32)
        logit[0, tok] = 10.0
        return [logit]

    def release(self):
        self.released = True


@pytest.fixture
def fakes(monkeypatch):
    made: dict[str, FakeModel] = {}

    def factory(path, **kw):
        made[path] = FakeModel(path)
        return made[path]

    monkeypatch.setattr(zf_mod, "OnDeviceModel", factory)
    return made


@pytest.fixture
def sidecar(tmp_path):
    meta = {
        "encoder_inputs": _INPUTS,
        "encoder_outputs": _OUTPUTS,
        "state_increments": {"cached_len_0": 16},
        "state_feedback_transpose": [0, 2, 3, 1],
        "T": 39,
        "decode_chunk_len": 32,
        "context_size": 2,
        "vocab_size": _VOCAB,
    }
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    (tmp_path / "tokens.txt").write_text("<blk> 0\n▁x 7\n▁y 9\n")
    for name in ("encoder.rknn", "decoder.rknn", "joiner.rknn"):
        (tmp_path / name).write_bytes(b"fake")
    return tmp_path


def _cfg(sidecar, **kw):
    vals = {
        "encoderPath": str(sidecar / "encoder.rknn"),
        "decoderPath": str(sidecar / "decoder.rknn"),
        "joinerPath": str(sidecar / "joiner.rknn"),
        "tokensPath": str(sidecar / "tokens.txt"),
        "metaPath": str(sidecar / "meta.json"),
    }
    vals.update(kw)
    return ZipformerSttConfig.model_validate(vals)


def _one_second() -> bytes:
    return np.full(16000, 300, dtype="<i2").tobytes()


def test_builds_from_sidecar_with_declared_state_contract(fakes, sidecar):
    adapter = ZipformerOnDeviceStt.from_config(_cfg(sidecar))
    assert fakes[str(sidecar / "encoder.rknn")].calls  # load probe ran a real chunk
    s = adapter.stream_start()
    assert s.states["cached_len_0"].dtype == np.int64
    assert s.states["cached_len_0"].shape == (2, 1)
    assert s.states["cached_avg_0"].dtype == np.float32
    assert adapter._chunk_t == 39 and adapter._chunk_shift == 32
    adapter.release()
    assert all(m.released for m in fakes.values())


def test_states_feed_in_declared_order_and_len_increments_host_side(fakes, sidecar):
    adapter = ZipformerOnDeviceStt.from_config(_cfg(sidecar))
    enc = fakes[str(sidecar / "encoder.rknn")]
    pre = len(enc.calls)  # the construction probe's chunk
    s = adapter.stream_start()
    s.accept(_one_second())
    n_chunks = len(enc.calls) - pre
    assert n_chunks, "encoder never ran on a full second of audio"
    for call in enc.calls:
        assert call == ["x", "cached_len_0", "cached_avg_0", "cached_key_0"]
    # len state advanced by the sidecar increment per chunk; the others took the
    # model's new_* outputs (ones), NOT an increment.
    assert s.states["cached_len_0"].tolist() == [[16 * n_chunks], [16 * n_chunks]]
    assert (s.states["cached_avg_0"] == 1.0).all()
    adapter.release()


def test_text_decodes_through_the_rknn_path(fakes, sidecar):
    adapter = ZipformerOnDeviceStt.from_config(_cfg(sidecar))
    fakes[str(sidecar / "joiner.rknn")].joiner_script = [7, 0, 9]  # token, blank, token
    s = adapter.stream_start()
    s.accept(_one_second())
    text = s.finish()
    assert text == "x y"  # the joiner script, ▁ -> space
    adapter.release()


def test_missing_meta_path_degrades_to_none(fakes, sidecar):
    cfg = SttConfig.model_validate({
        "provider": "zipformer",
        "zipformer": {
            "encoderPath": str(sidecar / "encoder.rknn"),
            "decoderPath": str(sidecar / "decoder.rknn"),
            "joinerPath": str(sidecar / "joiner.rknn"),
            "tokensPath": str(sidecar / "tokens.txt"),
        },
    })
    assert make_stt(cfg) is None
    assert not fakes  # sidecar-first: the models were never loaded


def test_4d_states_transpose_back_to_the_declared_feed_layout(fakes, sidecar):
    adapter = ZipformerOnDeviceStt.from_config(_cfg(sidecar))
    s = adapter.stream_start()
    s.accept(_one_second())
    # new_cached_key_0 comes back NCHW ones[1,8,4,1]; the sidecar permutation
    # restores the declared NHWC feed [1,4,1,8]. 3D/2D states are untouched.
    assert s.states["cached_key_0"].shape == tuple(_INPUTS[3][1])
    assert (s.states["cached_key_0"] == 1.0).all()
    assert s.states["cached_avg_0"].shape == (1, 1, _DIM)
    adapter.release()


def test_state_without_feedback_path_raises(fakes, sidecar):
    meta = json.loads((sidecar / "meta.json").read_text())
    meta["state_increments"] = {}  # cached_len_0 now has no new_ output AND no increment
    (sidecar / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="no feedback path"):
        ZipformerOnDeviceStt.from_config(_cfg(sidecar))
    assert all(m.released for m in fakes.values())  # ExitStack gave the NPU back


def test_extra_declared_output_fails_at_load_not_mid_utterance(fakes, sidecar):
    meta = json.loads((sidecar / "meta.json").read_text())
    meta["encoder_outputs"] = [*_OUTPUTS, "new_ghost_0"]  # meta/model output mismatch
    (sidecar / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):  # the probe's strict output zip
        ZipformerOnDeviceStt.from_config(_cfg(sidecar))
    assert all(m.released for m in fakes.values())


def test_bad_sidecar_names_the_file(fakes, sidecar):
    (sidecar / "meta.json").write_text('{"encoder_inputs": []}')
    with pytest.raises(RuntimeError, match="bad zipformer meta sidecar"):
        ZipformerOnDeviceStt.from_config(_cfg(sidecar))
    assert not fakes  # sidecar-first: the models were never loaded
