"""``nanobot-voice wake-eval``: what the configured wake word scores on a recording, chunk
by chunk on the recording's own clock."""

from __future__ import annotations

import json
import types
import wave

import pytest

from nanobot_channel_voice import cli


def _wav(path, rate: int, seconds: float = 1.0, channels: int = 1, width: int = 2) -> str:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(rate)
        f.writeframes(bytes(int(rate * seconds) * width * channels))
    return str(path)


def _config(tmp_path, section) -> str:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"channels": {"voice": section}}), encoding="utf-8")
    return str(path)


class _Detector:
    """Scores 0.95 on pushes 5-7 (a run of three) and fires on the 6th; quiet elsewhere."""

    last_hit_back_bytes = 0

    def __init__(self):
        self.pushes = 0
        self.last_score = None
        self.released = False
        self.clock: list[float | None] = []

    def push(self, frame, now=None):
        assert len(frame) == cli._EVAL_CHUNK_B  # one decision per push
        self.clock.append(now)
        self.pushes += 1
        self.last_score = 0.95 if 5 <= self.pushes <= 7 else 0.01
        return self.pushes == 6

    def take_stats(self):
        return (0.95, 3)

    def reset(self):
        self.pushes = 0

    def release(self):
        self.released = True


@pytest.fixture
def detector(monkeypatch):
    det = _Detector()
    knobs = types.SimpleNamespace(threshold=0.6, patience=2)
    monkeypatch.setattr(cli, "_wake_detector", lambda args: (det, knobs, None))
    return det


def test_wake_eval_scores_a_recording_chunk_by_chunk(tmp_path, capsys, detector):
    wav = _wav(tmp_path / "phrase.wav", 16000)
    assert cli.main(["wake-eval", wav, "--config", _config(tmp_path, {})]) == 0
    assert capsys.readouterr().out.strip() == (
        f"{wav}: 12 chunks, peak 0.95 at 0.40 s, longest run 3 at/above 0.60, "
        "1 hit(s), phrase end at 0.48 s (patience 2)"
    )
    assert detector.clock[:3] == [0.08, 0.16, 0.24]  # the recording's clock, not the wall's
    assert detector.released


def test_wake_eval_places_a_hit_where_the_channel_places_the_phrase_end(tmp_path, capsys, detector):
    detector.last_hit_back_bytes = 2560  # the run's first chunk, one chunk before the decision
    wav = _wav(tmp_path / "phrase.wav", 16000)
    assert cli.main(["wake-eval", wav, "--config", _config(tmp_path, {})]) == 0
    assert "1 hit(s), phrase end at 0.40 s" in capsys.readouterr().out


def test_wake_eval_reads_every_recording_before_the_models_load(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        cli, "_wake_detector", lambda args: pytest.fail("the models loaded before a recording failed"),
    )
    good = _wav(tmp_path / "good.wav", 16000)
    assert cli.main(["wake-eval", good, str(tmp_path / "typo.wav"), "--config", _config(tmp_path, {})]) == 2
    out, err = capsys.readouterr()
    assert out == "" and "cannot read" in err  # nothing scored, nothing built


def test_wake_eval_says_when_a_recording_is_shorter_than_a_chunk(tmp_path, capsys, detector):
    short = _wav(tmp_path / "short.wav", 16000, seconds=0.05)
    assert cli.main(["wake-eval", short, "--config", _config(tmp_path, {})]) == 0
    assert capsys.readouterr().out.strip() == f"{short}: shorter than one 80 ms chunk"


def test_wake_eval_prints_every_chunk_on_request(tmp_path, capsys, detector):
    wav = _wav(tmp_path / "phrase.wav", 16000)
    assert cli.main(["wake-eval", wav, "--all", "--config", _config(tmp_path, {})]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "    0.08 s  0.010"
    assert lines[5] == "    0.48 s  0.950  HIT"
    assert len(lines) == 13  # 12 chunks + the summary


def test_wake_eval_prints_nan_for_a_chunk_the_detector_could_not_score(tmp_path, capsys, detector):
    orig = detector.push

    def push(frame, now=None):  # the 3rd chunk raises inside the detector: last_score untouched
        if detector.pushes == 2:
            detector.pushes += 1
            return False
        return orig(frame, now)

    detector.push = push
    wav = _wav(tmp_path / "phrase.wav", 16000)
    assert cli.main(["wake-eval", wav, "--all", "--config", _config(tmp_path, {})]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[2] == "    0.24 s  nan"  # not the 2nd chunk's score again
    assert lines[-1].endswith(
        "peak 0.95 at 0.40 s, longest run 3 at/above 0.60, 1 hit(s), phrase end at 0.48 s (patience 2)"
    )


def test_wake_eval_keeps_a_non_finite_score_out_of_the_peak(tmp_path, capsys, detector):
    orig = detector.push

    def push(frame, now=None):  # the 2nd chunk: a runtime gone bad scores inf
        hit = orig(frame, now)
        if detector.pushes == 2:
            detector.last_score = float("inf")
        return hit

    detector.push = push
    wav = _wav(tmp_path / "phrase.wav", 16000)
    assert cli.main(["wake-eval", wav, "--all", "--config", _config(tmp_path, {})]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "    0.16 s  inf"
    assert "peak 0.95 at 0.40 s" in lines[-1]  # not "at 0.16 s"


def test_wake_eval_prints_no_time_for_a_peak_of_nothing(tmp_path, capsys, detector):
    detector.push = lambda frame, now=None: False  # last_score stays None: nan every chunk
    detector.take_stats = lambda: (0.0, 0)
    wav = _wav(tmp_path / "phrase.wav", 16000)
    assert cli.main(["wake-eval", wav, "--config", _config(tmp_path, {})]) == 0
    assert capsys.readouterr().out.strip().endswith(
        "12 chunks, peak 0.00, longest run 0 at/above 0.60, 0 hit(s) (patience 2)"
    )


def test_wake_eval_resamples_and_resets_between_recordings(tmp_path, capsys, detector):
    a = _wav(tmp_path / "a.wav", 48000)
    b = _wav(tmp_path / "b.wav", 16000)
    assert cli.main(["wake-eval", a, b, "--config", _config(tmp_path, {})]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith(f"{a}: 12 chunks") and "phrase end at 0.48 s" in out[0]
    assert out[1].startswith(f"{b}: 12 chunks") and "phrase end at 0.48 s" in out[1]


def test_wake_eval_downmixes_stereo_and_refuses_what_is_not_16_bit(tmp_path, capsys, detector):
    stereo = _wav(tmp_path / "stereo.wav", 16000, channels=2)
    assert cli.main(["wake-eval", stereo, "--config", _config(tmp_path, {})]) == 0
    assert "12 chunks" in capsys.readouterr().out
    eight = _wav(tmp_path / "eight.wav", 16000, width=1)
    assert cli.main(["wake-eval", eight, "--config", _config(tmp_path, {})]) == 2
    assert "not a 16-bit PCM WAV" in capsys.readouterr().err


def test_wake_eval_checks_the_flags_before_anything_loads(tmp_path, capsys):
    cfg = _config(tmp_path, {})
    wav = _wav(tmp_path / "phrase.wav", 16000)
    assert cli.main(["wake-eval", wav, "--threshold", "2", "--config", cfg]) == 2
    assert "--threshold is a score" in capsys.readouterr().err
    assert cli.main(["wake-eval", wav, "--threshold", "nan", "--config", cfg]) == 2
    assert "--threshold is a score" in capsys.readouterr().err
    assert cli.main(["wake-eval", wav, "--patience", "0", "--config", cfg]) == 2
    assert "--patience is a count" in capsys.readouterr().err


# The configured acoustic tier, on the detector tests' fake models.
_TIER = {"wake": {
    "mode": "gate", "engine": "openwakeword", "phrases": ["hey nanobot"],
    "openwakeword": {"melPath": "mel.onnx", "embeddingPath": "emb.onnx", "modelPath": "head.onnx"},
}}


@pytest.fixture
def fake_models(monkeypatch):
    import numpy as np
    import test_wake_word as tw

    import nanobot_channel_voice.wake.openwakeword as oww_mod

    class _HalfHead(tw.FakeHead):
        def run(self, inputs):  # every chunk scores 0.5: under the default, over a lowered bar
            return [np.array([[0.5]], dtype=np.float32)]

    made = {"mel.onnx": tw.FakeMel(), "emb.onnx": tw.FakeEmb(), "head.onnx": _HalfHead()}
    monkeypatch.setattr(oww_mod, "OnDeviceModel", lambda path, **kw: made[path])
    return made


def test_wake_eval_builds_the_configured_detector_and_the_flags_reach_it(
    tmp_path, capsys, fake_models,
):
    wav = _wav(tmp_path / "phrase.wav", 16000)
    cfg = _config(tmp_path, _TIER)
    assert cli.main(["wake-eval", wav, "--config", cfg]) == 0
    out = capsys.readouterr().out
    assert "peak 0.50" in out and "longest run 0 at/above 0.60, 0 hit(s) (patience 2)" in out
    assert cli.main(["wake-eval", wav, "--config", cfg, "--threshold", "0.3", "--patience", "1"]) == 0
    out = capsys.readouterr().out
    # One hit at the first chunk; the score never dips, so the hysteresis never re-arms.
    assert "longest run 12 at/above 0.30, 1 hit(s), phrase end at 0.08 s (patience 1)" in out


def test_wake_eval_notes_a_config_that_runs_no_wake_word(tmp_path, capsys, fake_models):
    wav = _wav(tmp_path / "phrase.wav", 16000)
    off = {**_TIER["wake"], "mode": "off"}
    assert cli.main(["wake-eval", wav, "--config", _config(tmp_path, {"wake": off})]) == 0
    out, err = capsys.readouterr()
    assert "12 chunks" in out and "note: this config runs no acoustic wake word (wake.mode is off)" in err
    served = {"backend": "openai", **_TIER}  # realtime.uplink server: no gate, no detector
    assert cli.main(["wake-eval", wav, "--config", _config(tmp_path, served)]) == 0
    assert "(realtime.uplink is server)" in capsys.readouterr().err
    assert cli.main(["wake-eval", wav, "--config", _config(tmp_path, _TIER)]) == 0
    assert capsys.readouterr().err == ""  # the local pipeline runs it: nothing to note


def test_wake_eval_runs_the_detector_at_the_pipeline_s_capture_rate(tmp_path, capsys, fake_models):
    wav = _wav(tmp_path / "phrase.wav", 16000)
    local = _config(tmp_path, {"audio": {"sampleRate": 48000}, **_TIER})
    assert cli.main(["wake-eval", wav, "--config", local]) == 2
    err = capsys.readouterr().err  # the schema refuses it first, as it refuses the channel
    assert "does not validate" in err and "openwakeword" in err
    off = {**_TIER["wake"], "mode": "off"}  # the schema skips a tier that is off: the detector says it
    assert cli.main(["wake-eval", wav, "--config", _config(tmp_path, {"audio": {"sampleRate": 48000}, "wake": off})]) == 2
    assert "16 kHz" in capsys.readouterr().err
    gated = _config(tmp_path, {
        "audio": {"sampleRate": 48000}, "backend": "openai", "realtime": {"uplink": "wake"},
        "vad": {"engine": "silero"}, **_TIER,
    })
    assert cli.main(["wake-eval", wav, "--config", gated]) == 0  # a cloud gate captures at 16 kHz


def test_wake_eval_resolves_weights_from_the_named_store(tmp_path, capsys, fake_models, monkeypatch):
    import os

    monkeypatch.delenv("NANOBOT_VOICE_MODELS_DIR", raising=False)
    wav = _wav(tmp_path / "phrase.wav", 16000)
    store = tmp_path / "store"
    assert cli.main(["--models-dir", str(store), "wake-eval", wav, "--config", _config(tmp_path, _TIER)]) == 0
    assert os.environ["NANOBOT_VOICE_MODELS_DIR"] == str(store)  # as the channel resolves it


def test_wake_eval_needs_the_acoustic_tier_configured(tmp_path, capsys):
    wav = _wav(tmp_path / "phrase.wav", 16000)
    assert cli.main(["wake-eval", wav, "--config", _config(tmp_path, {})]) == 2
    assert "wake.engine" in capsys.readouterr().err
    section = {"wake": {"mode": "gate", "engine": "openwakeword", "phrases": ["hey nanobot"]}}
    assert cli.main(["wake-eval", wav, "--config", _config(tmp_path, section)]) == 2
    assert "unset" in capsys.readouterr().err  # no package named, no paths: said before a model loads
