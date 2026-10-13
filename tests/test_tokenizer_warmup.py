"""The tokenizer the core loads at its first token estimate is fetched off the loop at
start, into a cache that outlives the temp dir."""

from __future__ import annotations

import os
import sys
import threading
import types

from loguru import logger

from nanobot_channel_voice.channel import VoiceChannel, _persist_tokenizer_cache

_START = VoiceChannel._start_tokenizer_warmup  # before conftest's fixture stubs it


def _warm() -> None:
    VoiceChannel._warm_tokenizer(types.SimpleNamespace(logger=logger))


def test_the_cache_dir_is_ours_unless_the_operator_chose_one(monkeypatch, tmp_path):
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    _persist_tokenizer_cache()
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(tmp_path / "nanobot-voice" / "tiktoken")
    assert (tmp_path / "nanobot-voice" / "tiktoken").is_dir()

    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "/somewhere/else")
    _persist_tokenizer_cache()
    assert os.environ["TIKTOKEN_CACHE_DIR"] == "/somewhere/else"


def test_an_uncreatable_cache_dir_leaves_tiktokens_default(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(blocker))  # mkdir under a file fails
    _persist_tokenizer_cache()
    assert "TIKTOKEN_CACHE_DIR" not in os.environ


def test_no_home_directory_leaves_tiktokens_default(monkeypatch):
    from pathlib import Path

    def no_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    _persist_tokenizer_cache()  # no raise: the channel still comes up
    assert "TIKTOKEN_CACHE_DIR" not in os.environ


def test_the_encoding_is_fetched(monkeypatch):
    calls: list[str] = []
    monkeypatch.setitem(sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=calls.append))
    _warm()
    assert calls == ["cl100k_base"]


def test_without_tiktoken_nothing_happens(monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)  # import raises ImportError
    _warm()


def test_a_failed_fetch_only_logs(monkeypatch):
    def refuse(name):
        raise OSError("offline")

    monkeypatch.setitem(sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=refuse))
    _warm()  # no raise


def test_one_warmup_thread_at_a_time(monkeypatch):
    started: list[threading.Thread] = []

    class _Thread:
        def __init__(self, **kw):
            self.alive = True
            started.append(self)

        def start(self):
            pass

        def is_alive(self):
            return self.alive

    monkeypatch.setattr(threading, "Thread", _Thread)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "/already/set")
    ch = types.SimpleNamespace(_tokenizer_thread=None, _warm_tokenizer=lambda: None)
    _START(ch)
    _START(ch)  # a rebuild while the first still fetches
    assert len(started) == 1
    started[0].alive = False
    _START(ch)
    assert len(started) == 2
