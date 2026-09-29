"""Every test sees a weights store and an environment of its own.

The form, the validator and the plan all read the store and the operator's defaults from
the environment, and the model index from the config, so a machine that has ever run
``nanobot-voice list``, a board that exports ``$NANOBOT_VOICE_DEFAULTS``, or a real
``~/.nanobot/config.json`` would otherwise change what the tests assert. The validator
also reads what a started channel published, which a test that never stopped one leaves,
and the process's sync store, whose runs and locks belong to one test's event loop.
"""

import pytest


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    """The store this test writes and reads; its path is the fixture's value."""
    import nanobot.config.loader as loader

    from nanobot_channel_voice import running, webui_sync

    root = tmp_path / "store"
    monkeypatch.setenv("NANOBOT_VOICE_MODELS_DIR", str(root))
    monkeypatch.delenv("NANOBOT_VOICE_DEFAULTS", raising=False)
    monkeypatch.setattr(loader, "get_config_path", lambda: tmp_path / "nanobot" / "config.json")
    monkeypatch.setattr(running, "_current", None)
    monkeypatch.setattr(webui_sync, "_STORE", None)  # its locks and tasks bind to one loop
    return root
