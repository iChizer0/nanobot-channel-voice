"""Every test sees a weights store and an environment of its own.

The form, the validator and the plan all read the store and the operator's defaults from
the environment, so a machine that has ever run ``nanobot-voice list`` — or a board that
exports ``$NANOBOT_VOICE_DEFAULTS`` — would otherwise change what the tests assert.
"""

import pytest


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    """The store this test writes and reads; its path is the fixture's value."""
    root = tmp_path / "store"
    monkeypatch.setenv("NANOBOT_VOICE_MODELS_DIR", str(root))
    for leaked in ("NANOBOT_VOICE_DEFAULTS", "NANOBOT_VOICE_INDEX"):
        monkeypatch.delenv(leaked, raising=False)
    return root
