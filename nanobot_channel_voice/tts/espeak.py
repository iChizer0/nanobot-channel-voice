"""espeak-ng IPA phonemization for matcha. Ladder: ``espeakPath`` binary -> system
binary -> the ``[espeak]`` extra's espeakng-loader wheel (bundled GPL-3 libespeak-ng
+ data, for boards without a package manager), driven via ctypes."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

_TIMEOUT_S = 10.0

# speak_lib.h constants
_AUDIO_OUTPUT_SYNCHRONOUS = 2
_INITIALIZE_DONT_EXIT = 0x8000  # bad data path otherwise exit(1)s the process
_CHARS_UTF8 = 1
_PHONEMES_IPA = 0x02

# bundled 1.52.0 snprintf-truncates data paths at 160 bytes into a crashing state
_MAX_DATA_PATH = 140

_lock = threading.Lock()  # espeak has global state: one call at a time
_lib: ctypes.CDLL | None = None  # one espeak_Initialize per process, ever
_lib_voice: str | None = None


def make_ipa_phonemizer(voice: str, *, espeak_path: str | None = None) -> Callable[[str], str]:
    """text -> IPA (espeak ``--ipa`` form, one clause per line). Probed once here so
    a bad path/voice/library fails at BUILD time (registry falls back to system TTS),
    not per chunk where the shell swallows errors into a silent channel."""
    if espeak_path and not (os.path.isfile(espeak_path) and os.access(espeak_path, os.X_OK)):
        raise RuntimeError(f"tts espeakPath '{espeak_path}' is not an executable file")
    exe = espeak_path or shutil.which("espeak-ng") or shutil.which("espeak")
    if exe:
        phonemize = _subprocess_phonemizer(exe, voice)
    else:
        try:
            lib = _load_library()
        except ImportError:
            raise RuntimeError(
                "espeak-ng not found: install the system package (apt install espeak-ng), "
                "point tts.matcha.espeakPath at a binary, or pip install "
                "'nanobot-channel-voice[espeak]' for the bundled library"
            ) from None
        phonemize = _library_phonemizer(lib, voice)
    try:
        phonemize("Okay.")
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001 - OSError/Timeout/ctypes: name the probe
        raise RuntimeError(f"espeak-ng probe failed ({exe or 'bundled library'}): {exc}") from exc
    return phonemize


def _subprocess_phonemizer(exe: str, voice: str) -> Callable[[str], str]:
    def phonemize(text: str) -> str:
        proc = subprocess.run(
            [exe, "-q", "--ipa", "-v", voice, "--", text],
            capture_output=True, text=True, timeout=_TIMEOUT_S,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"espeak-ng phonemization failed ({proc.returncode}): "
                f"{proc.stderr.strip()[:200]}"
            )
        return proc.stdout

    return phonemize


def _load_library() -> ctypes.CDLL:
    global _lib
    with _lock:
        if _lib is not None:
            return _lib
        import espeakng_loader  # ImportError -> the caller names the [espeak] extra

        data = Path(espeakng_loader.get_data_path())
        if len(str(data)) >= _MAX_DATA_PATH:
            data = _shorten_data_path(data)
        lib = ctypes.CDLL(str(espeakng_loader.get_library_path()))
        lib.espeak_Initialize.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
        ]
        lib.espeak_Initialize.restype = ctypes.c_int
        lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        lib.espeak_SetVoiceByName.restype = ctypes.c_int
        lib.espeak_TextToPhonemes.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int,
        ]
        lib.espeak_TextToPhonemes.restype = ctypes.c_char_p
        rate = lib.espeak_Initialize(
            _AUDIO_OUTPUT_SYNCHRONOUS, 0, str(data).encode(), _INITIALIZE_DONT_EXIT
        )
        if rate <= 0:
            raise RuntimeError(f"espeak-ng library failed to initialize (data: {data})")
        _lib = lib
        return lib


def _shorten_data_path(data: Path) -> Path:
    # fresh mkdtemp: unique + 0700, no fixed /tmp name to race or trust
    import tempfile

    link = Path(tempfile.mkdtemp(prefix="nb-espeak-")) / "data"
    link.symlink_to(data)
    if len(str(link)) >= _MAX_DATA_PATH:
        raise RuntimeError(
            f"espeak-ng data path too long for the bundled library ({link}); "
            "use a shorter TMPDIR or install the system espeak-ng"
        )
    return link


def _library_phonemizer(lib: ctypes.CDLL, voice: str) -> Callable[[str], str]:
    def phonemize(text: str) -> str:
        global _lib_voice
        with _lock:
            if _lib_voice != voice:
                if lib.espeak_SetVoiceByName(voice.encode()) != 0:
                    raise RuntimeError(f"espeak-ng: unknown voice '{voice}'")
                _lib_voice = voice
            # one clause per call; the pointer advances and NULLs at end of text
            buf = ctypes.create_string_buffer(text.encode("utf-8"))
            ptr = ctypes.c_void_p(ctypes.addressof(buf))
            clauses = []
            while ptr.value:
                clause = lib.espeak_TextToPhonemes(
                    ctypes.byref(ptr), _CHARS_UTF8, _PHONEMES_IPA
                )
                if clause:
                    clauses.append(clause.decode("utf-8", "replace"))
        return "\n".join(clauses)

    return phonemize
