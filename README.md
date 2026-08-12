# nanobot-channel-voice

A voice channel plugin for [nanobot](https://github.com/HKUDS/nanobot), talk to your agent naturally over audio.

- **ALSA-direct**: capture and playback via `arecord`/`aplay`, shared `dsnoop`/`dmix`/`plug` devices work by name with no extra audio dependencies.
- **Pluggable STT**: reuses nanobot's own `transcription` config (Whisper, cloud or LAN), or runs on-device over ONNX/RKNN, batch or streaming (decoding *while* you speak).
- **Pluggable TTS**: OpenAI-compatible `/audio/speech` (cloud or a local server), on-device over ONNX/RKNN, or zero-dependency `espeak-ng`/`say`.
- **Two on-device inference backends**: `.onnx` (CPU, or GPU/DLA on Jetson via the TensorRT/CUDA execution providers) and `.rknn` (Rockchip NPU). One `OnDeviceModel` dispatches on file extension, so the same adapter code drives both.
- **Duck-then-confirm barge-in**: half-duplex mutes the mic while speaking, or the open-mic modes (hardware AEC, software AEC3, or none at all) duck the reply the moment you start talking, then confirm. A real interruption stops playback (with streaming STT, mid-sentence) and the agent is told how much of its reply you actually heard. An echo, a cough, or an "uh-huh" releases the duck and the reply continues.
- **E2E speech-to-speech**: alternative backend, one WebSocket session to an OpenAI-Realtime-dialect provider does turn detection + ASR + reasoning + TTS, while the model's tool calls still route through nanobot's guarded tool registry.

```mermaid
flowchart TB
  mic([mic]) -->|"ALSA capture"| duplex
  duplex["duplex + echo control<br/>half-duplex mic gate, or<br/>open mic with hardware,<br/>software, or no AEC"]
  sink["audio sink<br/>paced playback ·<br/>duck · flush"]
  spk([speaker])
  agent["nanobot core<br/>LLM · tools · memory"]

  subgraph listen["local backend · listen"]
    vad["VAD + endpointing<br/>energy · spectral · neural<br/>+ ML end-of-turn model"] --> stt
    stt["STT<br/>cloud/LAN API, or<br/>on-device batch or streaming"]
  end

  subgraph speak["local backend · speak"]
    chunker["sentence chunker<br/>speaks as the reply streams"] --> tts
    tts["TTS<br/>cloud/LAN API, on-device<br/>neural, or system voice"]
  end

  s2s["E2E speech-to-speech<br/>one realtime session:<br/>VAD + ASR + LLM + TTS"]

  duplex -->|"backend: local"| vad
  duplex -->|"backend: a realtime provider"| s2s
  stt -->|"utterance text"| agent
  agent -->|"streamed reply"| chunker
  s2s -.->|"tool calls (guarded registry)"| agent
  tts -->|"reply audio"| sink
  s2s -->|"reply audio"| sink
  sink -->|"ALSA playback"| spk
  sink -.->|"echo reference"| duplex

  vad ==>|"barge-in: duck, then flush"| sink
  stt ==>|"confirmed: /stop the turn"| agent
  s2s ==>|"server VAD barge-in"| sink
```

> Thick arrows are the interrupt path, dotted ones side channels; the bullets above name the engines behind each slot.

## Install

nanobot >= 0.3.0 discovers channels as subpackages of `nanobot.channels`; this wheel installs a dependency-free manifest into that namespace (`nanobot/channels/voice`). It needs Linux with ALSA, Python 3.11-3.13 (a [uv](https://github.com/astral-sh/uv)-managed venv is recommended).

```sh
uv pip install nanobot-ai
uv pip install -e ./nanobot-channel-voice
nanobot channels status
```

The default stack (ALSA, energy VAD, OpenAI-compatible TTS) needs no extras. Each of these is lazily imported, so the plugin runs and degrades gracefully without them:

- `[ondevice]` is every on-device engine over local ONNX models (CPU, via onnxruntime): STT `whisper`/`sensevoice`/`zipformer`, TTS `mms`/`supertonic`, VAD `firered`, end-of-turn `smartturn` (Smart Turn v3).
- `[rknn]` adds Rockchip's NPU runtime (`rknn-toolkit-lite2`) for `.rknn` artifacts, on top of `[ondevice]`. Requires aarch64 arch and Python <= 3.12, elsewhere it installs nothing, since `.onnx` is the non-board path.
- `[realtime]` is the WebSocket client for the E2E cloud backends (`backend: "openai"` and its dialects).
- `[aec]` is software echo cancellation (`aec: "webrtc"`, WebRTC AEC3), large wheel.
- `[webrtc]` is the spectral VAD (`vad.engine: "webrtc"`).
- `[pyalsa]` is the in-process libasound backend (`audio.backend: "pyalsa"`) instead of the `arecord`/`aplay` subprocesses, needs `libasound2-dev` to build.
- `[otel]` exports tool-call outcomes to OpenTelemetry (`telemetry.enabled`), cloud backends only.
- `[uroman]` / `[japanese]` are MMS-TTS text frontends for non-Latin scripts (`tts.mms.textFrontend`): romanization, and kanji-aware readings for Japanese (need build).

## Get started

Run `nanobot webui`, open **Settings -> Channels -> Voice**, paste your config into the **Import Json** box - the whole form; a partial paste patches just those keys over the current section - and **Check and enable** - see [Example Configs](docs/EXAMPLE_CONFIGS.md) for various copy-pasteable setups and troubleshooting. `nanobot-voice config` prints the current section back, paste-ready (API keys withheld unless `--secrets`). Every key - default, range, per-field note - is documented inline in the schema, see [config.py](nanobot_channel_voice/config.py).

## License

See [LICENSE](LICENSE). Optional on-device model weights carry their own licenses, review them for your deployment, no weights are bundled.
