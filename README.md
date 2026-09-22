# nanobot-channel-voice

A voice channel plugin for [nanobot](https://github.com/HKUDS/nanobot), talk to your agent naturally over audio.

- **ALSA-direct**: capture and playback via `arecord`/`aplay`, shared `dsnoop`/`dmix`/`plug` devices work by name with no extra audio dependencies.
- **Pluggable STT**: reuses nanobot's own `transcription` config (Whisper, cloud or LAN), or runs on-device over ONNX/RKNN, batch or streaming (decoding *while* you speak).
- **Pluggable TTS**: OpenAI-compatible `/audio/speech` (cloud or a local server), on-device over ONNX/RKNN, or zero-dependency `espeak-ng`/`say`.
- **Two on-device inference backends**: `.onnx` (CPU, or GPU/DLA on Jetson via the TensorRT/CUDA execution providers) and `.rknn` (Rockchip NPU). One `OnDeviceModel` dispatches on file extension, so the same adapter code drives both.
- **Duck-then-confirm barge-in**: half-duplex mutes the mic while speaking, or the open-mic modes (hardware AEC, software AEC3, or none at all) duck the reply the moment you start talking, then confirm. A real interruption stops playback (with streaming STT, mid-sentence) and the agent is told how much of its reply you actually heard. An echo, a cough, or an "uh-huh" releases the duck and the reply continues.
- **Wake word**: optional, off by default. Two tiers — a transcript-prefix match in any language the STT covers, plus an acoustic detector (openWakeWord) that hears through the bot's own playback. `gate` asks for the phrase on a cold start and then leaves follow-ups natural for a while; `strict` also makes the phrase the *only* thing that interrupts a reply, so a room full of other people never perturbs it. A summon can answer with a spoken ack or a wordless earcon.
- **E2E speech-to-speech**: alternative backend, one WebSocket session to an OpenAI-Realtime-dialect provider or Gemini Live does turn detection + ASR + reasoning + TTS, while the model's tool calls still route through nanobot's guarded tool registry. The on-device VAD and wake word can gate what leaves the box (`realtime.uplink`), so a cloud brain is billed for speech, not for listening.

```mermaid
flowchart TB
  mic([mic]) -->|"ALSA capture"| duplex
  duplex["duplex + echo control<br/>half-duplex mic gate, or<br/>open mic with hardware,<br/>software, or no AEC"]
  sink["audio sink<br/>paced playback ·<br/>duck · flush"]
  spk([speaker])
  agent["nanobot core<br/>LLM · tools · memory"]

  subgraph listen["local backend · listen"]
    vad["VAD + endpointing<br/>energy · spectral · neural<br/>+ ML end-of-turn model"] --> stt
    stt["STT<br/>cloud/LAN API, or<br/>on-device batch or streaming"] --> wake
    wake["wake gate · optional<br/>transcript phrase + acoustic<br/>detector; strict mode also<br/>owns barge-in"]
  end

  subgraph speak["local backend · speak"]
    chunker["sentence chunker<br/>speaks as the reply streams"] --> tts
    tts["TTS<br/>cloud/LAN API, on-device<br/>neural, or system voice"]
  end

  s2s["E2E speech-to-speech<br/>one realtime session:<br/>VAD + ASR + LLM + TTS"]

  duplex -->|"backend: local"| vad
  duplex -->|"backend: a realtime provider"| s2s
  wake -->|"utterance text<br/>(wake phrase stripped)"| agent
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

nanobot (>= 0.3.5) discovers channels as subpackages of `nanobot.channels`; this wheel installs a dependency-free manifest into that namespace (`nanobot/channels/voice`). It needs Linux with ALSA, Python 3.11-3.13 (a [uv](https://github.com/astral-sh/uv)-managed venv is recommended).

```sh
uv pip install nanobot-ai
uv pip install -e ./nanobot-channel-voice
nanobot channels status
```

The default stack (ALSA, energy VAD, OpenAI-compatible TTS) needs no extras. Each of these is lazily imported, so the plugin runs and degrades gracefully without them:

- `[ondevice]` is every on-device engine over local ONNX models (CPU, via onnxruntime): STT `whisper`/`sensevoice`/`zipformer`, TTS `mms`/`supertonic`/`matcha`, VAD `firered`/`silero` (Silero VAD v6), end-of-turn `smartturn` (Smart Turn v3). Matcha's English front-end needs espeak-ng: the system package, or add `[espeak]` for a pip-bundled library on boards without a package manager.
- `[rknn]` adds Rockchip's NPU runtime (`rknn-toolkit-lite2`) for `.rknn` artifacts, on top of `[ondevice]`. Requires aarch64 arch and Python <= 3.12, elsewhere it installs nothing, since `.onnx` is the non-board path.
- `[realtime]` is the WebSocket client for the E2E cloud backends (`backend: "openai"` and its dialects, `"gemini"`).
- `[aec]` is software echo cancellation (`aec: "webrtc"`, WebRTC AEC3), large wheel.
- `[webrtc]` is the spectral VAD (`vad.engine: "webrtc"`).
- `[pyalsa]` is the in-process libasound backend (`audio.backend: "pyalsa"`) instead of the `arecord`/`aplay` subprocesses, needs `libasound2-dev` to build.
- `[otel]` exports tool-call outcomes to OpenTelemetry (`telemetry.enabled`), cloud backends only.
- `[uroman]` / `[japanese]` are MMS-TTS text frontends for non-Latin scripts (`tts.mms.textFrontend`): romanization, and kanji-aware readings for Japanese (need build).

## Get started

Run `nanobot webui`, open **Settings -> Channels -> Voice**. On a core that ships the voice panel, the pane is a form the channel shapes for its own state - pick the backend and the sections follow (speech-to-text, text-to-speech, waiting, wake word, cues; or the realtime provider), every edit re-checks the section and refreshes the **Resolved setup** rows (which selected engine would fall back at start, why, and what stands in until then, or on a cloud provider's gated uplink which detector would not load, the channel not starting without it; what the chosen backend ignores; the audio devices opened at start) and the identity line (the backend and its engines or model), and **Advanced** opens **Config import** (the same edits as the `channels.voice` keys they set, where a whole section or a patch can be pasted), shows in place the less common rows, shaded (the hardware facts - Device, Microphone, Speaker, Audio backend - the audio-path details - Echo cancellation, Barge-in, Audio format - the listening internals - Voice activity detector, End of speech, End of turn, their models - then thresholds and timings, the served endpoint, the phrase lists, the access list; Audio, Interruptions, Listening and Access are those rows alone, so they show under Advanced only, the identity line and the Resolved setup naming what listens and speaks, and any engine that would not load) and the sections this setup does not use, each with a note naming the switch that would (Waiting and Interruptions with replies not spoken, a cloud provider's Listening and Wake word until its uplink gate runs them, its served Speech-to-text); a set value stays in force while its row is hidden and offers **Reset to defaults**, a pending patch like any other: `{"$reset": true}` drops the section's own keys so the defaults show through, with later edits stacked on top and **Discard pending changes** undoing it before it applies. The defaults are the schema's, beneath the operator's `$NANOBOT_VOICE_DEFAULTS` in the gateway environment (a JSON file path or the JSON itself, a partial section): a layer under `channels.voice`, every key the section leaves unset coming from it and then from the schema, so a fresh config runs a board image's baseline as is, the pane shows it resolved, and Reset returns to it. An on-device engine's **Model** is picked from the weights index (cached in the store, so the panel opens on it at once and offline too, and reloaded in the background whenever the panel opens; each pill shows the model's languages and size; the CPU builds, plus the NPU builds for the chip named under **Device**, whose note lists the chips the index has builds for) or typed as any store key - and for the wake word the same row switches the tier, **Transcript** or a head that fills **Phrases** with what it hears, **Custom** being a phrase head you trained yourself; a model built for CPU and chip gets a **Build** row, and picking an engine picks its first model, a cloud provider's On speech or After wake word the Silero detector it runs, After wake word the wake mode too, so the **Wake word** section follows the pick with the head left to you (and leaving that pick takes back a mode nothing has filled, which would else refuse the section from a section the form has folded away) - and **Apply** downloads what the resolved setup runs (the selected engines' models, not a key an earlier engine pick left behind) with progress, removes the models the panel itself installed and the setup no longer runs, then starts or restarts the channel; the selected model's license shows under its row, and a model whose license carries a notice is fetched only after you tick it under **Models**, Apply saying it waits for the tick and the Resolved setup naming the model whose notice holds it. On an official core without the panel, the pane is that paste box alone (**Import Json**, the whole form or a partial patch; `enabled` belongs to the toggle and is dropped from a paste) and a refused paste shows its reason when you enable. Either way edits are pending until the channel (re)starts and expands them into the real `config.json` keys - and until then they are kept as written, API keys included, and echoed back to every signed-in WebUI session, so enable or **Apply** promptly after typing a key - and a start failure shows its reason in the channel's **Failed** box. See [Example Configs](docs/EXAMPLE_CONFIGS.md) for various copy-pasteable setups and troubleshooting. `nanobot-voice config` prints the current section back, paste-ready (API keys withheld unless `--secrets`). Every key - default, range, per-field note - is documented inline in the schema, see [config.py](nanobot_channel_voice/config.py).

Setups vary too much for one recipe (hardware, languages, cloud vs local, interactivity), so let an agent drive: run a coding agent (e.g. Claude Code) on the target machine and paste this prompt:

```
Set up nanobot-channel-voice (https://github.com/iChizer0/nanobot-channel-voice) on this machine. Ground truth: docs/EXAMPLE_CONFIGS.md and the schema in nanobot_channel_voice/config.py; inspect first (OS/arch, RAM, sound devices via arecord -l / aplay -l, NPU, network, existing nanobot config).
Ask me one round of questions (reply languages, cloud keys or fully local, barge-in or half-duplex, wake word, latency vs quality), then pick the stack: cloud engines when keys + network allow, else on-device ([ondevice]/[rknn] extras), models via nanobot-voice sync, AEC only with a raw-PCM TTS at a rate divisible by 100.
Verify audio with a short record + replay (dsnoop/dmix if the card is shared), write channels.voice, validate with `nanobot channels status`, then a live smoke test - a robotic espeak voice means a fallback fired, fix the logged cause. Ask before installing packages or spending credits; finish with the final config and each choice's trade-off.
```

## License

See [LICENSE](LICENSE). Optional on-device model weights carry their own licenses, review them for your deployment, no weights are bundled.
