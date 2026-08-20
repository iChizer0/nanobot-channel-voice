# Example Configurations

Complete `~/.nanobot/config.json` files for common deployments, ready to copy and adjust, plus the setup that precedes them.

## Foundation

- API keys resolve config first, then environment: transcription through core's provider entry (`GROQ_API_KEY` in the examples), TTS and realtime from `tts.apiKey` / `realtime.apiKey` falling back to `OPENAI_API_KEY` — for *every* provider, so set keys explicitly when mixing vendors.
- The default is half-duplex: the mic is muted while the bot speaks. To interrupt mid-reply, pick an open-mic mode; without echo cancellation your voice must acoustically out-compete the playback before the VAD triggers.
- Long tool calls are masked by an agent-spoken status line, with opt-in canned filler (`prologue.enabled`) as the fallback.
- If the configured TTS cannot be built, the channel degrades to `espeak-ng` with a warning rather than going silent — a robotic voice means read the log.

## Pick audio devices

Capture and playback default to ALSA's `default` device — fine on a desktop. On a board, or when anything else needs the sound card too, name devices explicitly. List what ALSA sees, then record three seconds and play it back (substitute your card and device numbers):

```sh
arecord -l && aplay -l
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 3 /tmp/mic-test.wav
aplay   -D plughw:0,0 /tmp/mic-test.wav
```

`plughw:` opens the device exclusively — fine for a test, wrong for real use. Share instead with `dsnoop` (capture) and `dmix` (playback) under names of your own (redefining ALSA's stock `dsnoop`/`dmix` is a config error). A minimal `/etc/asound.conf`, mic on card 1, speaker on card 0:

```
pcm.mic     { type dsnoop; ipc_key 2048; slave { pcm "hw:1,0"; rate 48000; } }
pcm.speaker { type dmix;   ipc_key 2049; slave { pcm "hw:0,0"; rate 48000; } }
```

Retest with `-D plug:mic` and `-D plug:speaker` — exactly what the channel opens — then set them as `captureDevice` and `playbackDevice`. The `plug:` wrapper converts rate and format in software, so 16 kHz mono capture works at any native rate. The examples below assume these two names.

## Cloud and LAN

The local pipeline against APIs — metered cloud, or your own servers speaking the same open protocols. Same config shape, different endpoints.

### Quick start (half-duplex)

The smallest useful config: Groq Whisper STT, OpenAI TTS, energy VAD, mic muted while the bot speaks. No extras needed.

```json
{
  "transcription": {
    "enabled": true,
    "provider": "groq",
    "language": "en"
  },
  "channels": {
    "voice": {
      "enabled": true,
      "audio": {
        "captureDevice": "plug:mic",
        "playbackDevice": "plug:speaker",
        "sampleRate": 16000
      },
      "tts": {
        "provider": "openai",
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "audioFormat": "wav",
        "apiKey": "sk-..."
      }
    }
  }
}
```

### Open-mic barge-in with software AEC

The same cloud engines, but you can talk over the bot: software echo cancellation (`[aec]` extra) plus raw-PCM TTS — `audioFormat: "pcm"` enables gapless playback, dynamic ducking, and the AEC reference signal in one move. The `[webrtc]` extra improves the VAD, which matters once the mic stays open.

```json
{
  "transcription": {
    "enabled": true,
    "provider": "groq",
    "language": "en"
  },
  "channels": {
    "voice": {
      "enabled": true,
      "aec": "webrtc",
      "duckDb": -12,
      "audio": {
        "captureDevice": "plug:mic",
        "playbackDevice": "plug:speaker",
        "sampleRate": 16000
      },
      "vad": { "engine": "webrtc" },
      "tts": {
        "provider": "openai",
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "audioFormat": "pcm",
        "apiKey": "sk-..."
      }
    }
  }
}
```

If your device or OS already cancels echo, set `"aec": "hardware"` and drop the `[aec]` extra. Without either, `"aec": "soft"` still allows barge-in but reacts later.

### LAN servers, no cloud

STT and TTS on machines you control: any OpenAI-compatible Whisper server for transcription, any OpenAI-compatible `/audio/speech` server (Kokoro-FastAPI here) for synthesis. Only the `[aec]` extra is needed, for the open mic.

- Core resolves transcription only through its fixed provider registry (`groq`, `openai`, `siliconflow`, … — the generic `custom` slot is chat-only): repoint the `openai` entry at your server and give it any non-empty `apiKey`. If you also chat through real OpenAI, borrow another OpenAI-shaped slot (`siliconflow`) instead.
- `tts.language` declares what the server's voice speaks — the channel cannot know.
- If the server's PCM is not 24 kHz, set `tts.pcmSampleRate` to match or playback is pitch-shifted.

```json
{
  "transcription": {
    "enabled": true,
    "provider": "openai",
    "model": "Systran/faster-whisper-base",
    "language": "en"
  },
  "providers": {
    "openai": {
      "apiKey": "unused",
      "apiBase": "http://192.168.1.10:8000/v1"
    }
  },
  "channels": {
    "voice": {
      "enabled": true,
      "aec": "webrtc",
      "audio": {
        "captureDevice": "plug:mic",
        "playbackDevice": "plug:speaker",
        "sampleRate": 16000
      },
      "tts": {
        "provider": "openai_compat",
        "apiBase": "http://192.168.1.10:8880/v1",
        "model": "kokoro",
        "voice": "af_bella",
        "audioFormat": "pcm",
        "language": "en"
      }
    }
  }
}
```

## On-device

Every engine runs in-process (`[ondevice]` extra), no servers or network. The file extension picks the runtime per engine: `.onnx` → the configured execution provider, `.rknn` → the RKNN runtime. Weights are never bundled: name every file yourself, or let `nanobot-voice` provision them from an index so an engine block names one store key (`weights`) instead of a path per file. Default index: [nanobot-channel-voice-models](https://huggingface.co/iChizer0/nanobot-channel-voice-models) (currently RKNN builds).

```sh
export NANOBOT_VOICE_INDEX=https://example.com/voice-weights.json   # ONNX examples
nanobot-voice sync                                                  # fetch every weights key the config names, sha256-verified
```

### CPU

SenseVoice STT (fast CTC, zh/yue/en/ja/ko) and Supertonic-3 TTS (31 languages, no zh), both ONNX on CPU.

```json
{
  "channels": {
    "voice": {
      "enabled": true,
      "audio": {
        "captureDevice": "plug:mic",
        "playbackDevice": "plug:speaker",
        "sampleRate": 16000
      },
      "stt": {
        "provider": "sensevoice",
        "sensevoice": { "weights": "stt/sensevoice/small/onnx", "language": "en" }
      },
      "tts": {
        "provider": "supertonic",
        "supertonic": { "weights": "tts/supertonic-3/onnx", "language": "en" }
      }
    }
  }
}
```

Swap `stt.provider` to `"whisper"` for broader language coverage at more CPU, or `"zipformer"` for streaming decodes and mid-sentence barge-in confirmation. For noisy rooms add `"vad": { "engine": "firered", "firered": { "weights": "vad/firered/streaming/onnx" } }` — the fewest false triggers from background noise. The alternative is Silero: `"vad": { "engine": "silero", "silero": { "modelPath": "model/silero_vad.onnx" } }`.

### Matcha-TTS (CPU)

Matcha-TTS (`tts.provider="matcha"`) prefers the OFFICIAL export from [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) (MIT): export the released checkpoint once with the HiFi-GAN vocoder embedded, and the engine needs exactly one file — the symbol table is fixed upstream, so there is no tokens file:

```sh
pip install matcha-tts   # export-time only, never on the device
curl -LO https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt
curl -LO https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/generator_v1
python -m matcha.onnx.export matcha_ljspeech.ckpt matcha_ljspeech_hifigan.onnx \
  --n-timesteps 5 --vocoder-name hifigan_T2_v1 --vocoder-checkpoint-path generator_v1
```

```json
"tts": {
  "provider": "matcha",
  "matcha": { "acousticModelPath": "model/matcha_ljspeech_hifigan.onnx" }
}
```

`matcha_vctk.ckpt` exports the same way (vocoder `hifigan_univ_v1`), adding 108 voices via `speakerId`. A mel-only export (no `--vocoder-name`) pairs with a `vocoderPath` graph instead — HiFi-GAN, or sherpa's `vocos-22khz-univ.onnx` (ISTFT runs host-side). English phonemizes through espeak-ng, resolved `espeakPath` → system binary → the `[espeak]` pip extra bundling libespeak-ng + data (GPL-3, opt-in); with none of the three the channel falls back to system TTS and says so.

Tuning:

- `--n-timesteps` is baked at export: 5 is the default, 10 is upstream's demo, 3 audibly flattens prosody.
- `denoiserStrength` (default 0.00025, 0 = off): host-side denoiser for separate HiFi-GAN vocoders. An embedded vocoder keeps a faint hiss, so mel-only + `vocoderPath` is the higher-fidelity route; Vocos needs no denoising.
- `speed` (>1 = faster); `noiseScale` (default 0.667; lower = cleaner but flatter).

The icefall/sherpa-onnx [`matcha-icefall-*` releases](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html) are also consumed unmodified. For pure Chinese, `matcha-icefall-zh-baker` (training data non-commercial-use-only; digits verbalize to 汉字). Embedded English gets a loanword accent, not bilingual synthesis: acronyms spell as letter readings (USB → you-ai-si-bi), words transliterate to nearby pinyin (hello → he-lou) — enough to make "打开 WiFi" speakable:

```json
"tts": {
  "provider": "matcha",
  "matcha": {
    "acousticModelPath": "model/matcha-icefall-zh-baker/model-steps-3.onnx",
    "vocoderPath": "model/vocos-22khz-univ.onnx",
    "tokensPath": "model/matcha-icefall-zh-baker/tokens.txt",
    "lexiconPath": "model/matcha-icefall-zh-baker/lexicon.txt"
  }
}
```

Polyphone fixes go in `lexiconOverridesPath` — same `<word> <phone>...` format, entries win over the model lexicon (a misread 抽空 → `kong1` is fixed by one `抽空 chou1 kong4` line) — never by editing the model's own `lexicon.txt`.

**Bilingual in one voice (zh + en)**: [`matcha-icefall-zh-en`](https://huggingface.co/csukuangfj/matcha-icefall-zh-en) speaks real English natively — Chinese through its lexicon, English through espeak IPA (mandatory here; voice fixed to en-us, the trained phoneme set) — so code-switching keeps one voice. 16 kHz, single speaker. Keep the model's bundled `espeak-ng-data` beside the model files: espeak's IPA drifts between releases and phonemes must match training; the pack is auto-detected (`espeakDataDir` overrides, and running without one is logged). The bundled zh normalization FSTs are unused:

```json
"tts": {
  "provider": "matcha",
  "matcha": {
    "acousticModelPath": "model/matcha-icefall-zh-en/model-steps-3.onnx",
    "vocoderPath": "model/vocos-16khz-univ.onnx",
    "tokensPath": "model/matcha-icefall-zh-en/tokens.txt",
    "lexiconPath": "model/matcha-icefall-zh-en/lexicon.txt"
  }
}
```

**Bilingual in two voices (en + zh)**: `matcha.secondary` loads a second complete matcha engine for the other script — CJK runs route to the zh engine, Latin to the en one — and one Vocos session serves both when the engines name the same `vocoderPath` (~+54 MB total). Code-switching switches voices mid-sentence; the zh-en model above stays single-voice if that matters more:

```json
"tts": {
  "provider": "matcha",
  "matcha": {
    "acousticModelPath": "model/matcha-icefall-zh-baker/model-steps-3.onnx",
    "vocoderPath": "model/vocos-22khz-univ.onnx",
    "tokensPath": "model/matcha-icefall-zh-baker/tokens.txt",
    "lexiconPath": "model/matcha-icefall-zh-baker/lexicon.txt",
    "secondary": {
      "acousticModelPath": "model/matcha-icefall-en_US-ljspeech/model-steps-3.onnx",
      "vocoderPath": "model/vocos-22khz-univ.onnx",
      "tokensPath": "model/matcha-icefall-en_US-ljspeech/tokens.txt"
    }
  }
}
```

### Matcha-TTS static split (icefall, NPU-ready)

A static Matcha deployment uses a matched encoder/decoder/vocoder set — **not** a conversion of dynamic `model-steps-3.onnx`; duration regulation, noise, padding, and ISTFT stay in the adapter. The vocoder slot takes Vocos (host ISTFT) or a fixed-shape single-output HiFi-GAN (NPU-friendlier); a build-time probe classifies the graph. Convert and validate artifacts together, never mixed across conversions.

```json
"tts": {
  "provider": "matcha",
  "matcha": {
    "encoderPath": "models/matcha/encoder.rknn",
    "decoderPath": "models/matcha/decoder.rknn",
    "vocoderPath": "models/matcha/vocoder.rknn",
    "tokensPath": "models/matcha/tokens.txt"
  }
}
```

The same config with `.onnx` paths validates a split off-board before RKNN conversion (do not set `acousticModelPath`). Bucket geometry and mel statistics resolve from config (`encoderLen`/`melLen`/`melScale`/`melBias`), else an exporter `meta.json` beside the models (or `metaPath`), else the en_US-ljspeech values (200/800, `2.0661438`/`-5.5238085`). The sidecar may also declare the frontend (`"frontend": "zh_en" | "lexicon" | "espeak"`) plus `sample_rate`, `pad_id`, and `use_eos_bos` when the token-table conventions ("_" pad, ^/$ framing, 22.05 kHz) don't hold: `"zh_en"` is how a split of the bilingual zh-en model (16 kHz) becomes usable — undeclared zh-en artifacts are refused, as is a lexicon/espeak declaration beside zh-en tokens (wrong sidecar). Output is 22,050 Hz PCM; a piece overflowing a bucket is halved and retried — long content costs a pause, not an error.

For an icefall Chinese split (`matcha-icefall-zh-baker`) add `lexiconPath`: this selects the lexicon frontend and 汉字 digit verbalization. Its mel statistics are `"melScale": 2.7628188, "melBias": -5.9870973` — required, the LJSpeech defaults play audibly wrong audio. Each artifact set needs its own ONNX-to-RKNN parity validation, and the Baker dataset/model is non-commercial-use-only.

### RKNN runtime

The same shape with `.rknn` artifacts (`[ondevice,rknn]` extras): Whisper STT, MMS TTS, and FireRed VAD can be provisioned from an index with `nanobot-voice sync` (the CLI may prompt to accept model licenses).

```json
{
  "channels": {
    "voice": {
      "enabled": true,
      "audio": {
        "captureDevice": "plug:mic",
        "playbackDevice": "plug:speaker",
        "sampleRate": 16000
      },
      "vad": {
        "engine": "firered",
        "firered": { "weights": "vad/firered/streaming/rknn.rv1126b" }
      },
      "stt": {
        "provider": "whisper",
        "whisper": { "weights": "stt/whisper/base/rknn.rv1126b", "language": "en" }
      },
      "tts": {
        "provider": "mms",
        "mms": { "weights": "tts/mms/en/rknn.rv1126b" }
      }
    }
  }
}
```

On a multi-core accelerator, pick the index key built for that target and optionally pin engines to separate cores with `coreMask`, so a long decode does not starve VAD. If language detection on a quantized model is unreliable, add `"languageMinConfidence": 0.6` to the Whisper block.

#### Wiring it by hand, without an index

For converted-it-yourself models or air-gapped installs, name every artifact directly — explicit `*Path` fields win over a `weights` key, and the styles mix per engine. `chunkLength`/`maxLength` must match the exported models; `chunkLength` is a decode window, not a cap: longer audio is decoded in window-sized pieces cut at the quietest gap and joined.

```json
"vad": {
  "engine": "firered",
  "firered": {
    "modelPath": "model/fireredvad_stream_vad_with_cache.rknn",
    "cmvnPath": "model/cmvn.ark"
  }
},
"stt": {
  "provider": "whisper",
  "whisper": {
    "encoderPath": "model/whisper_encoder_base_20s.rknn",
    "decoderPath": "model/whisper_decoder_base_20s.rknn",
    "vocabPath": "model/multilingual.tiktoken",
    "melFiltersPath": "model/mel_80_filters.txt",
    "language": "en",
    "chunkLength": 20
  }
},
"tts": {
  "provider": "mms",
  "mms": {
    "encoderPath": "model/mms_tts_eng_encoder_200.rknn",
    "decoderPath": "model/mms_tts_eng_decoder_200.rknn",
    "maxLength": 200
  }
}
```

### NVIDIA Jetson (GPU via TensorRT)

The same shape as [CPU](#cpu) but accelerated on the GPU: install a JetPack-matched `onnxruntime-gpu` wheel (from the Jetson Zoo, not PyPI) and add execution providers to each engine block. The first load builds and caches TensorRT engines, so it is slow once.

```json
{
  "channels": {
    "voice": {
      "enabled": true,
      "audio": {
        "captureDevice": "plug:mic",
        "playbackDevice": "plug:speaker",
        "sampleRate": 16000
      },
      "stt": {
        "provider": "whisper",
        "whisper": {
          "weights": "stt/whisper/base/onnx",
          "language": "en",
          "executionProviders": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
          "providerOptions": [{ "trt_engine_cache_enable": true, "trt_engine_cache_path": "trt-cache" }, {}, {}]
        }
      },
      "tts": {
        "provider": "supertonic",
        "supertonic": {
          "weights": "tts/supertonic-3/onnx",
          "language": "en",
          "executionProviders": ["CUDAExecutionProvider", "CPUExecutionProvider"],
          "providerOptions": [{}, {}]
        }
      }
    }
  }
}
```

### Japanese assistant

On-device Whisper detecting between Japanese and English, MMS-TTS speaking Japanese (`[ondevice,japanese]` extras). A supplied `vocab.json` carries no language label, so `tts.language` declares it and the agent is told to reply in Japanese. The `context` line is the voice-only seam for extra guidance.

```json
{
  "channels": {
    "voice": {
      "enabled": true,
      "context": "Prefer short, conversational sentences.",
      "audio": {
        "captureDevice": "plug:mic",
        "playbackDevice": "plug:speaker",
        "sampleRate": 16000
      },
      "stt": {
        "provider": "whisper",
        "whisper": {
          "weights": "stt/whisper/base/onnx",
          "language": "ja",
          "languages": ["ja", "en"],
          "languageMinConfidence": 0.5
        }
      },
      "tts": {
        "provider": "mms",
        "language": "ja",
        "mms": { "weights": "tts/mms-jpn/onnx", "textFrontend": "japanese" }
      }
    }
  }
}
```

### STT serving core too

One loaded model transcribing for everything (`[ondevice]` extra): the voice pipeline uses SenseVoice directly, and `stt.serve` exposes the same adapter as a local OpenAI-compatible endpoint for core's `transcription` consumers (WebUI dictation, chat-channel voice notes). Repoint core's `openai` provider entry at the loopback (or borrow `siliconflow` if you also chat through real OpenAI); its `apiKey` must match `stt.serve.apiKey` — the endpoint checks the bearer token.

```json
{
  "transcription": {
    "enabled": true,
    "provider": "openai",
    "model": "sensevoice-small"
  },
  "providers": {
    "openai": {
      "apiKey": "shared-secret",
      "apiBase": "http://127.0.0.1:8035/v1"
    }
  },
  "channels": {
    "voice": {
      "enabled": true,
      "audio": {
        "captureDevice": "plug:mic",
        "playbackDevice": "plug:speaker",
        "sampleRate": 16000
      },
      "stt": {
        "provider": "sensevoice",
        "sensevoice": { "weights": "stt/sensevoice/small/onnx" },
        "serve": { "enabled": true, "port": 8035, "apiKey": "shared-secret" }
      },
      "tts": {
        "provider": "supertonic",
        "supertonic": { "weights": "tts/supertonic-3/onnx", "language": "en" }
      }
    }
  }
}
```

Note the direction: the voice channel itself keeps `stt.provider: "sensevoice"` (pointing it at `"nanobot"` while serving would be circular, and startup rejects it).

## Turn-taking and barge-in

Drop-in tuning for any local-backend example above:

- `vad.turn` layers Smart Turn v3, an audio-native end-of-turn model (`[ondevice]` extra, 16 kHz capture, ~25–60 ms per pause), over the silence endpointer: COMPLETE closes the turn early, INCOMPLETE waits the silence out. `hangoverMs` becomes an upper bound rather than the decision, so raising it for hesitant speakers stops costing every turn.
- `hangoverMinMs`: each hangover starts there (snappy) and grows toward `hangoverMs` only when a real pause was cut short.
- `bargeIn.mode: "pause"` halts playback during the confirm window instead of ducking to `duckDb`, resuming exactly where it stopped on a false alarm.
- `bargeIn.stopPhrases` (defaults cover en/zh/ja): a bare "stop"/"别说了"/"やめて" kills the reply *silently* — consumed, never forwarded — while "stop, use Tokyo instead" still publishes as a normal interruption.
- Many false barge-in candidates per minute? Usually `aec: "webrtc"` or `bargeIn.mode: "duck"`.

### Open mic with Smart Turn

The open-mic cloud example with the tuning applied (`[aec,webrtc,ondevice]` extras): a generous hangover made affordable by the turn model; pause instead of duck while an interruption is confirmed.

```json
{
  "transcription": {
    "enabled": true,
    "provider": "groq",
    "language": "en"
  },
  "channels": {
    "voice": {
      "enabled": true,
      "aec": "webrtc",
      "bargeIn": { "mode": "pause" },
      "audio": {
        "captureDevice": "plug:mic",
        "playbackDevice": "plug:speaker",
        "sampleRate": 16000
      },
      "vad": {
        "engine": "webrtc",
        "hangoverMs": 1200,
        "hangoverMinMs": 400,
        "turn": { "engine": "smartturn", "modelPath": "model/smart-turn-v3.2-cpu.onnx" }
      },
      "tts": {
        "provider": "openai",
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "audioFormat": "pcm",
        "apiKey": "sk-..."
      }
    }
  }
}
```

The turn model is an on-device engine like any other: the extension picks the runtime (a `.rknn` conversion serves from the NPU, `coreMask` pins it to its own core), a `weights` key provisions it via `nanobot-voice sync`, and a missing model degrades loudly to silence-only endpointing, never a crash.

### Wake word for public spaces

`wake.mode` gates the pipeline behind a wake phrase (local backend only). `"gate"` requires the phrase to *start* a conversation from cold; follow-ups and barge-in stay natural for `windowS` seconds after each turn. `"strict"` additionally requires it to *interrupt* a live reply — non-wake speech neither ducks nor stops playback: the posture for public/multi-speaker spaces. An utterance that is *only* the phrase publishes nothing: it kills a live reply and listens.

Detection is two-tier; both feed the same gate:

- **Transcript tier** (`phrases`, zero models): the phrase as a transcript prefix, in any language the STT can spell; stripped from the published turn.
- **Acoustic tier** (`engine: "openwakeword"`): an openWakeWord/livekit-wakeword ONNX head (~3.7 MB, one decision per 80 ms, `[ondevice]` extra) that hears through the bot's own playback, so a mid-reply hit is a high-precision interrupt. It reports where the phrase *ended* and that audio is cut from what STT decodes, so a language-limited STT cannot smear the phrase into the transcript. Required whenever the STT cannot spell the phrase.

In **half-duplex** (no AEC, mic muted during replies) the acoustic tier stays hot over the muted mic and a hit is the *only* barge-in: the reply stops, the mic reopens, the follow-up is captured. The contract is *phrase, beat, command* — the reopen discards ~100–350 ms after the phrase, so a same-breath command can lose its first word (open-mic modes capture it cleanly). A reply that literally *speaks* a wake phrase does not wake the bot through its own leak.

```json
{
  "channels": {
    "voice": {
      "enabled": true,
      "aec": "webrtc",
      "bargeIn": { "mode": "pause" },
      "wake": {
        "mode": "strict",
        "phrases": ["hey jarvis"],
        "windowS": 30,
        "engine": "openwakeword",
        "openwakeword": {
          "melPath": "model/mel.onnx",
          "embeddingPath": "model/embedding.onnx",
          "modelPath": "model/hey_jarvis_v0.1.onnx",
          "threshold": 0.5
        }
      }
    }
  }
}
```

Notes:

- openWakeWord's official pretrained heads are **CC-BY-NC-SA (non-commercial)** — for commercial use train your own phrase head (openWakeWord's Colab or livekit-wakeword's pipeline, both Apache toolchains).
- `"strict"` with a *batch* STT and no acoustic engine confirms interrupts only at the endpoint decode — pair strict with the acoustic engine or a streaming STT (`zipformer`).
- An on-device engine like any other: a `weights` key provisions all three files, `.rknn` serves from the NPU, a missing model degrades loudly to transcript-tier gating.
- Gated utterances land in the audio dump as `utt-<id>-gated.wav`, so a phrase that "doesn't work" is diagnosed by ear.

### Debugging false barge-in by ear

When the logs show `false barge-in (empty)` / `(echo)` / `(probe)` streaks and you want to know *what the pipeline actually heard*:

```json
{
  "channels": {
    "voice": {
      "debug": { "dumpAudio": true }
    }
  }
}
```

Every endpointed capture segment lands under `~/.local/share/nanobot-voice/dumps/<session>/` (override: `debug.dumpDir`) as `utt-<id>-<verdict>.wav` — `<id>` matches the `utt #N:` log line, verdict ∈ `empty`, `echo`, `ack`, `blip`, `probe`, `gap`, `stop`, `gated`, `wake`, `interrupt`, `publish`. A `manifest.jsonl` carries one record per segment (id, verdict, duration, rms, STT cost, VAD confidence, …; transcript only with `logTranscripts` on) for `jq` filtering before listening. With `aec: "webrtc"` each segment gets a `.raw.wav` twin of the same span *before* cancellation. Reading the pair:

- TTS clearly audible in the **post-AEC** file (`.wav`) -> the canceller is not converging (check `audio.playoutDelayMs`, give it a few seconds of clean playback to adapt, or the device is looping audio somewhere AEC3 can't model).
- TTS audible only in the **`.raw.wav`** twin, post-AEC quiet -> AEC is doing its job; the trigger is something else (VAD floor, room noise, a real voice).
- Real room sound in both -> not an echo problem at all: tune the VAD (`vad.firered.minVolume`, `bargeIn.duckStartFrames`) instead of the canceller.

Segments are recordings of the room — leave `dumpAudio` off outside debugging sessions. Disk use is capped (`debug.dumpMaxMb`, default 200); oldest sessions and segments prune first.

For latency questions rather than by-ear ones, `debug.metricsIntervalS` (e.g. `30`) logs the in-process metrics snapshot — latency percentiles (`stt_ms`, `tts_synth_ms`, `ttfa_ms`, …) and counters — as one JSON line on that cadence.

### Filler while the agent works

Two layers mask long waits: the voice context asks the agent to speak a short status sentence before slow tools (the canned filler defers behind it), and opt-in canned filler guarantees the line never goes dead — recommended on for tool-using deployments, since the status sentence is a preference the model may skip:

```json
{
  "channels": {
    "voice": {
      "prologue": {
        "enabled": true,
        "afterMs": 2000,
        "intervalMs": 8000,
        "phrases": ["One moment.", "Still working on it.", "This is taking a bit longer - hang on."]
      }
    }
  }
}
```

- `afterMs` is a floor, not the actual delay: the first filler waits past the session's typical first-reply latency, so it marks *anomalously* long waits only.
- `phrases` is an escalation script, consumed in order, repeating the last every `intervalMs`. Omit it for built-in phrases matched to the engine's language (en/zh/ja/ko/de); `[]` disables. An agent-spoken status line counts as the opener: the first canned filler then waits a full `intervalMs` and continues from the second phrase.
- Phrases are synthesized once at warmup with the session's own voice (local engines only; cloud TTS pays lazily) and cached; an unspeakable phrase is warned about at warmup.
- Fillers are killed by barge-in like any reply audio and don't count toward latency metrics. Keep them short: in half-duplex the mic is gated while one plays.

## Realtime

Set `backend` to `"openai"`, `"xai"`, `"azure"`, `"qwen"`, `"glm"`, or `"stepfun"` (`[realtime]` extra) and the provider replaces the whole local pipeline: turn detection + ASR + reasoning + TTS in one WebSocket session, while the plugin keeps capture/playback and routes the model's tool calls through nanobot's guarded `ToolRegistry` under `nanobot gateway`. Cloud-only: not a privacy or offline path. Do not set `audio.sampleRate`; the provider profile fixes the rates.

### Minimal

Default gated barge-in: you interrupt after the bot finishes a phrase.

```json
{
  "channels": {
    "voice": {
      "enabled": true,
      "backend": "openai",
      "audio": {
        "captureDevice": "plug:mic",
        "playbackDevice": "plug:speaker"
      },
      "realtime": {
        "model": "gpt-realtime",
        "voice": "marin",
        "apiKey": "sk-..."
      }
    }
  }
}
```

Other providers change `backend` and the `realtime` block only. For Qwen (Alibaba DashScope), tool calling needs the `qwen3.5` model (the `qwen3` default is persona-only), and outside mainland China add your workspace's international `realtime.baseUrl` from the DashScope console:

```json
"backend": "qwen",
"realtime": { "model": "qwen3.5-omni-flash-realtime", "apiKey": "sk-dashscope-..." }
```

Set the key explicitly rather than via `OPENAI_API_KEY`, which is the fallback for every provider.

### Supervisor with open-mic barge-in

The robust-tools variant: the realtime model owns the conversation but delegates every reasoning and tool step to nanobot's full agent loop (`realtime.toolMode: "supervisor"`), with the software AEC (`[aec]` extra) keeping the mic open so you can cut it off mid-reply. `delegationTimeoutS` budgets the agent's tool work; `inputTranscriptionModel` turns on user-side transcripts.

```json
{
  "channels": {
    "voice": {
      "enabled": true,
      "backend": "openai",
      "aec": "webrtc",
      "audio": {
        "captureDevice": "plug:mic",
        "playbackDevice": "plug:speaker"
      },
      "realtime": {
        "model": "gpt-realtime",
        "voice": "marin",
        "apiKey": "sk-...",
        "toolMode": "supervisor",
        "bargeIn": "aec",
        "delegationTimeoutS": 120,
        "inputTranscriptionModel": "whisper-1",
        "persona": "You are a calm, dry-witted assistant. Keep answers under three sentences unless asked to elaborate."
      }
    }
  }
}
```

With hardware or OS echo cancellation, replace `"aec": "webrtc"` with `"realtime": { "aecAvailable": true, ... }` and drop the `[aec]` extra. The persona replaces style only; never mention tools in it.

## Headless

### No audio hardware

For CI, containers, or protocol work: the `null` backend captures nothing and discards playback, while the rest of the pipeline runs normally. `logTranscripts` is acceptable here because nothing sensitive is spoken.

```json
{
  "transcription": {
    "enabled": true,
    "provider": "groq",
    "language": "en"
  },
  "channels": {
    "voice": {
      "enabled": true,
      "logTranscripts": true,
      "audio": { "backend": "null" },
      "tts": { "provider": "system" }
    }
  }
}
```

## Troubleshooting

- Channel missing from `nanobot channels status`: an editable core resolves channels through its source checkout, where no plugin manifest lands; reinstall core as a wheel (editable stays fine for this plugin).
- Startup error naming a config key: typos in `channels.voice` are startup errors by design; the message names the offending key.
- `arecord`/`aplay` errors or a busy device: another process holds the raw device; switch to the `dsnoop`/`dmix` setup above and check nothing opens `hw:` directly.
- Robotic voice: the configured TTS failed to build and the channel degraded to espeak-ng; the log names the cause.
- Bot answers but stays silent on some replies: look for "cannot voice" warnings - the agent replied in a language the on-device TTS cannot speak, and the unvoiceable text was dropped rather than played as noise. Configure a TTS for that language or constrain the agent (`tts.language`, top-level `context`).
- First reply after startup is slow: model warmup runs once in the background after start; later turns are unaffected.
- "no API key for realtime provider": set `realtime.apiKey`, or `OPENAI_API_KEY` - remembering it is the fallback for every provider.
- "has no default endpoint": you picked `azure` (or a custom deployment) without `realtime.baseUrl`.
- "cloud open-mic needs echo cancellation": `realtime.bargeIn: "aec"` needs `aec: "webrtc"`, `aec: "hardware"`, or `realtime.aecAvailable: true`; otherwise fall back to `"gated"`.
- An "install the extra" hint despite `[realtime]` being installed: check for a stale `websockets` older than 13; the extra requires `websockets>=13`.
- The bot cuts itself off every turn on a cloud backend: your hardware does not actually cancel echo; unset `realtime.aecAvailable` and use `aec: "webrtc"` or `"gated"` instead.
- Frequent `false barge-in (...)` log lines and you can't tell leak from real sound: set `debug.dumpAudio: true` and listen to the verdict-named segments (see "Debugging false barge-in by ear" above).
