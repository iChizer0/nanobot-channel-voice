# Example Configurations

Complete `~/.nanobot/config.json` files for common deployments, ready to copy and adjust, plus the setup that precedes them.

## Foundation

API keys resolve per service, config first, then environment: transcription through core's provider entry (`GROQ_API_KEY` in the examples), TTS and realtime from `tts.apiKey` / `realtime.apiKey` falling back to `OPENAI_API_KEY` - for *every* provider, so set keys explicitly when mixing vendors.

Speak, the agent transcribes, thinks, and replies through your speaker. The default is half-duplex: the mic is muted while the bot speaks, so wait for it to finish. To interrupt mid-reply, pick an open-mic mode and just start talking, note that without echo cancellation your voice has to acoustically out-compete the bot before the VAD even triggers.

Long tool calls are masked by an agent-spoken status line, with opt-in canned filler (`prologue.enabled`) as the fallback. If the configured TTS cannot be built the channel degrades to `espeak-ng` with a warning rather than going silent, a robotic voice is your cue to read the log.

## Pick audio devices

Capture and playback stay on ALSA's `default` device until the config says otherwise - fine on a desktop. On a board, or whenever anything else needs the sound card too, pick devices and share them by name. List what ALSA sees, then record three seconds and play it back, substituting your own card and device numbers:

```sh
arecord -l && aplay -l
arecord -D plughw:1,0 -f S16_LE -r 16000 -c 1 -d 3 /tmp/mic-test.wav
aplay   -D plughw:0,0 /tmp/mic-test.wav
```

The `plughw:` opens the device exclusively, which is fine for a test but locks everything else out. For real use share them with `dsnoop` (capture) and `dmix` (playback) under names of your own - redefining ALSA's stock `dsnoop`/`dmix` is a config error. A minimal `/etc/asound.conf`, with the mic on card 1 and the speaker on card 0:

```
pcm.mic     { type dsnoop; ipc_key 2048; slave { pcm "hw:1,0"; rate 48000; } }
pcm.speaker { type dmix;   ipc_key 2049; slave { pcm "hw:0,0"; rate 48000; } }
```

Retest with `-D plug:mic` and `-D plug:speaker` - exactly what the channel runs - then put those two names in the config as `captureDevice` and `playbackDevice`. The `plug:` wrapper converts rate and format in software, so 16 kHz mono capture works whatever the hardware's native rate. The examples below assume these two names, substitute `default` or your own.

## Cloud and LAN

The local pipeline against APIs: metered cloud services, or any server of yours speaking the same open protocols - identical config shape, different endpoints.

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

The same cloud engines, but you can talk over the bot: software echo cancellation (`[aec]` extra) plus raw-PCM TTS, which enables gapless playback, dynamic ducking, and the AEC reference signal in one move. The `[webrtc]` extra improves the VAD, which matters more once the mic stays open.

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

If your device or OS already cancels echo, set `"aec": "hardware"` instead and drop the `[aec]` extra; without either, `"aec": "soft"` still allows barge-in but reacts later, since your voice must out-compete the playback acoustically.

### LAN servers, no cloud

STT and TTS on machines you control: any OpenAI-compatible Whisper server for transcription and an OpenAI-compatible `/audio/speech` server (Kokoro-FastAPI here) for synthesis. Only the `[aec]` extra is needed, for the open mic. Core's `transcription` section only picks the provider and model; the endpoint and key live on a provider entry, and core resolves transcription only through its fixed provider registry (`groq`, `openai`, `siliconflow`, ... - the generic `custom` slot is chat-only), so repoint the `openai` entry at your server and give it any non-empty `apiKey`, which core requires to consider transcription configured. If you also chat through real OpenAI, borrow another OpenAI-shaped registry slot (`siliconflow`) instead of hijacking its entry. `tts.language` declares what the server's voice speaks, since the channel cannot know. `audioFormat: "pcm"` buys gapless playback, dynamic ducking, and the software-AEC reference; if the server emits PCM at a rate other than 24 kHz, set `tts.pcmSampleRate` to match or playback is pitch-shifted.

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

Every engine runs in-process (`[ondevice]` extra), with no servers or network: `.onnx` uses the configured execution provider and `.rknn` uses the RKNN runtime. The file extension selects the runtime independently for each engine. Weights are never bundled: name every file yourself or let the `nanobot-voice` CLI provision them from an index, so an engine block names one store key (`weights`) instead of a path per file. The default index is [nanobot-channel-voice-models](https://huggingface.co/iChizer0/nanobot-channel-voice-models), currently serving RKNN builds; the package itself ships no entries and redistributes no weights, only the mechanism.

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

Swap `stt.provider` to `"whisper"` (with a `stt.whisper` block) for broader language coverage at more CPU, or to `"zipformer"` for streaming decodes and mid-sentence barge-in confirmation. For noisy rooms add `"vad": { "engine": "firered", "firered": { "weights": "vad/firered/streaming/onnx" } }` — the fewest false triggers from background noise, and a false trigger is a spurious duck once the mic is open. The alternative is Silero VAD, `"vad": { "engine": "silero", "silero": { "modelPath": "model/silero_vad.onnx" } }`.

### Matcha-TTS (CPU)

Matcha-TTS (`tts.provider="matcha"`) prefers the OFFICIAL export from [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS) (MIT): export the released checkpoint once, with the HiFi-GAN vocoder embedded, and the engine needs exactly one file — the symbol table is fixed upstream, so there is no tokens file:

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

`matcha_vctk.ckpt` exports the same way (vocoder `hifigan_univ_v1`) and adds 108 voices via `speakerId`. A mel-only export (no `--vocoder-name`) instead pairs with a `vocoderPath` graph — HiFi-GAN, or sherpa's `vocos-22khz-univ.onnx` (ISTFT runs host-side). English phonemizes through espeak-ng, resolved `espeakPath` → system binary → the `[espeak]` pip extra bundling libespeak-ng + data for boards without a package manager (GPL-3, opt-in); with none of the three the channel falls back to system TTS and says so.

Timbre: `--n-timesteps` (ODE steps) is baked at export — 5 is the export default, upstream's demo runs 10, 3 audibly flattens prosody; only the small flow decoder scales with it. Upstream's CLI also denoises HiFi-GAN output but its ONNX export doesn't, so the plugin applies the same spectral denoiser host-side to separate waveform vocoders (`denoiserStrength`, default 0.00025, 0 = off) — an embedded vocoder can't be probed for its bias and keeps a faint hiss, making mel-only + `vocoderPath` the higher-fidelity route; Vocos has no such bias and needs none. `speed` (>1 = faster) and `noiseScale` (flow temperature, default 0.667; lower = cleaner but flatter, higher = breathier) tune delivery.

The icefall/sherpa-onnx [`matcha-icefall-*` releases](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html) are also consumed unmodified (front-end contract read from their metadata). For pure Chinese, `matcha-icefall-zh-baker` (training data licensed non-commercial-use-only; digits are verbalized to 汉字 automatically). Embedded English is voiced as Mandarin nativizes it — acronyms spell as the letter readings (USB → you-ai-si-bi), words transliterate through espeak IPA to the nearest pinyin syllables (hello → he-lou, python → pai-sen; letters-only without any espeak) — a loanword accent that makes "打开 WiFi" speakable, not bilingual synthesis:

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

**Bilingual in one voice (zh + en)**: [`matcha-icefall-zh-en`](https://huggingface.co/csukuangfj/matcha-icefall-zh-en) speaks real English natively — Chinese through its lexicon, English through espeak IPA (espeak is mandatory here, not a fallback tier, and the voice is fixed to en-us — the phoneme set the model was trained on) — so code-switching keeps one voice with no engine seam. 16 kHz, single speaker; the agent is told it may reply in either language, digits verbalize to 汉字. The bundled zh normalization FSTs are not used (the plugin's own verbalizer covers numbers), but its `espeak-ng-data` copy **is**: espeak's IPA spellings drift between releases (en-us FORCE moved `oː` → `ɔː`, so "more"/"four"/"course"/"supports" land on a different embedding than the model trained on), so the pack shipped beside the model files is auto-detected and phonemization runs against it. `espeakDataDir` overrides the detection; without any pack the installed espeak's data is used and the log says so:

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

**Bilingual in two voices (en + zh)**: `matcha.secondary` loads a second complete matcha engine for the other script — text routes per script run (CJK runs to the zh engine, Latin to the en one, digits/punctuation riding along), the agent is told it may reply in either language, and one Vocos session serves both when the dynamic engines name the same `vocoderPath` (measured ~+54 MB over a single engine). Each language speaks in its own checkpoint's voice, so code-switching mid-sentence switches voices — the zh-en model above stays single-voice if that matters more:

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

A static Matcha deployment uses a matched encoder/decoder/vocoder set — **not** a direct conversion of dynamic `model-steps-3.onnx`; the data-dependent duration regulator, noise, padding, and ISTFT stay in the adapter. The vocoder slot takes Vocos (host ISTFT) or a fixed-shape single-output HiFi-GAN (conv-only, NPU-friendlier, gets the spectral denoiser) — a build-time probe classifies the graph, no metadata needed. Convert and validate the artifacts together, never mixed across conversions.

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

The file extension picks the runtime per graph, so the same config with `.onnx` paths validates a split off-board before RKNN conversion (do not set `acousticModelPath`). The split cuts before the graph's final denormalization, so bucket geometry and mel statistics resolve from `encoderLen`/`melLen`/`melScale`/`melBias` in config, else an exporter `meta.json` beside the models (`{"mel_scale", "mel_bias", "encoder_len", "mel_len"}`, or named via `metaPath`), else the en_US-ljspeech values (200/800, `2.0661438`/`-5.5238085`). Output is 22,050 Hz PCM; a piece overflowing a bucket is halved and retried, so long content costs a pause, not an error.

For an icefall Chinese split (`matcha-icefall-zh-baker`) add `lexiconPath`: this selects the lexicon frontend and 汉字 digit verbalization. Its mel statistics are `"melScale": 2.7628188, "melBias": -5.9870973` — required, the LJSpeech defaults play audibly wrong audio. Each artifact set needs its own ONNX-to-RKNN parity validation, and the Baker dataset/model is non-commercial-use-only.

### RKNN runtime

The same shape with `.rknn` artifacts (`[ondevice,rknn]` extras): Whisper STT, MMS TTS, and FireRed VAD can be provisioned from an index with `nanobot-voice sync`. The CLI may prompt for confirmation while fetching models whose licenses require acceptance.

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

On a multi-core accelerator, select the index key built for that target and optionally pin engines to separate cores with `coreMask`, so a long decode does not starve VAD. If language detection on a quantized model becomes unreliable, add `"languageMinConfidence": 0.6` to the Whisper block.

#### Wiring it by hand, without an index

For converted-it-yourself models or an air-gapped installation, name every artifact directly. Explicit `*Path` fields always win over a `weights` key, and the two styles mix freely per engine, so drop these blocks in place of the ones above. `chunkLength` and `maxLength` must match the exported models. `chunkLength` is a decode window, not a cap: audio longer than it (a `vad.maxUtteranceMs`-sized utterance or a WebUI dictation through `stt.serve`) is decoded in window-sized pieces cut at the quietest gap and joined, so there is no need to re-export a longer window for long speech.

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

On-device Whisper detecting between Japanese and English, MMS-TTS speaking Japanese (`[ondevice,japanese]` extras). The `mms-tts-jpn` export brings its own `vocab.json` via the weights entry; because a supplied vocabulary carries no language label, `tts.language` declares it, and the agent is told to reply in Japanese through the runtime-context block. The `context` line is the voice-only seam for extra guidance.

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

One loaded model transcribing for everything (`[ondevice]` extra): the voice pipeline uses SenseVoice directly, and `stt.serve` exposes the same adapter as a local OpenAI-compatible endpoint that core's `transcription` consumers (WebUI dictation, chat-channel voice notes) call back into. Core reaches it through its `openai` provider entry repointed at the loopback endpoint (if you also chat through real OpenAI, borrow another OpenAI-shaped registry slot such as `siliconflow` instead), whose `apiKey` must match `stt.serve.apiKey`, since the serve endpoint checks the bearer token core sends.

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

Drop-in tuning for any local-backend example above. `vad.turn` layers Smart Turn v3, an audio-native end-of-turn model (`[ondevice]` extra, 16 kHz capture, one ~25-60 ms inference per pause), over the silence endpointer: a COMPLETE verdict closes the turn ~300 ms before `hangoverMs`, INCOMPLETE waits the silence out - the hangover becomes an upper bound rather than the decision, so raising it for hesitant speakers stops costing every turn. `hangoverMinMs` composes with it: each hangover starts there (snappy) and grows toward `hangoverMs` only on evidence a real pause was cut short. On the open-mic side, `bargeIn.mode: "pause"` halts playback outright during the confirm window instead of ducking it to `duckDb`, resuming exactly where it stopped on a false alarm — and a false alarm resolves fast: leak-triggered candidates release a few hundred ms after the pause silences them (the pause-probe), and empty streaming partials acquit early instead of waiting out the endpoint. `bargeIn.stopPhrases` (defaults cover en/zh/ja) makes a bare "stop"/"别说了"/"やめて" kill the reply *silently* — consumed, never forwarded, so the agent can't answer it — while "stop, use Tokyo instead" still publishes as a normal interruption. If a session logs many false barge-in candidates per minute, the fix is usually `aec: "webrtc"` (layers AEC3 over weak/absent hardware AEC) or `bargeIn.mode: "duck"`.

### Open mic with Smart Turn

The open-mic cloud example with the tuning applied (`[aec,webrtc,ondevice]` extras): a hangover generous enough for hesitant speakers, made affordable by the turn model; the adaptive floor; pause instead of duck while an interruption is confirmed.

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

The turn model is an on-device engine like any other: the file extension picks the runtime, so a `.rknn` conversion (`rknn-toolkit2`, same as the other models) serves it from the NPU, with `"coreMask"` pinning it to its own core on a multi-core board, and a `weights` key works once an index serves one - `nanobot-voice sync` finds it there like in every other engine block and provisions it. A missing or unfetched model degrades loudly to silence-only endpointing, never a crash.

### Wake word for public spaces

`wake.mode` gates the pipeline behind a wake phrase (local backend only). `"gate"` requires the phrase to *start* a conversation from cold; once engaged, follow-ups and barge-in stay natural for `windowS` seconds after each turn. `"strict"` additionally requires it to interrupt a live reply: while the bot speaks, non-wake speech neither ducks nor stops playback — the whole duck-then-confirm machinery stays cold until a wake hit claims the utterance, which is both the robust posture for public/multi-speaker spaces and the *simple* barge-in path (a hit is the entire verdict). Detection is two-tier and both tiers feed the same gate: the transcript prefix (`phrases`, zero models, any language the STT covers — hesitation fillers may precede it, and the phrase is stripped from the published turn) always counts, and `engine: "openwakeword"` adds an acoustic detector (openWakeWord/livekit-wakeword-format ONNX, ~3.7 MB total, one decision per 80 ms, `[ondevice]` extra, 16 kHz capture) that hears through the bot's own playback — the bot never says its own wake phrase, so an acoustic hit mid-reply is a high-precision interrupt that confirms without the min-words bar. An utterance that is *only* the phrase publishes nothing: it kills a live reply, opens the attention window, and listens.

An acoustic hit also cleans the *audio*: the detector reports where the phrase ended, and everything before that point is cut out of what STT decodes (streaming decoders restart their handle at the hit) — so a language-limited STT (a ja/de/zh model hearing an English phrase) can neither smear the phrase into the transcript nor turn a bare phrase into garbage content, and a hit whose trimmed remainder is silence resolves to attention-only. Only the transcript tier depends on STT being able to *spell* the phrase; with a mismatched language, configure the acoustic engine. In **half-duplex** (no AEC, mic muted during replies) the acoustic tier stays hot over the muted mic and a hit is the *only* barge-in: the reply stops, the mic reopens (the gate flush discards the phrase audio), and the follow-up is captured normally — wake-word barge-in with none of the open-mic duck/confirm machinery, the controlled posture for public spaces; keep the open-mic modes for personal environments where free interruption matters.

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

The acoustic engine is an on-device engine like any other: a `weights` key provisions all three files via `nanobot-voice sync`, `.rknn` conversions serve from the NPU, and a missing/unfetchable model degrades loudly to transcript-tier gating, never a crash. Notes: openWakeWord's official pretrained heads are **CC-BY-NC-SA (non-commercial)** — for commercial deployments train your own phrase head (openWakeWord's Colab or livekit-wakeword's one-command pipeline, both Apache-licensed toolchains) and point `modelPath` at it. `"strict"` with a *batch* STT and no acoustic engine detects the phrase only at the eager/endpoint decode, so interrupts confirm late — pair strict either with the acoustic engine or a streaming STT (`zipformer`). Gated utterances appear in the audio dump as `utt-<id>-gated.wav`, so a phrase that "doesn't work" is diagnosed by ear like any false barge-in.

### Debugging false barge-in by ear

When the logs show `false barge-in (empty)` / `(echo)` / `(probe)` streaks and you want to know *what the pipeline actually heard*, turn on the audio dump:

```json
{
  "channels": {
    "voice": {
      "debug": { "dumpAudio": true }
    }
  }
}
```

Every endpointed capture segment is then written under `~/.local/share/nanobot-voice/dumps/<session>/` (override with `debug.dumpDir`) as `utt-<id>-<verdict>.wav`, where `<id>` matches the `utt #N:` log line that judged it and the verdict is its outcome: `empty`, `echo`, `ack`, `blip`, `probe`, `gap`, `stop`, `gated`, `wake`, `interrupt`, `publish`. A `manifest.jsonl` in the same directory carries one record per segment (id, verdict, duration, rms, close shape, STT cost/path, VAD confidence, capture-side wall stamp; the transcript only with `logTranscripts` on), so a big dump is filtered with `jq` before anything is listened to. With `aec: "webrtc"` each segment gets a `.raw.wav` twin holding the same span *before* cancellation. Reading the pair:

- TTS clearly audible in the **post-AEC** file (`.wav`) -> the canceller is not converging (check `audio.playoutDelayMs`, give it a few seconds of clean playback to adapt, or the device is looping audio somewhere AEC3 can't model).
- TTS audible only in the **`.raw.wav`** twin, post-AEC quiet -> AEC is doing its job; the trigger is something else (VAD floor, room noise, a real voice).
- Real room sound in both -> not an echo problem at all: tune the VAD (`vad.firered.minVolume`, `bargeIn.duckStartFrames`) instead of the canceller.

Segments are recordings of the room - leave `dumpAudio` off outside debugging sessions. Disk use is capped (`debug.dumpMaxMb`, default 200): old sessions are pruned at startup and the oldest segments of a long session are deleted first.

For latency questions rather than by-ear ones, `debug.metricsIntervalS` (e.g. `30`) logs the in-process metrics snapshot - latency percentiles (`stt_ms`, `tts_synth_ms`, `ttfa_ms`, ...) and counters - as one JSON line on that cadence, so distributions are readable while you reproduce instead of only in the session-end summary.

### Filler while the agent works

Long waits (tool calls, slow reasoning) are masked in two layers. The voice context already asks the agent to speak one short status sentence before slow tools - a contextual "let me check the calendar" beats any canned phrase, and the backend detects it (`agent_prologue`) and defers the canned filler behind it. The fallback is opt-in canned filler - recommended on for tool-using deployments, since the status sentence is a preference the model may skip, and the filler is what guarantees the line never goes dead:

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

`afterMs` is a floor, not the actual delay: the first filler waits past the session's typical first-reply latency (a running average the backend keeps), so it marks *anomalously* long waits — a filler at typical latency would play "One moment." immediately followed by the answer and delay that answer behind its own audio. `phrases` is an escalation script: each wait consumes it **in order from the top**, repeating the last phrase every `intervalMs` until the reply arrives, so later entries can acknowledge a longer wait. Omit it for built-in phrases matched to the TTS engine's language (en/zh/ja/ko/de) — an on-device engine speaks exactly one language, and an unspeakable phrase synthesizes to silence (warned at warmup); an explicit list always wins, `[]` disables the script. Phrases are synthesized once with the session's own voice at warmup (local engines only; a cloud TTS is never billed at startup and pays lazily on first use) and cached. Fillers are killed by barge-in like any reply audio, never play over the user's speech, and don't count toward the latency metrics. When the agent spoke its own status line at a tool boundary, that line counts as the script's opener: the first canned filler waits a full `intervalMs` and continues from the second phrase. Keep phrases short: in half-duplex the mic is gated while one plays.

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

Other providers change `backend` and the `realtime` block only. For Qwen (Alibaba DashScope), tool calling needs the `qwen3.5` model (the `qwen3` default is persona-only), and outside mainland China you add your workspace's international `realtime.baseUrl` from the DashScope console:

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
