# Fish Audio S2 Pro Wyoming bridge

Streaming text-to-speech for Home Assistant, using
[Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro) with voice
cloning and free-form inline tags. The model runs in-process and speaks the
[Wyoming protocol](https://github.com/OHF-Voice/wyoming), so HA connects to
it directly.

## Quick start

```bash
./run.sh
```

The first run installs uv if needed, clones
[fish-speech](https://github.com/fishaudio/fish-speech) at a pinned commit
into `src/`, builds the venv and downloads `fishaudio/s2-pro` (~11 GB) into
`models/`. The first start also compiles the decode step with
`torch.compile` (2-3 minutes). The compile result is cached in
`.cache/inductor`, so later starts take about 70 s.

The bridge listens on `tcp://0.0.0.0:10215` and accepts connections while
the model is still loading. It is ready when the log shows
`engine ready — synthesis enabled`.

- `SETUP_ONLY=1 ./run.sh` sets everything up without starting the bridge.
- `REINSTALL=1 ./run.sh` reinstalls the Python dependencies.
- Extra arguments go to `bridge_s2.py` (`./run.sh --help` lists them).

Requires Linux with an NVIDIA GPU with about 7 GB free (int8).

## Performance

Measured on an RTX 4070 (12 GB):

| Setting          | VRAM    | RTF  |
|------------------|---------|------|
| int8 + compile   | 6.9 GB  | 0.56-0.65 |
| bf16 + compile   | 10.2 GB | 0.85 |
| bf16, no compile | 10.2 GB | 1.34 |

First audio arrives about 0.35 s after a block starts generating, and about
0.7 s after HA sends a streamed reply (with the 1 s jitter buffer). Decoding
is memory-bandwidth bound, which is why int8 weights are faster.

## Configuration

Copy `.env.example` to `.env` and edit. Everything is optional:

| Variable             | Default                  | Description                                  |
|----------------------|--------------------------|----------------------------------------------|
| `PROMPT_WAV`         | nicoleheart reference    | Voice to clone (10-30 s clean WAV); empty turns cloning off |
| `PROMPT_TEXT`        | its transcript           | Exact transcript of `PROMPT_WAV`             |
| `VOICE`              | `nicoleheart`            | Voice name shown in Home Assistant           |
| `DEFAULT_MOOD`       | (none)                   | Tag for text before any mood tag             |
| `URI`                | `tcp://0.0.0.0:10215`    | Wyoming listen address                       |
| `MODEL_ID`           | `fishaudio/s2-pro`       | HuggingFace model downloaded on first run    |
| `MODEL_DIR`          | `models/s2-pro`          | Model location                               |
| `TORCH_EXTRA`        | `cu128`                  | PyTorch build (`cu126`, `cu128`, `cu129`)    |
| `INT8`               | `1`                      | Int8 weight-only transformer layers          |
| `COMPILE`            | `1`                      | `torch.compile` the decode step              |
| `TEMPERATURE`        | `0.7`                    | Sampling temperature                         |
| `TOP_P`              | `0.7`                    | Nucleus sampling                             |
| `SEED`               | `42`                     | Seed per block (`-1` for random)             |
| `PREBUFFER_SECONDS`  | `1`                      | Audio buffered before playback starts        |
| `MIN_MERGED_SECONDS` | `3`                      | Speech merged per model call after the first sentence |
| `EXTRA_ARGS`         | (empty)                  | More `bridge_s2.py` arguments                |

## Tags

S2 Pro reads free-form natural-language tags in square brackets, e.g.
`[excited]`, `[whisper]`, `[angry]`, `[laughing]`, `[sigh]`, `[inhale]`,
`[pause]`, `[chuckle]`, `[whisper in small voice]`. Put them where the
effect should happen.

- **Sound tags** (`[laughing]`, `[sigh]`, `[chuckle]`, `[inhale]`,
  `[gasp]`, `[cough]`, `[pause]`, ...) apply where they appear.
- **Any other tag is a mood tag**. It stays in effect for the following
  sentences of the reply until the next mood tag, because the bridge sends
  long replies to the model in several blocks.
- `(laughs)`, `(sighs)` and `{tag}` are converted, and CosyVoice's
  `[laughter]` becomes `[laughing]`, so the CosyVoice HA prompt keeps working.

```
[excited] Your package just arrived! [laughing] I knew it would come today.
```

**Known limitation:** with a voice-cloning reference, S2 Pro copies the
reference's delivery and mostly ignores emotion tags. With the calm
nicoleheart clip, tagged and untagged sentences differed by about 1 dB.
Without a reference (`PROMPT_WAV=`) the tags work clearly (whisper was
12 dB quieter, excited 7 dB louder), but the voice is S2 Pro's own. The
plan is a LoRA fine-tune on the target voice, so it can run without a
reference.

## Home Assistant

1. Add the **Wyoming Protocol** integration with this host and port 10215.
2. Pick the `nicoleheart` voice as the text-to-speech engine in your voice
   assistant.

## Running as a service

`fish-wyoming.service` runs the bridge with systemd:

```bash
cp fish-wyoming.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now fish-wyoming
```

The 12 GB GPU can't hold this bridge and the CosyVoice bridge at the same
time, so enable only one of them.

## Files

- `bridge_s2.py`: the Wyoming server (sentence splitting, tag handling,
  streaming generation and codec decoding).
- `run.sh`: setup and launcher.
- `fish-wyoming.service`: systemd unit.
- `test_bridge.py`: test client for a running bridge. It covers one-shot and
  streaming requests, the duplicate requests HA sends, and blank text
  (`.venv/bin/python test_bridge.py --port 10215`).
- `scripts/smoke_s2.py`: load time, VRAM and RTF for bf16, compile or int8.
- `scripts/bench_tags.py`: renders tagged sentences next to untagged
  controls across references and temperatures.
- `scripts/test_engine.py`: the engine without Wyoming, including chunked
  vs full codec decoding.

## Implementation notes

- fish-speech's loader builds the 4.4B model in fp32 on the CPU first
  (~17.6 GB). The bridge builds it on the meta device instead and loads the
  safetensors shards directly, quantizing each layer as it moves to the GPU.
- The codec's transformers each register a 32768×32768 mask (1 GB). These
  are cut to the KV cache length before the codec moves to the GPU.
- Audio is decoded every 16 frames (~0.37 s) with 256 frames of left
  context. With less context the chunked audio drifts from a full decode
  (64 frames gave 7 dB SNR, 256 gave 39 dB).

## Credits

Text-to-speech by [Fish Audio S2 Pro](https://huggingface.co/fishaudio/s2-pro)
and [fish-speech](https://github.com/fishaudio/fish-speech). The S2 Pro
weights are under the Fish Audio Research License: free for research and
non-commercial use, commercial use needs a license from Fish Audio.

The Wyoming handler comes from the CosyVoice Wyoming bridge in
`/root/cosyvoice-wyoming`.
