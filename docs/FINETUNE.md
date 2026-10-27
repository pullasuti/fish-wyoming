# Handoff: LoRA fine-tune of S2 Pro on the nicoleheart voice

Written 2026-10-27 on the RTX 4070 box (`/root/fish`). The fine-tune runs on
the separate 24 GB machine; only the finished merged model comes back here.

## Goal

A merged S2 Pro checkpoint that speaks in the nicoleheart voice **without a
reference clip**, with S2 Pro's inline tags (`[excited]`, `[angry]`,
`[laughing]`, `[sigh]`, ...) still working. The bridge here (`bridge_s2.py`)
then runs it with `PROMPT_WAV=` (no cloning).

## Why

Measured with `scripts/bench_tags.py` (int8 + compile, same text tagged vs
untagged, results in `out/tags/` on the 4070 box):

| Setup | `[whisper]` vs control | `[excited]` vs control |
|---|---|---|
| No reference | 12 dB quieter, 50 % longer | 7 dB louder |
| nicoleheart reference (18 s) | within 1 dB | within 1 dB |
| nicole reference (8 s) | within 1 dB | within 1 dB |

Raising the temperature from 0.7 to 0.9 didn't help. Neither did adding `<|speaker:0|>` to the target text, or
describing the reference's style in its transcript. The user heard only
a slight sigh. S2 copies the reference's delivery, and the reference is
calm sleep narration. So the voice has to go into the weights.

## Code

- fish-speech **`bdd39477922a`** (the parent of `1ff3c3e`, the Drama 3 update that
  deleted all training code). Use this commit for training only. The bridge
  stays on `541566932abb`.
- Relevant files at that commit: `fish_speech/train.py`,
  `fish_speech/configs/text2semantic_finetune.yaml`,
  `fish_speech/configs/lora/{r_8_alpha_16,r_32_alpha_16_fast}.yaml`,
  `fish_speech/models/text2semantic/lora.py`, `fish_speech/datasets/semantic.py`,
  `tools/vqgan/extract_vq.py`, `tools/llama/build_dataset.py`,
  `tools/llama/merge_lora.py`, `docs/en/finetune.md`.
- The model loader there already handles S2 Pro: the `fish_qwen3_omni`
  config and sharded safetensors. LoRA can target the slow and the fast
  transformer (`target_modules`).

## Must fix before training

1. **Prompt format (the important one).** `datasets/semantic.py` was last
   changed for S1 (Oct 2025) and builds S1-style samples: system
   `"Speak out the provided text."`, user `"<|speaker:user|> ..."`, assistant
   `"<|speaker:assistant|> <|voice|>" + codes`. S2 Pro inference (and the
   bridge without a reference) uses:
   - system: `"convert the provided text to speech"`
   - user: the text, tags inline
   - assistant: `modality="voice"`, then the VQ codes
   
   See `generate_long()` in `fish_speech/models/text2semantic/inference.py`
   at `541566932abb`, and `S2Engine._build_prompt_base/_encode` in
   `bridge_s2.py`. Rewrite `pack_sentences` to build exactly that
   `Conversation`, with loss only on the VQ part. Training on the S1 format
   would teach a prompt the bridge never sends.
2. **Config leftovers from S1.** `text2semantic_finetune.yaml` points at
   `checkpoints/openaudio-s1-mini` and `tokenizer.tiktoken`. S2 Pro ships
   `tokenizer.json`, so check how `FishTokenizer.from_pretrained` is called.
   The docs' download commands are for S1 too.
3. **Memory on 24 GB.** I haven't measured this. The bf16 base is about 8.6 GB.
   Expect batch size 1-2 with gradient accumulation and checkpointing, and
   measure first.

## Data: reuse the VoxCPM2 pipeline

Source: `/mnt/Kakkonen/voxcpm_lora_pipeline` (on the 24 GB box). It has 203
CosyVoice3 takes in the nicoleheart voice, 13.6 min, 25-26 lines per emotion,
already trimmed, emotion-relative loudness, WER- and similarity-filtered.
Recipe details are in the `voxcpm2-voice-lora` skill.

- **Layout** for `extract_vq.py` / `build_dataset.py`: `data/nicoleheart/<id>.wav`
  plus `<id>.lab` (transcript).
- **Do not run `fap loudness-norm`** (recommended in the upstream doc). It
  flattens the per-emotion loudness the pipeline set on purpose.
- **Tags:** rewrite VoxCPM2's `(tag)Text` prefixes as S2 inline tags at the
  start of the line:

  | VoxCPM2 tag | S2 tag |
  |---|---|
  | (none, neutral) | (none) |
  | `(happy, cheerful tone)` | `[happy]` |
  | `(sad, quiet and slow)` | `[sad]` |
  | `(angry, sharp and forceful)` | `[angry]` |
  | `(surprised, astonished)` | `[surprised]` |
  | `(scared, trembling and anxious)` | `[scared]` |
  | `(warm, gentle and calm)` | `[warm]` |
  | `(soft, gentle whisper)` | **drop these takes** |

  Drop whisper because CosyVoice can't whisper. Training `[whisper]` on its soft
  voiced takes would overwrite S2's real whisper, which works without a
  reference.
- **Inline sounds (new lines):** CosyVoice3's `[laughter]` and `[sigh]` are
  audible. `[breath]` is untrained in CosyVoice3 and does nothing. Generate
  about 20-30 lines with a laugh or sigh mid-sentence, and label them with S2's
  `[laughing]` / `[sigh]` at the same position. Otherwise the fine-tune may
  unlearn S2's own sounds.
- Keep the pipeline's held-out eval sentences out of training.

## Training

- Start from the upstream LoRA configs (r=8 alpha=16 on everything, or r=32
  on the fast transformer only). The voice lives in the slow AR as well, so
  prefer targeting both. lr 1e-4, warmup 10 steps, `bf16-true`.
- Count steps, not epochs (lesson from the VoxCPM2 LoRA: 3 epochs learned
  nothing). Plan for roughly 400-1000 steps and save a checkpoint every
  50-100 steps (`val_check_interval`, `every_n_train_steps`).
- Upstream warnings (`docs/en/finetune.md`): fine-tuning an RL-trained model
  can shift its distribution and degrade it, and short training learns
  "speech patterns, not timbre". Both mean: pick the checkpoint by eval, and
  prefer the earliest one that sounds like her.

## Evaluate each checkpoint (no reference)

Merge each candidate (see below), then render with `PROMPT_WAV` unset:

- **Voice:** resemblyzer similarity to nicoleheart, as in the pipeline.
  Without a reference, S2's base voice is random, so this must rise clearly.
- **Tags:** `scripts/bench_tags.py --refs noref --model <merged dir>` renders
  tagged sentences next to untagged controls. Compare with
  `compare_wavs.py` from the `wyoming-tts-bridge` skill. Tags must keep the
  no-reference effect sizes from the table above, and `[whisper]` especially
  must stay a real whisper.
- **Intelligibility:** Whisper WER on held-out sentences, 3 seeds each, and
  short replies ("Okay!", "Done.").
- **Listen** before picking. Numbers screen; the ear decides.

## Export and bring back

- `tools/llama/merge_lora.py` writes a single **`model.pth`** via
  `save_pretrained`. The bridge's loader (`S2Engine._load_llm`) currently reads
  only `model.safetensors.index.json` plus shards with the original HF key
  names, which it remaps itself. Either convert the merged model to the
  original sharded safetensors layout (reverse the key remap), or teach
  `_load_llm` to read `model.pth` (keys already internal, no remap). Check
  which keys `save_pretrained` drops (e.g. `audio_*`).
- The merged dir must also contain `config.json`, `tokenizer.json`,
  `tokenizer_config.json`, `special_tokens_map.json`, `chat_template.jinja`,
  and `codec.pth` (copy these from `fishaudio/s2-pro`; the codec isn't fine-tuned).
- On the 4070: copy to `/root/fish/models/s2-pro-nicoleheart`, then in
  `/root/fish/.env` set `MODEL_DIR=models/s2-pro-nicoleheart` and `PROMPT_WAV=`.
  Then `systemctl restart fish-wyoming` and
  `.venv/bin/python test_bridge.py --port 10215`. The int8 + compile path
  should give the same speed (RTF about 0.6, 6.9 GB).

## Facts from the 4070 box

- S2 Pro: 4B slow AR (36 layers) + 400M fast AR (4 layers, 10 codebooks),
  codec 44.1 kHz at 2048 samples per frame (21.5 frames/s).
- Decoding is memory-bandwidth bound: bf16 + compile RTF 0.85, int8 + compile
  RTF 0.56.
- License: Fish Audio Research License, free for non-commercial use.
