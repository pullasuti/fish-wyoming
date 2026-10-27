"""Exercise S2Engine without Wyoming: load, first-chunk latency, RTF, and
whether the chunked (streaming) codec decode matches a full decode.

  .venv/bin/python scripts/test_engine.py [--no-int8] [--no-compile] [--out out/engine]
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "fish-speech"))

import logging

import numpy as np
import soundfile as sf
import torch

import bridge_s2

TEXTS = [
    "The kitchen lights are off and the front door is locked.",
    "[excited] Your package just arrived! [laughing] I knew it would come today.",
    "Okay.",
]
REF_TEXT = (
    "The day had gently ended. In the soft room, the person lay down. Slowly, "
    "the light faded over the floor. Eyes drifted shut and the body relaxed. "
    "Thoughts slowed and turned soft. Warmth spread through the sheets and the "
    "body let go, drifting into sleep."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-int8", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--no-ref", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "out" / "engine"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    eng = bridge_s2.S2Engine(
        model_dir=str(ROOT / "models" / "s2-pro"),
        prompt_wav=None if args.no_ref else "/root/cosyvoice-wyoming/voices/nicoleheart.wav",
        prompt_text=REF_TEXT,
        int8=not args.no_int8,
        compile=not args.no_compile,
        max_seq_len=4096,
        temperature=0.7, top_p=0.7, top_k=30, seed=42,
        first_frames=8, chunk_frames=16, context_frames=256,
    )
    t = time.monotonic()
    eng.load()
    print(f"[load+warmup] {time.monotonic() - t:.1f}s, peak "
          f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB", flush=True)

    for i, text in enumerate(TEXTS):
        t = time.monotonic()
        chunks, first = [], None
        for pcm, sr in eng.synthesize_iter(text):
            first = first if first is not None else time.monotonic() - t
            chunks.append(np.frombuffer(pcm, dtype=np.int16))
        total = time.monotonic() - t
        audio = np.concatenate(chunks)
        dur = len(audio) / sr
        print(f"[block {i}] first chunk {first:.2f}s, {len(chunks)} chunks, "
              f"audio {dur:.2f}s in {total:.2f}s (RTF {total / dur:.2f})", flush=True)
        sf.write(out / f"block{i}.wav", audio, sr)

    # Chunked decode vs one full decode of the same frames
    with torch.inference_mode():
        torch.manual_seed(42)
        frames = list(eng._frames(TEXTS[0]))
        full = eng._decode_audio(frames, 0, len(frames))
        parts, done = [], 0
        while done < len(frames):
            n = min(len(frames), done + (8 if done == 0 else 16))
            parts.append(eng._decode_audio(frames, done, n))
            done = n
    a = np.frombuffer(full, dtype=np.int16).astype(np.float64)
    b = np.frombuffer(b"".join(parts), dtype=np.int16).astype(np.float64)
    print(f"[decode] frames {len(frames)}, samples full {len(a)} chunked {len(b)}, "
          f"hop {eng.hop}", flush=True)
    n = min(len(a), len(b))
    err = np.abs(a[:n] - b[:n])
    snr = 10 * np.log10((a[:n] ** 2).sum() / max(((a[:n] - b[:n]) ** 2).sum(), 1e-9))
    print(f"[decode] max abs diff {err.max():.0f} / 32767, SNR {snr:.1f} dB", flush=True)


if __name__ == "__main__":
    main()
