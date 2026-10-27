"""S2 Pro smoke test: load time, VRAM, RTF per sentence, tag rendering.

Run from the repo root:
  .venv/bin/python scripts/smoke_s2.py --out out/smoke [--compile]
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "fish-speech"))

import soundfile as sf
import torch

from fish_speech.models.text2semantic.inference import (
    decode_one_token_ar,
    decode_to_audio,
    encode_audio,
    generate_long,
    load_codec_model,
)
from fish_speech.models.text2semantic.llama import DualARTransformer

SENTENCES = [
    ("plain", "The kitchen lights are off and the front door is locked."),
    ("laugh", "Oh, that is a good one [laughing] I did not see it coming."),
    ("sigh", "[sigh] Fine, I will turn the heating down again."),
    ("whisper", "[whisper] Everyone is asleep, so I will keep this quiet."),
    ("excited", "[excited] Your package just arrived at the front door!"),
]


def gb(x):
    return f"{x / 2**30:.2f} GB"


def mem(tag):
    torch.cuda.synchronize()
    print(
        f"[mem] {tag}: alloc {gb(torch.cuda.memory_allocated())}, "
        f"peak {gb(torch.cuda.max_memory_allocated())}",
        flush=True,
    )


def shrink_codec_masks(codec, size):
    """Codec Transformers register a 32768x32768 bool causal mask each (1 GB)."""
    n = 0
    for m in codec.modules():
        mask = getattr(m, "causal_mask", None)
        if isinstance(mask, torch.Tensor) and mask.size(-1) > size:
            m.causal_mask = mask[:size, :size].clone()
            n += 1
    torch.cuda.empty_cache()
    return n


def load_llm(path, dev, max_seq_len):
    """Build on the meta device and load shards straight to the GPU.

    from_pretrained() first builds the 4.4B model in fp32 on the CPU (~17.6 GB),
    which does not fit in 15 GB of RAM.
    """
    import json

    from safetensors.torch import load_file

    from fish_speech.models.text2semantic.llama import (
        _remap_fish_qwen3_omni_keys,
        precompute_freqs_cis,
    )

    with torch.device("meta"):
        model = DualARTransformer.from_pretrained(
            path, load_weights=False, max_length=max_seq_len
        )
    index = json.loads((Path(path) / "model.safetensors.index.json").read_text())
    for shard in sorted(set(index["weight_map"].values())):
        weights = _remap_fish_qwen3_omni_keys(load_file(str(Path(path) / shard), device=dev))
        model.load_state_dict(weights, strict=False, assign=True)
        del weights
    left = [n for n, p in model.named_parameters() if p.is_meta]
    if left:
        raise RuntimeError(f"weights missing from checkpoint: {left[:5]} ({len(left)})")

    c = model.config
    with torch.device(dev):
        model.freqs_cis = precompute_freqs_cis(c.max_seq_len, c.head_dim, c.rope_base)
        model.causal_mask = torch.tril(
            torch.ones(c.max_seq_len, c.max_seq_len, dtype=torch.bool)
        )
        model.fast_freqs_cis = precompute_freqs_cis(
            c.num_codebooks, c.fast_head_dim, c.rope_base
        )
    left = [n for n, b in model.named_buffers() if b.is_meta]
    if left:
        raise RuntimeError(f"buffers left on meta: {left}")
    model = model.to(dtype=torch.bfloat16).eval()
    model._cache_setup_done = False
    return model


class Int8Linear(torch.nn.Module):
    """Weight-only int8 with a per-output-channel scale (gpt-fast style)."""

    def __init__(self, lin):
        super().__init__()
        w = lin.weight.detach().float()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127
        self.register_buffer("weight", torch.round(w / scale[:, None]).to(torch.int8))
        self.register_buffer("scales", scale.to(lin.weight.dtype))
        self.bias = lin.bias

    def forward(self, x):
        y = torch.nn.functional.linear(x, self.weight.to(x.dtype)) * self.scales
        return y if self.bias is None else y + self.bias


def quantize_int8(model):
    """Swap every nn.Linear inside the slow and fast transformer blocks."""
    n = 0
    for blocks in (model.layers, model.fast_layers):
        for block in blocks:
            for name, mod in list(block.named_modules()):
                for cname, child in list(mod.named_children()):
                    if isinstance(child, torch.nn.Linear):
                        setattr(mod, cname, Int8Linear(child))
                        n += 1
            torch.cuda.empty_cache()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "s2-pro"))
    ap.add_argument("--ref-wav", default="/root/cosyvoice-wyoming/voices/nicoleheart.wav")
    ap.add_argument(
        "--ref-text",
        default=(
            "The day had gently ended. In the soft room, the person lay down. "
            "Slowly, the light faded over the floor. Eyes drifted shut and the "
            "body relaxed. Thoughts slowed and turned soft. Warmth spread through "
            "the sheets and the body let go, drifting into sleep."
        ),
    )
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--codec-mask", type=int, default=4096)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "out" / "smoke"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    t0 = time.perf_counter()
    model = load_llm(args.model, dev, args.max_seq_len)
    print(f"[load] llm {time.perf_counter() - t0:.1f}s", flush=True)
    mem("llm")
    if args.int8:
        n = quantize_int8(model)
        torch.cuda.reset_peak_memory_stats()
        mem(f"llm int8 ({n} linears)")

    t1 = time.perf_counter()
    codec = load_codec_model(Path(args.model) / "codec.pth", "cpu", torch.bfloat16)
    n = shrink_codec_masks(codec, args.codec_mask)
    codec = codec.to(dev)
    print(f"[load] codec {time.perf_counter() - t1:.1f}s ({n} masks shrunk)", flush=True)
    mem("llm+codec")

    decode_one_token = decode_one_token_ar
    if args.compile:
        decode_one_token = torch.compile(
            decode_one_token_ar, backend="inductor", fullgraph=True, dynamic=True
        )

    t2 = time.perf_counter()
    ref_codes = encode_audio(args.ref_wav, codec, dev)
    print(
        f"[ref] {ref_codes.shape} codes in {time.perf_counter() - t2:.2f}s", flush=True
    )
    sr = codec.sample_rate

    def synth(text):
        torch.manual_seed(args.seed)
        torch.cuda.synchronize()
        ts = time.perf_counter()
        codes = []
        for r in generate_long(
            model=model,
            device=dev,
            decode_one_token=decode_one_token,
            text=text,
            max_new_tokens=1024,
            prompt_text=[args.ref_text],
            prompt_tokens=[ref_codes],
            compile=args.compile,
        ):
            if r.action == "sample":
                codes.append(r.codes)
        torch.cuda.synchronize()
        tg = time.perf_counter() - ts
        audio = decode_to_audio(torch.cat(codes, 1).to(dev), codec).float().cpu().numpy()
        torch.cuda.synchronize()
        tt = time.perf_counter() - ts
        return audio, tg, tt

    print("[warmup] ...", flush=True)
    _, tg, tt = synth("Hello there, this is a warm-up sentence.")
    print(f"[warmup] {tt:.1f}s", flush=True)
    mem("after warmup")

    for name, text in SENTENCES:
        audio, tg, tt = synth(text)
        dur = len(audio) / sr
        print(
            f"[rtf] {name:8s} audio {dur:5.2f}s gen {tg:5.2f}s total {tt:5.2f}s "
            f"RTF {tt / dur:.2f}  frames/s {dur * sr / 2048 / tg:.1f}",
            flush=True,
        )
        sf.write(out / f"{name}.wav", audio, sr)
    mem("end")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
