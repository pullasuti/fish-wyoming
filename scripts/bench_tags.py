"""Render tagged sentences and untagged controls across references and temperatures.

  .venv/bin/python scripts/bench_tags.py --out out/tags

Writes <out>/<ref>_t<temp>/<name>.wav. Compare each tagged clip with its
untagged control (same text) using the skill's compare_wavs.py, then listen.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import soundfile as sf
import torch
from smoke_s2 import ROOT, load_llm, quantize_int8, shrink_codec_masks

from fish_speech.models.text2semantic.inference import (
    decode_one_token_ar,
    decode_to_audio,
    encode_audio,
    generate_long,
    load_codec_model,
)

LONG_TEXT = (
    "The day had gently ended. In the soft room, the person lay down. Slowly, "
    "the light faded over the floor. Eyes drifted shut and the body relaxed. "
    "Thoughts slowed and turned soft. Warmth spread through the sheets and the "
    "body let go, drifting into sleep."
)
SHORT_TEXT = (
    "The day had gently ended. In the soft room, the person lay down. Slowly, "
    "the light faded over the floor. Eyes drifted shut"
)
HEART = "/root/cosyvoice-wyoming/voices/nicoleheart.wav"
# name: (wav, reference transcript, prefix for the target text)
REFS = {
    "noref": (None, None, ""),
    "heart": (HEART, LONG_TEXT, ""),
    "nicole": ("/root/ming3/voices/nicole.wav", SHORT_TEXT, ""),
    "heart_spk": (HEART, LONG_TEXT, "<|speaker:0|>"),
    "heart_desc": (HEART, "[soft sleepy narration] " + LONG_TEXT, ""),
    "heart_both": (HEART, "[soft sleepy narration] " + LONG_TEXT, "<|speaker:0|>"),
}

WHISPER = "Everyone is asleep, so I will keep this quiet."
EXCITED = "Your package just arrived at the front door!"
ANGRY = "I told you three times to close the window."
SENTENCES = [
    ("whisper_ctrl", WHISPER),
    ("whisper", f"[whisper] {WHISPER}"),
    ("whisper_small", f"[whisper in small voice] {WHISPER}"),
    ("excited_ctrl", EXCITED),
    ("excited", f"[excited] {EXCITED}"),
    ("excited_tone", f"[excited tone] {EXCITED}"),
    ("angry_ctrl", ANGRY),
    ("angry", f"[angry] {ANGRY}"),
    ("laughing", "That is hilarious [laughing] I really did not see that coming."),
    ("prolonged_laugh", "That is hilarious [prolonged laugh] I really did not see that coming."),
    ("chuckle", "That is hilarious [chuckle] I really did not see that coming."),
    ("sigh", "[sigh] Fine, I will turn the heating down again."),
    ("inhale", "[inhale] Okay, here is the plan for tomorrow."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "s2-pro"))
    ap.add_argument("--refs", default="noref,heart,nicole")
    ap.add_argument("--temps", default="0.7,0.9")
    ap.add_argument("--top-p", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bf16", action="store_true", help="skip int8")
    ap.add_argument("--out", default=str(ROOT / "out" / "tags"))
    args = ap.parse_args()
    dev = "cuda"

    model = load_llm(args.model, dev, 4096)
    if not args.bf16:
        quantize_int8(model)
    codec = load_codec_model(Path(args.model) / "codec.pth", "cpu", torch.bfloat16)
    shrink_codec_masks(codec, 4096)
    codec = codec.to(dev)
    decode = torch.compile(
        decode_one_token_ar, backend="inductor", fullgraph=True, dynamic=True
    )
    sr = codec.sample_rate

    def synth(text, ref, temp):
        wav, ref_text, prefix = ref
        text = prefix + text
        kw = {}
        if wav:
            kw = dict(prompt_text=[ref_text], prompt_tokens=[ref_codes[wav]])
        torch.manual_seed(args.seed)
        codes = [
            r.codes
            for r in generate_long(
                model=model, device=dev, decode_one_token=decode, text=text,
                max_new_tokens=1024, temperature=temp, top_p=args.top_p,
                compile=True, **kw,
            )
            if r.action == "sample"
        ]
        return decode_to_audio(torch.cat(codes, 1).to(dev), codec).float().cpu().numpy()

    refs = {k: REFS[k] for k in args.refs.split(",")}
    ref_codes = {w: encode_audio(w, codec, dev) for w, _, _ in refs.values() if w}

    t = time.perf_counter()
    synth("Hello there, this is a warm-up sentence.", next(iter(refs.values())), 0.7)
    print(f"[warmup] {time.perf_counter() - t:.0f}s", flush=True)

    for rname, ref in refs.items():
        for temp in map(float, args.temps.split(",")):
            d = Path(args.out) / f"{rname}_t{temp}"
            d.mkdir(parents=True, exist_ok=True)
            for name, text in SENTENCES:
                audio = synth(text, ref, temp)
                sf.write(d / f"{name}.wav", audio, sr)
                print(f"[done] {d.name}/{name} {len(audio) / sr:.2f}s", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
