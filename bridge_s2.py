#!/usr/bin/env python3
"""Fish Audio S2 Pro Wyoming TTS bridge.

Loads S2 Pro in-process (fish-speech code, no HTTP server) and streams audio
to Home Assistant while it is generated:

  1. STREAMING — frames are decoded to audio every few frames by the causal
     codec (with left context), instead of after the whole block.
  2. SENTENCE MERGING — sentences batched to --min-merged-seconds of
     estimated speech; the first sentence goes alone for fast first audio.
  3. PRE-BUFFERING (jitter buffer) — AudioStart deferred until
     --prebuffer-seconds is queued; afterwards every chunk is written
     immediately.
  4. UNIFORM RE-CHUNKING — PCM is re-emitted in fixed-size AudioChunks.
  5. WYOMING ERROR EVENTS — a failed block is reported to the client;
     the remaining blocks of the utterance still run.
  6. ONE-SHOT REQUESTS — a plain synthesize (tts.speak, "Try voice") is
     answered with audio-start/chunks/audio-stop. The copy HA sends
     alongside a stream, or right after it, is ignored.
  7. TAGS — S2 Pro reads free-form [tags] inline, so they pass through.
     A mood tag stays in effect for the following blocks until the next
     one; (laughs)-style forms are converted to [laughing].
  8. MEMORY — the model is built on the meta device and each weight shard
     goes straight to the GPU (optionally int8 weight-only), so neither
     host RAM nor VRAM ever holds a full fp32/bf16 copy.
  9. WARMUP — torch.compile and a throwaway synthesis run before the
     ready event, so the first real request is as fast as later ones.

Startup order: bind the port immediately, load the model in the background,
queue any sentences that arrive during the load.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler, AsyncServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

_LOGGER = logging.getLogger(__name__)

# Rough speech-rate estimate for merging decisions: ~2.8 words/second.
WORDS_PER_SECOND = 2.8

# A one-shot synthesize arriving this soon after synthesize-stop, with
# matching text, is a late duplicate of the stream just played, not a request.
LATE_DUPLICATE_WINDOW_S = 3.0

# Synthesized at startup so the first real request doesn't pay for
# torch.compile and CUDA warmup.
WARMUP_TEXTS = ("Hello there.", "[excited] The lights in the kitchen are on.")

# ---------------------------------------------------------------------------
# Tag handling — S2 Pro takes free-form [tags] inline. Sound tags mark an
# event at their position; any other tag is a mood/delivery tag and stays in
# effect for the following blocks of the reply.
# ---------------------------------------------------------------------------

SOUND_TAGS = {
    "laughing", "chuckle", "chuckling", "giggle", "sigh", "inhale", "exhale",
    "gasp", "cough", "tsk", "pause", "short pause", "breath", "sniff",
    "clearing throat", "panting",
}

# S2 Pro also takes open-ended tags ("[laughs softly]", "[small nervous
# laugh]"). Any tag containing one of these words is a sound, so it never
# becomes the mood carried into later blocks.
SOUND_WORDS_RE = re.compile(
    r"\b(laugh|chuckl|giggl|snicker|sigh|inhal|exhal|gasp|cough|breath|sniff|"
    r"throat|pause|tsk|pant|yawn|snort|hmm|gulp|whistl)", re.I
)


def is_sound(tag: str) -> bool:
    return tag in SOUND_TAGS or bool(SOUND_WORDS_RE.search(tag))


# Other spellings (incl. the CosyVoice bridge's tags) -> S2 Pro tag
ALIASES = {
    "laughter": "laughing",
    "laughs": "laughing",
    "laugh": "laughing",
    "chuckles": "chuckle",
    "giggles": "giggle",
    "sighs": "sigh",
    "gasps": "gasp",
    "coughs": "cough",
    "breaths": "breath",
    "inhales": "inhale",
    "exhales": "exhale",
    "sniffs": "sniff",
}

TAG_RE = re.compile(r"[\[\{]([^\]\}\[\{]+)[\]\}]")
PAREN_RE = re.compile(r"\(\s*([a-zA-Z]+(?: [a-zA-Z]+)?)\s*\)")
DANGLING_RE = re.compile(r"[\[\{][^\]\}]*$")
TAG_RUN_RE = re.compile(r"(?:\[[^\]\[]+\]\s*)+")

_TERM_RE = re.compile(r"[.!?…][\"'”’)\]]*\s+")
_WORD_RE = re.compile(r"[A-Za-z.]+$")

_ABBREVIATIONS = {
    "e.g", "i.e", "eg", "ie", "etc", "mr", "mrs", "ms", "dr", "prof",
    "sr", "jr", "vs", "st",
}


def canonical_tag(tag: str) -> str:
    key = " ".join(tag.lower().split())
    return ALIASES.get(key, key)


def normalize(text: str) -> str:
    """{tag} -> [tag]; (laughs) -> [laughing]; other parentheses stay text."""
    def paren(m: re.Match) -> str:
        key = canonical_tag(m.group(1))
        return f"[{key}]" if is_sound(key) else m.group(0)

    text = PAREN_RE.sub(paren, text)
    text = TAG_RE.sub(lambda m: f"[{canonical_tag(m.group(1))}]", text)
    return DANGLING_RE.sub("", text)


def prepare_block(text: str, mood: Optional[str]) -> tuple[str, Optional[str]]:
    """Return (text for the model, mood in effect after this block).

    The mood is a group of consecutive mood tags, e.g. "[warm] [whisper]";
    the last such group in a block becomes the mood for the next blocks. A
    block that does not open with a mood tag gets the current mood
    prepended, so merged blocks keep the reply's delivery.
    """
    text = " ".join(normalize(text).split())
    groups = []
    for run in TAG_RUN_RE.finditer(text):
        tags = [t for t in TAG_RE.findall(run.group(0)) if not is_sound(t)]
        if tags:
            groups.append((run.start(), " ".join(f"[{t}]" for t in tags)))
    if mood and not (groups and groups[0][0] == 0):
        text = f"{mood} {text}"
    return text, (groups[-1][1] if groups else mood)


def spoken_words(text: str) -> str:
    return " ".join(TAG_RE.sub(" ", text).split())


def estimate_seconds(text: str) -> float:
    """Very rough spoken-duration estimate for merging decisions."""
    return len(spoken_words(text).split()) / WORDS_PER_SECOND


def norm_text(text: str) -> str:
    """Tag-free, whitespace- and case-insensitive form for duplicate checks."""
    return spoken_words(text).lower()


# ---------------------------------------------------------------------------
# Model wrapper — the ONLY place that talks to fish-speech
# ---------------------------------------------------------------------------

class Int8Linear(torch.nn.Module):
    """Weight-only int8 with a per-output-channel scale (gpt-fast style).

    Decoding is memory-bandwidth bound, so halving the weight bytes nearly
    doubles the frame rate (RTX 4070: RTF 0.85 -> 0.56 with torch.compile).
    """

    def __init__(self, lin: torch.nn.Linear) -> None:
        super().__init__()
        w = lin.weight.detach().float()
        scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127
        self.register_buffer("weight", torch.round(w / scale[:, None]).to(torch.int8))
        self.register_buffer("scales", scale.to(lin.weight.dtype))
        self.bias = lin.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.linear(x, self.weight.to(x.dtype)) * self.scales
        return y if self.bias is None else y + self.bias


class S2Engine:
    """Loads S2 Pro once and exposes a blocking, streaming synthesize call."""

    def __init__(
        self,
        model_dir: str,
        prompt_wav: Optional[str],
        prompt_text: str,
        int8: bool,
        compile: bool,
        max_seq_len: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int,
        first_frames: int,
        chunk_frames: int,
        context_frames: int,
        device: str = "cuda",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.prompt_wav = prompt_wav
        self.prompt_text = prompt_text
        self.int8 = int8
        self.compile = compile
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.first_frames = first_frames
        self.chunk_frames = chunk_frames
        self.context_frames = context_frames
        self.device = device

        self.model: Any = None
        self.codec: Any = None
        self.sample_rate: int = 44100
        self.hop: int = 2048
        self._lock = threading.Lock()

    # -- startup ------------------------------------------------------------

    def _load_llm(self) -> None:
        """Build on the meta device and load each shard straight to the GPU.

        from_pretrained() first builds the 4.4B model in fp32 on the CPU
        (~17.6 GB of RAM), and quantizing after a full bf16 load needs the
        whole bf16 model in VRAM. Quantizing per shard avoids both.
        """
        from safetensors.torch import load_file  # noqa: PLC0415

        from fish_speech.models.text2semantic.llama import (  # noqa: PLC0415
            DualARTransformer,
            _remap_fish_qwen3_omni_keys,
            precompute_freqs_cis,
        )

        with torch.device("meta"):
            model = DualARTransformer.from_pretrained(
                str(self.model_dir), load_weights=False, max_length=self.max_seq_len
            )
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text())
        for shard in sorted(set(index["weight_map"].values())):
            weights = load_file(str(self.model_dir / shard), device="cpu")
            model.load_state_dict(
                _remap_fish_qwen3_omni_keys(weights), strict=False, assign=True
            )
            del weights
            if self.int8:
                self._quantize_loaded(model, self.device)

        left = [n for n, p in model.named_parameters() if p.is_meta]
        left += [n for n, b in model.named_buffers()
                 if b.is_meta and n not in ("freqs_cis", "causal_mask", "fast_freqs_cis")]
        if left:
            raise RuntimeError(f"weights missing from checkpoint: {left[:5]} ({len(left)})")

        c = model.config
        with torch.device(self.device):
            model.freqs_cis = precompute_freqs_cis(c.max_seq_len, c.head_dim, c.rope_base)
            model.causal_mask = torch.tril(
                torch.ones(c.max_seq_len, c.max_seq_len, dtype=torch.bool)
            )
            model.fast_freqs_cis = precompute_freqs_cis(
                c.num_codebooks, c.fast_head_dim, c.rope_base
            )
        model = model.to(device=self.device, dtype=torch.bfloat16).eval()
        with torch.device(self.device):
            model.setup_caches(max_batch_size=1, max_seq_len=c.max_seq_len,
                               dtype=torch.bfloat16)
        model._cache_setup_done = True
        self.model = model

    @staticmethod
    def _quantize_loaded(model: Any, device: str) -> None:
        """Move every loaded nn.Linear of the slow and fast blocks to the GPU
        as int8, one at a time, so VRAM never holds their bf16 copies."""
        for blocks in (model.layers, model.fast_layers):
            for block in blocks:
                for mod in block.modules():
                    for name, child in list(mod.named_children()):
                        if isinstance(child, torch.nn.Linear) and not child.weight.is_meta:
                            setattr(mod, name, Int8Linear(child.to(device)))

    def _load_codec(self) -> None:
        """Codec Transformers register a 32768x32768 bool causal mask each
        (1 GB); shrink them on the CPU before moving the codec to the GPU."""
        from fish_speech.models.text2semantic.inference import (  # noqa: PLC0415
            load_codec_model,
        )

        codec = load_codec_model(self.model_dir / "codec.pth", "cpu", torch.bfloat16)
        for m in codec.modules():
            mask = getattr(m, "causal_mask", None)
            if isinstance(mask, torch.Tensor) and mask.size(-1) > self.max_seq_len:
                m.causal_mask = mask[: self.max_seq_len, : self.max_seq_len].clone()
        self.codec = codec.to(self.device)
        self.sample_rate = int(codec.sample_rate)
        self.hop = int(codec.frame_length) if hasattr(codec, "frame_length") else 2048

    def _build_prompt_base(self) -> None:
        from fish_speech.content_sequence import TextPart, VQPart  # noqa: PLC0415
        from fish_speech.conversation import Conversation, Message  # noqa: PLC0415
        from fish_speech.models.text2semantic.inference import (  # noqa: PLC0415
            encode_audio,
        )

        if self.prompt_wav:
            codes = encode_audio(self.prompt_wav, self.codec, self.device).cpu()
            _LOGGER.info("reference %s: %d frames (%.1fs)", self.prompt_wav,
                         codes.size(1), codes.size(1) * self.hop / self.sample_rate)
            parts = [
                TextPart(text="convert the provided text to speech reference to "
                         "the following:\n\nText:\n", cal_loss=False),
                TextPart(text=f"<|speaker:0|>{self.prompt_text}", cal_loss=False),
                TextPart(text="\n\nSpeech:\n", cal_loss=False),
                VQPart(codes=codes, cal_loss=False),
            ]
        else:
            _LOGGER.warning("no reference WAV: S2 Pro picks its own voice "
                            "(fixed per text by --seed)")
            parts = [TextPart(text="convert the provided text to speech", cal_loss=False)]

        self._Message, self._TextPart = Message, TextPart
        self._base = Conversation()
        self._base.append(Message(role="system", parts=parts, cal_loss=False,
                                  add_im_start=True, add_im_end=True))

    def load(self) -> None:
        from fish_speech.models.text2semantic.inference import (  # noqa: PLC0415
            decode_one_token_ar,
        )
        from fish_speech.tokenizer import IM_END_TOKEN  # noqa: PLC0415

        t0 = time.monotonic()
        with torch.inference_mode():
            self._load_llm()
            _LOGGER.info("S2 Pro loaded in %.1fs (int8=%s), %.2f GB allocated",
                         time.monotonic() - t0, self.int8,
                         torch.cuda.memory_allocated() / 2**30)
            self._load_codec()
            self._build_prompt_base()

            m = self.model
            dtype = torch.bfloat16
            self._im_end = m.tokenizer.get_token_id(IM_END_TOKEN)
            bias = torch.full((1, 1, m.config.vocab_size), float("-inf"),
                              device=self.device, dtype=dtype)
            bias[0, 0, m.config.semantic_begin_id: m.config.semantic_end_id + 1] = 0.0
            bias[0, 0, self._im_end] = 0.0
            self._bias = bias
            self._temp = torch.tensor(self.temperature, device=self.device, dtype=dtype)
            self._top_p = torch.tensor(self.top_p, device=self.device, dtype=dtype)

        self._prefill = decode_one_token_ar
        self._decode = decode_one_token_ar
        if self.compile:
            self._decode = torch.compile(decode_one_token_ar, backend="inductor",
                                         fullgraph=True, dynamic=True)
        _LOGGER.info("model ready in %.1fs (sample_rate=%d, %.2f GB allocated)",
                     time.monotonic() - t0, self.sample_rate,
                     torch.cuda.memory_allocated() / 2**30)
        self._warmup()

    def _warmup(self) -> None:
        """First call compiles the decode step (a few minutes on a cold
        cache, ~1 min with a warm one); the second confirms the speed."""
        for text in WARMUP_TEXTS:
            t0 = time.monotonic()
            for _ in self.synthesize_iter(text):
                pass
            _LOGGER.info("warmup %r done in %.1fs", text, time.monotonic() - t0)

    # -- inference ----------------------------------------------------------

    def _encode(self, text: str) -> tuple[torch.Tensor, Any, Any]:
        conv = deepcopy(self._base)
        conv.append(self._Message(role="user", parts=[self._TextPart(text=text, cal_loss=False)],
                                  cal_loss=False, add_im_start=True, add_im_end=True))
        conv.append(self._Message(role="assistant", parts=[], cal_loss=False,
                                  modality="voice", add_im_start=True, add_im_end=False))
        encoded, audio_masks, audio_parts = conv.encode_for_inference(
            self.model.tokenizer, num_codebooks=self.model.config.num_codebooks
        )
        return encoded.to(self.device), audio_masks, audio_parts

    def _frames(self, text: str) -> Iterator[torch.Tensor]:
        """Yield codec frames (num_codebooks,) as the model produces them.

        Same steps as fish-speech's generate()/decode_n_tokens(), as a
        generator: prefill, then one compiled decode step per frame with
        repetition-aware sampling, until <|im_end|> or the frame budget.
        """
        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

        from fish_speech.models.text2semantic.inference import (  # noqa: PLC0415
            RAS_WIN_SIZE,
        )

        m = self.model
        cb = m.config.num_codebooks + 1
        encoded, audio_masks, audio_parts = self._encode(text)
        T = encoded.size(1)
        fps = self.sample_rate / self.hop
        budget = int(max(estimate_seconds(text), 1.0) * fps * 3) + 50
        budget = min(budget, m.config.max_seq_len - T - 1)
        if budget <= 0:
            raise ValueError(f"prompt too long for max_seq_len ({T} tokens)")

        sample = dict(temperature=self._temp, top_p=self._top_p, top_k=self.top_k,
                      semantic_logit_bias=self._bias, audio_masks=audio_masks,
                      audio_parts=audio_parts)
        tok = self._prefill(m, encoded.view(1, cb, -1),
                            torch.arange(0, T, device=self.device, dtype=torch.long),
                            kv_len=T, **sample).clone()
        prev = torch.zeros((cb, RAS_WIN_SIZE), dtype=torch.int, device=self.device)
        input_pos = torch.tensor([T], device=self.device, dtype=torch.int)
        n = 0
        while int(tok[0, 0]) != self._im_end:
            yield tok[1:, 0]
            n += 1
            if n >= budget:
                _LOGGER.warning("frame budget (%d) reached before end of speech", budget)
                return
            with sdpa_kernel(SDPBackend.MATH):
                tok = self._decode(model=m, x=tok.view(1, cb, -1), input_pos=input_pos,
                                   previous_tokens=prev, kv_len=T + n, **sample).clone()
            input_pos += 1
            prev = prev.roll(-1, dims=1)
            prev[:, -1] = tok[:, 0]

    def _decode_audio(self, frames: list[torch.Tensor], start: int, end: int) -> bytes:
        """Audio for frames[start:end], decoded with up to context_frames of
        left context. The codec is causal, but its transformers reach far
        back: against a full decode, 64 frames of context gave 7 dB SNR and
        256 frames 39 dB (the bf16 noise floor between kernel shapes)."""
        lo = max(0, start - self.context_frames)
        codes = torch.stack(frames[lo:end], dim=1)[None]
        audio = self.codec.from_indices(codes)[0, 0, (start - lo) * self.hop:]
        x = audio.float().clamp(-1.0, 1.0)
        return (x * 32767.0).to(torch.int16).cpu().numpy().tobytes()

    def synthesize_iter(self, text: str) -> Iterator[tuple[bytes, int]]:
        """Blocking generator: yields (pcm16 chunk, sample_rate) every few
        frames. Run it in a worker thread and forward each chunk immediately."""
        with self._lock, torch.inference_mode():
            if self.seed >= 0:
                torch.manual_seed(self.seed)
            t0 = time.monotonic()
            frames: list[torch.Tensor] = []
            done = 0
            for frame in self._frames(text):
                frames.append(frame)
                need = self.first_frames if done == 0 else self.chunk_frames
                if len(frames) - done >= need:
                    yield self._decode_audio(frames, done, len(frames)), self.sample_rate
                    done = len(frames)
            if len(frames) > done:
                yield self._decode_audio(frames, done, len(frames)), self.sample_rate
            gen = time.monotonic() - t0
            dur = len(frames) * self.hop / self.sample_rate
            _LOGGER.info("block: %.2fs audio in %.2fs (RTF %.2f)", dur, gen,
                         gen / dur if dur else float("nan"))
            if dur < 0.4 * estimate_seconds(text):
                _LOGGER.warning("audio much shorter than the text suggests "
                                "(%.2fs for ~%.1fs)", dur, estimate_seconds(text))


# ---------------------------------------------------------------------------
# Wyoming handler
# ---------------------------------------------------------------------------

class BridgeHandler(AsyncEventHandler):
    def __init__(
        self,
        *args: Any,
        engine: S2Engine,
        info: Info,
        sample_rate: int,
        ready: asyncio.Event,
        default_mood: Optional[str],
        min_merged_seconds: float,
        prebuffer_seconds: float,
        samples_per_chunk: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.info = info
        self.sample_rate = sample_rate
        self._ready = ready
        self._default_mood = default_mood
        self._min_merged = min_merged_seconds
        self._prebuffer = prebuffer_seconds
        self._samples_per_chunk = samples_per_chunk

        self._stash = ""
        self._buf = ""
        self._got_chunks = False
        self._audio_started = False
        self._mood: Optional[str] = default_mood

        # Sentence merging accumulator
        self._pending_sentences: list[str] = []
        self._first_queued = False

        # Duplicate detection for the one-shot synthesize HA sends with a stream
        self._streaming = False
        self._stream_text = ""
        self._last_stream_norm = ""
        self._last_stream_end = 0.0

        self._work_q: Optional[asyncio.Queue] = None
        self._play_q: Optional[asyncio.Queue] = None
        self._fetch_task: Optional[asyncio.Task] = None
        self._play_task: Optional[asyncio.Task] = None

    # -- sentence splitter -------------------------------------------------

    def _take_sentences(self, text: str) -> list[str]:
        self._buf += text
        out: list[str] = []
        pos = 0
        while (m := _TERM_RE.search(self._buf, pos)) is not None:
            word = _WORD_RE.search(self._buf[: m.start()])
            if word and word.group(0).lower() in _ABBREVIATIONS:
                pos = m.start() + 1
                continue
            # Hold back a sentence end inside a tag that has not closed yet
            if DANGLING_RE.search(self._buf[: m.start()]):
                pos = m.start() + 1
                continue
            sentence = self._buf[: m.end()].strip()
            self._buf = self._buf[m.end():]
            pos = 0
            if sentence:
                out.append(sentence)
        return out

    # -- sentence merging ----------------------------------------------------

    def _feed_sentence(self, sentence: str) -> None:
        """Accumulate sentences; queue a merged block once the estimated
        speech length exceeds the merge threshold. Hard cap of 6 sentences
        bounds onset latency for fragment-heavy replies."""
        self._pending_sentences.append(sentence)
        if not self._first_queued:
            # LATENCY-FIRST: don't wait to merge the opening sentence
            self._first_queued = True
            self._flush_pending()
            return
        est = sum(estimate_seconds(s) for s in self._pending_sentences)
        if est >= self._min_merged or len(self._pending_sentences) >= 6:
            self._flush_pending()

    def _flush_pending(self) -> None:
        if self._pending_sentences:
            merged = " ".join(self._pending_sentences)
            self._pending_sentences = []
            self._work_q.put_nowait(merged)

    # -- pipeline lifecycle --------------------------------------------------

    async def _start_pipeline(self) -> None:
        self._work_q = asyncio.Queue()
        self._play_q = asyncio.Queue()
        self._fetch_task = asyncio.create_task(self._fetch_worker())
        self._play_task = asyncio.create_task(self._play_worker())

    async def _stop_pipeline(self) -> None:
        if self._fetch_task is None:
            return
        self._flush_pending()
        await self._work_q.put(None)
        await self._fetch_task
        await self._play_q.put(None)
        await self._play_task
        self._fetch_task = None
        self._play_task = None
        self._work_q = None
        self._play_q = None

    # -- workers ---------------------------------------------------------------

    async def _fetch_worker(self) -> None:
        """Synthesize merged blocks off the event loop, one at a time.

        Failures are reported to the client as a Wyoming Error event, but
        the remaining blocks of the utterance still run.
        """
        await self._ready.wait()

        while (text := await self._work_q.get()) is not None:
            block, self._mood = prepare_block(text, self._mood)
            if not spoken_words(block):
                continue

            _LOGGER.info("synthesizing: %r", block[:80])

            try:
                await self._stream_block(block)
            except Exception as err:
                _LOGGER.exception("block synthesis failed: %r", block[:60])
                try:
                    await self.write_event(
                        Error(
                            text=f"synthesis failed: {err}",
                            code=err.__class__.__name__,
                        ).event()
                    )
                except Exception:
                    pass  # connection may already be gone
                continue

    async def _stream_block(self, text: str) -> None:
        """Run the model in a worker thread and hand every chunk to the play
        queue the moment it is produced (no waiting for the whole block)."""
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def produce() -> None:
            try:
                for item in self.engine.synthesize_iter(text):
                    loop.call_soon_threadsafe(q.put_nowait, item)
            except BaseException as err:  # noqa: BLE001
                loop.call_soon_threadsafe(q.put_nowait, err)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        fut = loop.run_in_executor(None, produce)
        error: Optional[BaseException] = None
        first = True
        t0 = time.monotonic()
        while (item := await q.get()) is not None:
            if isinstance(item, BaseException):
                error = item
                continue
            audio, rate = item
            if audio:
                if first:
                    _LOGGER.info("first audio chunk after %.2fs",
                                 time.monotonic() - t0)
                    first = False
                await self._play_q.put((audio, rate))
        await fut
        if error is not None:
            raise error

    async def _play_worker(self) -> None:
        """Write synthesized audio to the Wyoming client.

        Before AudioStart, accumulate until --prebuffer-seconds is buffered
        (or the utterance ends); afterwards drain continuously in fixed
        --samples-per-chunk AudioChunks, flushing the remainder at the end.
        """
        width = 2
        bytes_per_chunk = width * self._samples_per_chunk

        buf = bytearray()
        rate: Optional[int] = None
        started = False
        buffered_bytes = 0

        async def _maybe_emit_start() -> None:
            nonlocal started
            if not self._audio_started and buf:
                await self.write_event(
                    AudioStart(rate=rate, width=width, channels=1).event()
                )
                self._audio_started = True
            started = True

        while (item := await self._play_q.get()) is not None:
            audio, r = item
            if not audio:
                continue

            if rate is None:
                rate = r
            elif r != rate:
                _LOGGER.warning(
                    "sample rate changed mid-stream (%s -> %s)", rate, r
                )

            buf.extend(audio)
            buffered_bytes += len(audio)

            if not started:
                prebuffer_bytes = int(
                    self._prebuffer * 2 * (rate or self.sample_rate)
                )
                if buffered_bytes < prebuffer_bytes:
                    continue
                await _maybe_emit_start()

            while len(buf) >= bytes_per_chunk:
                await self.write_event(
                    AudioChunk(
                        audio=bytes(buf[:bytes_per_chunk]),
                        rate=rate,
                        width=width,
                        channels=1,
                    ).event()
                )
                del buf[:bytes_per_chunk]

        # Short utterances that never reached the prebuffer threshold are
        # emitted here in full.
        if buf:
            await _maybe_emit_start()
            await self.write_event(
                AudioChunk(
                    audio=bytes(buf), rate=rate, width=width, channels=1
                ).event()
            )
            buf.clear()

    # -- wyoming event handling ---------------------------------------------

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True

        if SynthesizeStart.is_type(event.type):
            self._reset_utterance()
            self._streaming = True
            self._stream_text = ""
            await self._start_pipeline()
            return True

        if SynthesizeChunk.is_type(event.type):
            self._got_chunks = True
            chunk_text = SynthesizeChunk.from_event(event).text
            self._stream_text += chunk_text
            for sentence in self._take_sentences(chunk_text):
                self._feed_sentence(sentence)
            return True

        if SynthesizeStop.is_type(event.type):
            if self._got_chunks:
                tail = self._buf.strip()
                self._buf = ""
                if tail:
                    self._feed_sentence(tail)
            else:
                text = self._stash
                self._stash = ""
                self._stream_text = text
                if text.strip():
                    self._feed_sentence(text.strip())

            await self._finish_utterance()

            self._streaming = False
            self._last_stream_norm = norm_text(self._stream_text)
            self._last_stream_end = time.monotonic()
            await self.write_event(SynthesizeStopped().event())
            return True

        if Synthesize.is_type(event.type):
            text = Synthesize.from_event(event).text or ""

            # (1) Compat one-shot that HA sends alongside a stream: ignore
            #     (but keep it as the text for a start/stop with no chunks).
            if self._streaming:
                self._stash = text
                return True

            # (2) Late duplicate right after a finished stream.
            if (self._last_stream_norm
                    and time.monotonic() - self._last_stream_end
                    < LATE_DUPLICATE_WINDOW_S
                    and norm_text(text) == self._last_stream_norm):
                _LOGGER.warning(
                    "late duplicate of finished stream suppressed: %r",
                    text[:60])
                return True

            # (3) Genuine non-streaming request (tts.speak, "Try voice", ...)
            self._reset_utterance()
            await self._start_pipeline()
            for sentence in self._take_sentences(text + " "):
                self._feed_sentence(sentence)
            tail = self._buf.strip()
            self._buf = ""
            if tail:
                self._feed_sentence(tail)
            await self._finish_utterance(always_audio=True)
            return True

        return True

    def _reset_utterance(self) -> None:
        self._stash = ""
        self._buf = ""
        self._got_chunks = False
        self._audio_started = False
        self._mood = self._default_mood
        self._pending_sentences = []
        self._first_queued = False

    async def _finish_utterance(self, always_audio: bool = False) -> None:
        """Drain the pipeline and close the audio stream. A one-shot client
        waits for audio-stop, so always_audio sends an empty start/stop pair
        when nothing was synthesized (blank text, every block failed)."""
        await self._stop_pipeline()
        if not self._audio_started and always_audio:
            await self.write_event(
                AudioStart(rate=self.sample_rate, width=2, channels=1).event()
            )
            self._audio_started = True
        if self._audio_started:
            await self.write_event(AudioStop().event())
            self._audio_started = False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fish Audio S2 Pro Wyoming bridge")
    p.add_argument("--uri", default="tcp://0.0.0.0:10215", help="Wyoming server URI")
    p.add_argument("--fish-speech-repo", default="src/fish-speech",
                   help="Path to the fish-speech checkout (added to sys.path)")
    p.add_argument("--model-dir", default="models/s2-pro",
                   help="S2 Pro checkpoint (safetensors shards + codec.pth)")
    p.add_argument("--prompt-wav", default="",
                   help="Reference WAV to clone (10-30 s). Empty: no cloning, "
                   "S2 Pro picks a voice")
    p.add_argument("--prompt-text", default="",
                   help="Exact transcript of --prompt-wav")
    p.add_argument("--default-mood", default="",
                   help="Tag for text before any mood tag, e.g. 'warm' (empty: none)")
    p.add_argument("--int8", action="store_true",
                   help="Int8 weight-only slow/fast transformers (~6.4 GB "
                   "instead of ~10.2 GB, ~1.5x faster)")
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile (fast start, ~1.5x slower decoding)")
    p.add_argument("--max-seq-len", type=int, default=4096,
                   help="KV cache length in tokens (reference + text + audio frames)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed per block (-1: random)")
    p.add_argument("--first-frames", type=int, default=8,
                   help="Frames (~46 ms each) before the first audio of a block")
    p.add_argument("--chunk-frames", type=int, default=16,
                   help="Frames per later audio chunk")
    p.add_argument("--context-frames", type=int, default=256,
                   help="Codec left context when decoding a chunk")
    p.add_argument("--min-merged-seconds", type=float, default=3.0,
                   help="Batch sentences until the estimated speech length "
                   "reaches this many seconds (0 disables merging)")
    p.add_argument("--prebuffer-seconds", type=float, default=1.0,
                   help="Hold playback until this much audio is queued (0 disables)")
    p.add_argument("--samples-per-chunk", type=int, default=2048,
                   help="Re-emit AudioChunks of this many samples (~46 ms at 44.1 kHz)")
    p.add_argument("--voice", default="nicoleheart", help="Voice name reported to HA")
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    repo = Path(args.fish_speech_repo).resolve()
    if not (repo / "fish_speech").is_dir():
        raise SystemExit(f"fish-speech repo not found: {repo}")
    sys.path.insert(0, str(repo))

    if args.prompt_wav and not os.path.exists(args.prompt_wav):
        raise SystemExit(f"reference WAV not found: {args.prompt_wav}")
    if args.prompt_wav and not args.prompt_text.strip():
        raise SystemExit("--prompt-text (the reference transcript) is required "
                         "with --prompt-wav")

    default_mood = " ".join(
        f"[{canonical_tag(t)}]" for t in TAG_RE.findall(normalize(args.default_mood))
    ) or (f"[{canonical_tag(args.default_mood)}]" if args.default_mood.strip() else None)

    engine = S2Engine(
        model_dir=args.model_dir,
        prompt_wav=args.prompt_wav or None,
        prompt_text=args.prompt_text.strip(),
        int8=args.int8,
        compile=not args.no_compile,
        max_seq_len=args.max_seq_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        first_frames=args.first_frames,
        chunk_frames=args.chunk_frames,
        context_frames=args.context_frames,
    )

    ready = asyncio.Event()

    async def _load_engine() -> None:
        try:
            await asyncio.to_thread(engine.load)
        except Exception:
            _LOGGER.critical("model load failed", exc_info=True)
            os._exit(1)
        _LOGGER.info("engine ready — synthesis enabled")
        ready.set()

    loader = asyncio.create_task(_load_engine())  # noqa: F841

    attribution = Attribution(name="Fish Audio", url="https://github.com/fishaudio/fish-speech")
    info = Info(
        tts=[
            TtsProgram(
                name="fish-s2-pro",
                attribution=attribution,
                installed=True,
                description="Fish Audio S2 Pro (in-process)",
                version="s2-pro" + ("-int8" if args.int8 else ""),
                voices=[
                    TtsVoice(
                        name=args.voice,
                        attribution=attribution,
                        installed=True,
                        description=f"S2 Pro voice ({args.voice})",
                        version="s2-pro",
                        languages=["en"],
                    )
                ],
                supports_synthesize_streaming=True,
            )
        ]
    )

    server = AsyncServer.from_uri(args.uri)
    _LOGGER.info("listening on %s (model=%s, int8=%s) — loading model in background",
                 args.uri, args.model_dir, args.int8)

    server_task = asyncio.create_task(
        server.run(
            partial(
                BridgeHandler,
                engine=engine,
                info=info,
                sample_rate=engine.sample_rate,
                ready=ready,
                default_mood=default_mood,
                min_merged_seconds=args.min_merged_seconds,
                prebuffer_seconds=args.prebuffer_seconds,
                samples_per_chunk=args.samples_per_chunk,
            )
        )
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, server_task.cancel)
    try:
        await server_task
    except asyncio.CancelledError:
        _LOGGER.info("server stopped")


if __name__ == "__main__":
    asyncio.run(main())
