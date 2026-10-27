"""Save WAVs from a running bridge, one per text, for listening to tags.

  .venv/bin/python scripts/save_clips.py --port 10215 --out out/clips

Moods and delivery tags all use the same sentence (with an untagged
control), so they can be compared directly; the bridge's fixed seed keeps
everything else equal. Sound tags and the HA prompt's examples follow.
"""

import argparse
import asyncio
import wave
from pathlib import Path

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.tts import Synthesize

SENTENCE = "The package arrived this afternoon, and it is sitting by the front door."

MOODS = ["warm", "cheerful", "playful", "excited", "curious", "confident",
         "surprised", "serious", "sad", "angry"]
DELIVERY = ["whisper", "low voice", "loud", "emphasis", "in a hurry"]

SOUNDS = {
    "laughing": "That is the funniest thing I have heard all week. [laughing] I cannot believe it.",
    "chuckle": "You asked me to remind you about the cake. [chuckle] It is still in the oven.",
    "sigh": "[sigh] The printer is out of paper again.",
    "inhale": "[inhale] Okay, here is the plan for tomorrow.",
    "exhale": "The storm has passed. [exhale] Everything is fine.",
    "tsk": "The freezer door was left open. [tsk] Again.",
    "clearing_throat": "[clearing throat] May I have your attention, please.",
    "pause": "The winner of the cleanest room award is. [pause] Nobody.",
}

EXAMPLES = [
    "[warm] Good morning! [inhale] The coffee is brewing already.",
    "[confident] Lights are off, doors are locked, thermostat set. All handled.",
    "[serious] The garage door has been open since two AM. [sigh] Closing it now would be wise.",
    "[cheerful] The heating bill is down twelve percent this month! [laughing] Nice work.",
    "[playful] You asked me to remind you about the cake. [chuckle] It is still in the oven.",
    "[warm] [whisper] It is almost midnight. Nothing is scheduled for tomorrow.",
    "[serious] [loud] Smoke detected in the kitchen. Please check it now.",
    "[curious] Let me think. [pause] Yes, the dishwasher finished an hour ago.",
    "[surprised] The freezer is at minus two degrees? [tsk] Someone left the door open.",
]


def slug(text):
    return text.replace(" ", "_")


async def render(host, port, text, path):
    async with AsyncTcpClient(host, port) as client:
        await client.write_event(Synthesize(text=text).event())
        rate, audio = 44100, bytearray()
        while True:
            event = await asyncio.wait_for(client.read_event(), 180)
            if event is None:
                break
            if AudioStart.is_type(event.type):
                rate = AudioStart.from_event(event).rate
            elif AudioChunk.is_type(event.type):
                audio += AudioChunk.from_event(event).audio
            elif AudioStop.is_type(event.type):
                break
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(audio))
    return len(audio) / 2 / rate


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=10215)
    ap.add_argument("--out", default="out/clips")
    args = ap.parse_args()
    out = Path(args.out)

    jobs = [("1_moods", "00_control", SENTENCE)]
    jobs += [("1_moods", f"{i + 1:02d}_{slug(m)}", f"[{m}] {SENTENCE}")
             for i, m in enumerate(MOODS)]
    jobs += [("2_delivery", f"{i + 1:02d}_warm+{slug(d)}", f"[warm] [{d}] {SENTENCE}")
             for i, d in enumerate(DELIVERY)]
    jobs += [("3_sounds", f"{i + 1:02d}_{name}", text)
             for i, (name, text) in enumerate(SOUNDS.items())]
    jobs += [("4_examples", f"{i + 1:02d}", text) for i, text in enumerate(EXAMPLES)]

    lines = []
    for folder, name, text in jobs:
        (out / folder).mkdir(parents=True, exist_ok=True)
        seconds = await render(args.host, args.port, text, out / folder / f"{name}.wav")
        lines.append(f"{folder}/{name}.wav\t{seconds:.1f}s\t{text}")
        print(lines[-1], flush=True)
    (out / "index.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
