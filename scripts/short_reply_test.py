"""One- and two-word answers: how often does the assistant say it 'did not understand' instead of moving on / confirming?
    PORT=8000 .venv/bin/python scripts/short_reply_test.py     (server running; synthetic noisy, quiet voices)
Verdicts: PROCEED = carried on with the booking   CONFIRM = asked 'did you say ...?'   REPEAT = asked to repeat (the annoying one)"""
import asyncio, json, os, sys, time
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import websockets
from live_test import say_pcm

FRAME = 1280
SETUP = "გამარჯობა, მინდა თმის შეჭრა."
SHORT = ["გიორგი", "ნიკა", "ხვალ", "ხუთზე", "ორშაბათს", "ნებისმიერი", "კი", "არა"]
if os.environ.get("SET") == "backchannel":   # everyday little replies that must never be answered with "I did not understand"
    SHORT = ["ეგრე იყოს", "კარგია", "დაიცა რა", "ხო", "აჰა", "მადლობა", "ერთი წუთით", "ოკ", "გასაგებია", "არაუშავს"]
VOICES = ["ka-GE-EkaNeural", "ka-GE-GiorgiNeural"]


def verdict(text: str) -> str:
    t = text or ""
    if t.strip() == "" or t.startswith("აქ ხართ") or "ისევ აქ ხართ" in t: return "MISSED"   # silence or our own "are you there?" nudge = the short word was not heard at all
    if any(k in t for k in ("გაიმეორ", "ვერ გავიგე", "არ გამიგია", "გაიმეოროთ", "ქართულად")): return "REPEAT"
    if "?" in t and any(k in t for k in ("თქვით", "გულისხმობთ", "ხომ ", "სწორია", "ანუ")): return "CONFIRM"
    return "PROCEED"


async def one(i, word):
    audio, ev = deque(), {"last": time.time(), "turn_end": False, "cur": "", "texts": []}
    port = os.environ.get("PORT", "8000")
    async with websockets.connect(f"ws://localhost:{port}/ws/call?lang=ka", max_size=None) as ws:
        async def sender():
            while True:
                await ws.send(audio.popleft() if audio else b"\0" * FRAME * 2); await asyncio.sleep(0.04)
        async def receiver():
            async for m in ws:
                ev["last"] = time.time()
                if isinstance(m, bytes): continue
                e = json.loads(m)
                if e["type"] == "ai": ev["cur"] = e["text"]
                if e["type"] == "turn_end":
                    ev["turn_end"] = True
                    if ev["cur"]: ev["texts"].append(ev["cur"]); ev["cur"] = ""
        s, r = asyncio.create_task(sender()), asyncio.create_task(receiver())
        async def quiet(timeout=40):
            ev["turn_end"] = False; t0 = time.time()
            while time.time() - t0 < timeout:
                await asyncio.sleep(0.2)
                if ev["turn_end"] and time.time() - ev["last"] > 1.6 and not audio: return
        async def say(text, noisy):
            pcm = await say_pcm(text, VOICES[i % 2], 12 if noisy else None, 0.5 if noisy else 1.0)
            for k in range(0, len(pcm), FRAME * 2): audio.append(pcm[k:k + FRAME * 2].ljust(FRAME * 2, b"\0"))
        await quiet()                       # greeting
        await say(SETUP, False); await quiet()
        n0 = len(ev["texts"])
        await say(word, True); await quiet()
        s.cancel(); r.cancel()
    reply = " ".join(ev["texts"][n0:]) or ev["cur"]
    return word, verdict(reply), reply[:90]


async def main():
    sem = asyncio.Semaphore(2); out = []
    async def run(i, w):
        async with sem:
            try: out.append(await one(i, w))
            except Exception as e: out.append((w, "ERROR", type(e).__name__))
    await asyncio.gather(*(run(i, w) for i, w in enumerate(SHORT)))
    out.sort(key=lambda x: SHORT.index(x[0]))
    for w, v, t in out: print(f"{v:8s} {w:12s} -> {t}")
    c = {k: sum(1 for o in out if o[1] == k) for k in ("PROCEED", "CONFIRM", "REPEAT", "MISSED", "ERROR")}
    print(f"\nSUMMARY: carried on {c['PROCEED']}, confirmed {c['CONFIRM']}, ASKED TO REPEAT {c['REPEAT']}, NOT HEARD AT ALL {c['MISSED']}, errors {c['ERROR']} (of {len(out)})")

if __name__ == "__main__":
    asyncio.run(main())
