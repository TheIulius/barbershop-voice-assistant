"""How well does the live assistant record hard Georgian names? (A/B: run the server with NAME_POLICY=old or new)
    .venv/bin/python scripts/name_test.py            (server on :8000; uses synthetic voices, 2 calls at a time)
Verdicts:  FULL = exact full name   FIRST = first name only (safe)   WRONG = a wrong surname was recorded   NONE = no booking prepared."""
import asyncio, json, sys, time
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import websockets
from live_test import say_pcm

NAMES = [  # (spoken full name, first, surname)
    ("ილია შინჯიაშვილი", "ილია", "შინჯიაშვილი"),      # famous-first-name trap (the real case)
    ("აკაკი ცხვედიანი", "აკაკი", "ცხვედიანი"),        # trap: Akaki Tsereteli
    ("შოთა რუსიაშვილი", "შოთა", "რუსიაშვილი"),        # trap: Shota Rustaveli
    ("ვახტანგ გორგაძე", "ვახტანგ", "გორგაძე"),        # trap: Vakhtang Gorgasali
    ("ნიკოლოზ ბარათაშვილი", "ნიკოლოზ", "ბარათაშვილი"),  # control: really famous
    ("ილია ჭავჭავაძე", "ილია", "ჭავჭავაძე"),          # control: really famous
    ("ნიკა ბერიძე", "ნიკა", "ბერიძე"),                # control: very common
    ("ნინო ჭყონია", "ნინო", "ჭყონია"),
    ("ლევან თულაშვილი", "ლევან", "თულაშვილი"),
    ("თამარ ხარაიშვილი", "თამარ", "ხარაიშვილი"),
    ("ზურაბ ქარჩავა", "ზურაბ", "ქარჩავა"),
    ("მარიამ ჩხაიძე", "მარიამ", "ჩხაიძე"),
]
FRAME = 1280


import difflib
_TR = dict(zip("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ", ["a","b","g","d","e","v","z","t","i","k","l","m","n","o","p","zh","r","s","t","u","p","k","gh","q","sh","ch","ts","dz","ts","ch","kh","j","h"]))
def translit(x): return "".join(_TR.get(c, c) for c in x.lower())
def sim(a, b):
    """similarity of two names, ignoring script (Georgian vs Latin spelling)"""
    return difflib.SequenceMatcher(None, translit(a).replace(" ", ""), translit(b).replace(" ", "")).ratio()


async def one(full, first, sur):
    lines = ["გამარჯობა, მინდა თმის შეჭრა ხვალ გიორგისთან, სამ საათზე.", f"{full}, ნომერი 555 123 457.", "დიახ, სწორია.", f"დიახ, {first}."]
    audio, ev = deque(), {"book": None, "asked_surname": False, "last": time.time(), "turn_end": False, "texts": []}
    async with websockets.connect("ws://localhost:" + __import__("os").environ.get("PORT", "8000") + "/ws/call?lang=ka", max_size=None) as ws:
        async def sender():
            while True:
                await ws.send(audio.popleft() if audio else b"\0" * FRAME * 2); await asyncio.sleep(0.04)
        async def receiver():
            async for m in ws:
                ev["last"] = time.time()
                if isinstance(m, bytes): continue
                e = json.loads(m)
                if e["type"] == "ai":
                    ev["cur"] = e["text"]
                if e["type"] == "turn_end":
                    ev["turn_end"] = True
                    if ev.get("cur"): ev["texts"].append(ev["cur"]); ev["cur"] = ""
                if e["type"] == "tool" and e.get("name") == "book" and ev["book"] is None: ev["book"] = e.get("args", {})
        s, r = asyncio.create_task(sender()), asyncio.create_task(receiver())
        async def quiet(timeout=40):
            ev["turn_end"] = False; t0 = time.time()
            while time.time() - t0 < timeout:
                await asyncio.sleep(0.2)
                if ev["turn_end"] and time.time() - ev["last"] > 1.6 and not audio: return
        await quiet()
        for ln in lines:
            if ev["book"] is not None: break
            pcm = await say_pcm(ln)
            for i in range(0, len(pcm), FRAME * 2): audio.append(pcm[i:i + FRAME * 2].ljust(FRAME * 2, b"\0"))
            await quiet()
        s.cancel(); r.cancel()
    asked = any("გვარ" in t and ("?" in t or "მითხრათ" in t or "დაასახელ" in t) for t in ev["texts"])
    rec = (ev["book"] or {}).get("customer_name", "")
    geo = bool(rec) and all(("\u10d0" <= ch <= "\u10ff") or not ch.isalpha() for ch in rec)
    if not ev["book"]: verdict = "NONE"
    elif sim(rec, full) >= 0.8: verdict = "FULL"
    elif sim(rec, first) >= 0.8 and len(rec.split()) == 1: verdict = "FIRST"
    else: verdict = "WRONG"
    if verdict in ("FULL", "FIRST") and not geo: verdict += "-latin"
    return full, rec, verdict, asked


async def main():
    sem = asyncio.Semaphore(2); out = []
    async def run(n):
        async with sem:
            try: out.append(await one(*n))
            except Exception as e: out.append((n[0], f"ERR {type(e).__name__}", "NONE", False))
    names = NAMES[:int(__import__('os').environ.get('N', len(NAMES)))]
    await asyncio.gather(*(run(n) for n in names))
    out.sort(key=lambda x: [n[0] for n in NAMES].index(x[0]))
    for full, rec, v, asked in out:
        print(f"{v:6s} said: {full:24s} recorded: {rec or '-':26s} {'(asked for surname)' if asked else ''}")
    c = {k: sum(1 for o in out if o[2].startswith(k)) for k in ("FULL", "FIRST", "WRONG", "NONE")}
    lat = sum(1 for o in out if o[2].endswith("-latin"))
    print(f"\nTOTAL {len(out)}: full name {c['FULL']}, first name only {c['FIRST']}, WRONG {c['WRONG']}, no booking {c['NONE']}; written in Latin letters: {lat}; asked for a surname in {sum(1 for o in out if o[3])} calls")

if __name__ == "__main__":
    asyncio.run(main())
