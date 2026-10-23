"""How long (and how silent) is the assistant's greeting? Repeats N times against a local server.
    PORT=8000 N=6 .venv/bin/python scripts/greeting_probe.py"""
import asyncio, json, os, sys, time
import numpy as np, websockets

async def once(port, lang):
    t0 = time.time(); pcm = bytearray(); first = None
    async with websockets.connect(f"ws://localhost:{port}/ws/call?lang={lang}", max_size=None, open_timeout=30) as ws:
        while time.time() - t0 < 45:
            m = await asyncio.wait_for(ws.recv(), 30)
            if isinstance(m, bytes):
                pcm += m; first = first or time.time() - t0
            elif json.loads(m)["type"] == "turn_end": break
        await ws.send(json.dumps({"type": "hangup"}))
    x = np.frombuffer(bytes(pcm), dtype=np.int16).astype(np.float32) / 32768
    win = 2400; env = [float(np.sqrt(np.mean(x[i:i + win] ** 2))) for i in range(0, max(1, len(x) - win), win)]
    quiet = [e < 0.01 for e in env]; best = cur = 0
    for q in quiet:
        cur = cur + 1 if q else 0; best = max(best, cur)
    return len(x) / 24000, sum(1 for q in quiet if not q) / 10, best / 10

async def main():
    port, n = os.environ.get("PORT", "8000"), int(os.environ.get("N", 6)); out = []
    for i in range(n):
        out.append(await once(port, "ka")); await asyncio.sleep(1.5)
    for a, b, c in out: print(f"  greeting audio {a:5.1f}s  with sound {b:4.1f}s  longest silence {c:4.1f}s")
    print(f"  -> average {sum(o[0] for o in out)/n:.1f}s, worst {max(o[0] for o in out):.1f}s")
asyncio.run(main())
