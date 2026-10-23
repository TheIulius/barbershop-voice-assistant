"""Acts like a browser on the live call: streams synthetic caller speech in real time and prints what happens.
    .venv/bin/python scripts/live_test.py booking_en | booking_ka | booking_ru | interrupt
(server must be running on :8000)"""
import asyncio, json, subprocess, sys, tempfile, time, wave
from collections import deque
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import websockets
from app import tts

FRAME = 1280  # 40 ms of 16 kHz 16-bit mono
async def say_pcm(text, voice=None, noise_db=None, gain=1.0):
    import numpy as np, edge_tts
    if voice:
        mp3 = bytearray()
        async for c in edge_tts.Communicate(text, voice).stream():
            if c["type"] == "audio": mp3 += c["data"]
        mp3 = bytes(mp3)
    else:
        mp3 = await tts.synthesize(text)
    d = tempfile.mkdtemp(); (Path(d) / "a.mp3").write_bytes(mp3)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", f"{d}/a.mp3", f"{d}/a.wav"], check=True)
    w = wave.open(f"{d}/a.wav"); raw = w.readframes(w.getnframes())
    if noise_db is None and gain == 1.0: return raw
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) * gain
    if noise_db is not None:
        sig = np.sqrt(np.mean(x ** 2)) + 1e-6; x += np.random.randn(len(x)).astype(np.float32) * sig / (10 ** (noise_db / 20))
    return np.clip(x, -32768, 32767).astype(np.int16).tobytes()

# lines may be "text" or ("text", voice, noise_db, gain)
SCENARIOS = {
 "booking_mon_en": ("en", ["Hi, I'd like a haircut on Monday with Giorgi, around three in the afternoon.", "My name is Nino and my number is 5 5 5, 1 2 3, 4 5 6.", "Yes, that number is correct.", "__AFTER_BOOK__Yes, that's right.", "No, that's all. Thank you, bye!"]),
 "booking_mon_ka": ("ka", ["გამარჯობა, მინდა თმის შეჭრა ორშაბათს გიორგისთან, დაახლოებით სამ საათზე.", "მე ვარ ნინო, ნომერი 555 123 457.", "დიახ, ნომერი სწორია.", "__AFTER_BOOK__დიახ, სწორია.", "არა, მადლობა, ნახვამდის."]),
 "services_ka": ("ka", ["გამარჯობა, რა სერვისები გაქვთ?"]),
 "services_en": ("en", ["Hi, what services do you have?"]),
 "robust_ka": ("ka", [("¿Qué? Hola, ¿qué tal?", "es-ES-ElviraNeural", None, 1.0), ("კი", "ka-GE-EkaNeural", 8, 0.5), ("გამარჯობა, მინდა თმის შეჭრა ხვალ სამ საათზე.", "ka-GE-GiorgiNeural", 10, 0.4),
        ("Hi, I'd like a haircut tomorrow at three, please.", "en-US-GuyNeural", None, 1.0), ("შეგიძლია გამიმეორო ის რაც თქვი?", "ka-GE-EkaNeural", None, 1.0)]),
 "locked_ka": ("ka", ["გამარჯობა, მინდა თმის შეჭრა.", "Здравствуйте, давайте лучше говорить по-русски, я хочу записаться на стрижку завтра днём.", "Хорошо, а сколько стоит стрижка?"]),
 "booking_en": ("en", ["Hi, I'd like a haircut tomorrow with Giorgi, around three in the afternoon.", "My name is Nino and my number is 5 5 5, 1 2 3, 4 5 6.", "Yes, that number is correct.", "__AFTER_BOOK__Yes, that's right.", "No, that's all. Thank you, bye!"]),
 "booking_ka": ("ka", ["გამარჯობა, მინდა თმის შეჭრა ხვალ გიორგისთან, დაახლოებით სამ საათზე.", "მე ვარ ნინო, ნომერი 555 123 457.", "დიახ, ნომერი სწორია.", "__AFTER_BOOK__დიახ, სწორია.", "არა, მადლობა, ნახვამდის."]),
 "booking_ru": ("ru", ["Здравствуйте, хочу записаться на стрижку завтра к Гиорги, часа в три.", "Меня зовут Анна, телефон 555 123 458.", "Да, номер верный.", "__AFTER_BOOK__Да, всё верно.", "Нет, спасибо, до свидания."]),
 "interrupt": ("en", ["__NOWAIT__Hello, please explain slowly and in great detail every service you offer, with the price and the duration of each, and also all your opening hours and policies.", "__INTERRUPT__Sorry, stop. Actually, just tell me the price of a haircut."]),
}

async def main(name):
    lang, lines = SCENARIOS[name]
    audio = deque(); events = []; got_audio = 0; t_start = time.time()
    async with websockets.connect("ws://localhost:8000/ws/call?lang=" + lang, max_size=None) as ws:
        async def sender():
            while True:
                if audio: await ws.send(audio.popleft())
                else: await ws.send(b"\x00" * (FRAME * 2))
                await asyncio.sleep(0.04)
        turn_ev = asyncio.Event(); state = {"audio_bytes": 0, "interrupted": 0, "booked": 0, "hangup": False, "book_seen": False, "last": time.time(), "turn_end": False}
        async def receiver():
            async for m in ws:
                state["last"] = time.time()
                if isinstance(m, bytes): state["audio_bytes"] += len(m); state.setdefault("pcm", bytearray()).extend(m); continue
                e = json.loads(m); t = time.time() - t_start
                if e["type"] in ("user", "ai"): events.append((round(t,1), e["type"], e["text"]))
                elif e["type"] in ("tool", "booked", "interrupted", "error", "ready", "turn_end", "hangup"):
                    events.append((round(t,1), e["type"], {k: v for k, v in e.items() if k != "type"}))
                if e["type"] == "tool" and e.get("name") == "book": state["book_seen"] = True
                if e["type"] == "interrupted": state["interrupted"] += 1
                if e["type"] == "booked": state["booked"] += 1
                if e["type"] == "hangup": state["hangup"] = True; turn_ev.set()
                if e["type"] == "turn_end": state["turn_end"] = True
        s, r = asyncio.create_task(sender()), asyncio.create_task(receiver())
        async def wait_turn(timeout=45):
            """done when the assistant finished a turn AND everything has been quiet for 1.8 s"""
            state["turn_end"] = False; t0 = time.time()
            while time.time() - t0 < timeout:
                await asyncio.sleep(0.2)
                if state["hangup"]: return
                if state["turn_end"] and time.time() - state["last"] > 1.8 and not audio: return
            events.append((round(time.time()-t_start,1), "TIMEOUT", "no quiet turn_end"))
        await wait_turn()                      # greeting
        for ln in lines:
            if state["hangup"]: break
            if isinstance(ln, str) and ln.startswith("__AFTER_BOOK__"):
                ln = ln[len("__AFTER_BOOK__"):]
                for _ in range(150):                 # wait (max 30 s) until the assistant has prepared the booking and read it back
                    if state["book_seen"]: break
                    await asyncio.sleep(0.2)
                await wait_turn()
            nowait = isinstance(ln, str) and ln.startswith("__NOWAIT__")
            if nowait: ln = ln[len("__NOWAIT__"):]
            if isinstance(ln, str) and ln.startswith("__INTERRUPT__"):
                b0 = state["audio_bytes"]
                for _ in range(200):           # wait until the assistant has been talking for ~2.5 s, then cut in
                    if state["audio_bytes"] - b0 > 120000: break
                    await asyncio.sleep(0.05)
                events.append((round(time.time()-t_start,1), "NOTE", f"assistant has spoken {(state['audio_bytes']-b0)/48000:.1f}s of this reply when caller cuts in"))
                ln = ln[len("__INTERRUPT__"):]
                cut = True
            else: cut = False
            if isinstance(ln, tuple): pcm = await say_pcm(ln[0], ln[1], ln[2], ln[3]); ln = ln[0]
            else: pcm = await say_pcm(ln)
            events.append((round(time.time()-t_start,1), "CALLER SAYS", ln))
            for i in range(0, len(pcm), FRAME * 2): audio.append(pcm[i:i + FRAME * 2].ljust(FRAME * 2, b"\x00"))
            if nowait: await asyncio.sleep(0.3); continue
            await wait_turn()
        s.cancel(); r.cancel()
    for i, e in enumerate(events):
        nxt = events[i + 1] if i + 1 < len(events) else None
        if e[1] in ("ai", "user") and nxt and nxt[1] == e[1]: continue   # show only the final text of a streamed line
        print(f"{e[0]:6.1f}s  {e[1]:12s} {e[2]}")
    import numpy as np
    x = np.frombuffer(bytes(state.get("pcm", b"")), dtype=np.int16).astype(np.float32) / 32768
    win = 2400; env = [float(np.sqrt(np.mean(x[i:i+win] ** 2))) for i in range(0, max(0, len(x) - win), win)]
    quiet = [e < 0.01 for e in env]; runs, cur = [], 0
    for q in quiet:
        if q: cur += 1
        else:
            if cur: runs.append(cur)
            cur = 0
    if cur: runs.append(cur)
    print(f"\nsilence_stats: {len(x)/24000:.1f}s of audio, {sum(1 for q in quiet if not q)/10:.1f}s with sound, longest silent stretch {max(runs or [0])/10:.1f}s, silent stretches >1s: {[r/10 for r in runs if r >= 10]}")
    print(f"\nreply audio received: {state['audio_bytes']/48000:.1f}s  interrupted events: {state['interrupted']}  bookings: {state['booked']}")
if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
