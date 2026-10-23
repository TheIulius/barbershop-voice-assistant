"""Direct hearing benchmark across Gemini live models (does NOT go through our server).
Each model is asked to repeat, in Georgian letters, exactly what it heard. We compare with the real sentence.
    .venv/bin/python scripts/hearing_bench.py              (needs GEMINI_API_KEY in .env)"""
import asyncio, base64, difflib, json, os, re, subprocess, sys, tempfile, time, wave
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.main import _load_env
_load_env()
import numpy as np, edge_tts, websockets

KEY = os.environ["GEMINI_API_KEY"]
URL = "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=" + KEY
SYSTEM = ("You are an echo machine. Repeat the caller's words EXACTLY as they were said, in Georgian letters, with no additions, "
          "no greeting and no comments. Names must be written in Georgian letters too. Output only the repeated words.")
from app.recorder import PHRASES as _PH, SAMPLES
PHRASES = [(c, t) for c, t, d in _PH]
DIGITS = {t: d for c, t, d in _PH}
VARIANTS = {
    "3.1-flash-live (minimal thinking)": ("gemini-3.1-flash-live-preview", {"thinkingConfig": {"thinkingLevel": "minimal"}}),
    "3.8-live (default)": ("gemini-3.8-live", {}),
    "3.8-live (thinking budget 0)": ("gemini-3.8-live", {"thinkingConfig": {"thinkingBudget": 0}}),
    "3.8-live-extended-thinking (low)": ("gemini-3.8-live-extended-thinking", {"thinkingConfig": {"thinkingLevel": "low"}}),
    "2.5-flash-native-audio": ("gemini-2.5-flash-native-audio-latest", {}),
}

def norm(t): return re.sub(r"[^ა-ჿa-z0-9 ]", "", t.lower()).strip()
def sim(a, b):
    """Similarity of what the model repeated (a) to the real phrase (b). For phone numbers the model may write digits
    instead of Georgian number words, so a number is also compared as digits."""
    best = difflib.SequenceMatcher(None, norm(a).replace(" ", ""), norm(b).replace(" ", "")).ratio()
    d = DIGITS.get(b, "")
    if d:
        got = re.sub(r"\D", "", a)
        best = max(best, difflib.SequenceMatcher(None, got, d).ratio())
    return best

async def make(text, voice, noisy):
    mp3 = bytearray()
    async for c in edge_tts.Communicate(text, voice).stream():
        if c["type"] == "audio": mp3 += c["data"]
    d = tempfile.mkdtemp(); open(d + "/a.mp3", "wb").write(mp3)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", d + "/a.mp3", d + "/a.wav"], check=True)
    w = wave.open(d + "/a.wav"); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)
    if noisy:
        x *= 0.6; sig = np.sqrt(np.mean(x ** 2)) + 1e-6; x += np.random.RandomState(1).randn(len(x)).astype(np.float32) * sig / (10 ** (15 / 20))
    return np.clip(x, -32768, 32767).astype(np.int16).tobytes()

async def run_variant(name, model, gen, clips):
    res = []
    setup = {"setup": {"model": "models/" + model, "systemInstruction": {"parts": [{"text": SYSTEM}]},
             "generationConfig": {"responseModalities": ["AUDIO"], **gen}, "outputAudioTranscription": {}}}
    try:
        async with websockets.connect(URL, max_size=None, open_timeout=20) as ws:
            await ws.send(json.dumps(setup)); first = json.loads(await asyncio.wait_for(ws.recv(), 15))
            if "setupComplete" not in first: return name, [], f"setup refused: {str(first)[:100]}"
            for label, text, noisy, pcm in clips:
                for i in range(0, len(pcm), 3200):
                    await ws.send(json.dumps({"realtimeInput": {"audio": {"data": base64.b64encode(pcm[i:i + 3200]).decode(), "mimeType": "audio/pcm;rate=16000"}}})); await asyncio.sleep(0.02)
                t_end = time.time(); out, first_audio = "", None
                for _ in range(40):  # trailing silence so the server detects the end of speech
                    await ws.send(json.dumps({"realtimeInput": {"audio": {"data": base64.b64encode(b"\0" * 3200).decode(), "mimeType": "audio/pcm;rate=16000"}}})); await asyncio.sleep(0.04)
                    try:
                        m = json.loads(await asyncio.wait_for(ws.recv(), 0.01))
                    except Exception: continue
                    break_now = False
                    sc = m.get("serverContent", {})
                    if (sc.get("modelTurn") or {}).get("parts") and first_audio is None: first_audio = time.time() - t_end
                    if sc.get("outputTranscription"): out += sc["outputTranscription"].get("text", "")
                    if sc.get("turnComplete"): break
                else:
                    pass
                deadline = time.time() + 20
                while time.time() < deadline and not (out and False):
                    try: m = json.loads(await asyncio.wait_for(ws.recv(), 6))
                    except Exception: break
                    sc = m.get("serverContent", {})
                    if (sc.get("modelTurn") or {}).get("parts") and first_audio is None: first_audio = time.time() - t_end
                    if sc.get("outputTranscription"): out += sc["outputTranscription"].get("text", "")
                    if sc.get("turnComplete"): break
                res.append((label, text, noisy, out.strip(), first_audio))
    except Exception as e:
        return name, res, f"{type(e).__name__}: {str(e)[:120]}"
    return name, res, ""

async def main():
    clips = []
    if "--real" in sys.argv:       # your own recordings (data/voice_samples/NN.wav, made with /record)
        for i, (label, text) in enumerate(PHRASES):
            f = SAMPLES / f"{i:02d}.wav"
            if f.exists():
                w = wave.open(str(f)); clips.append((label, text, False, w.readframes(w.getnframes())))
        if not clips: sys.exit("no recordings found: open http://localhost:8000/record first")
    else:
        voices = ["ka-GE-EkaNeural", "ka-GE-GiorgiNeural"]
        for i, (label, text) in enumerate(PHRASES):
            for noisy in (False, True):
                clips.append((label, text, noisy, await make(text, voices[i % 2], noisy)))
    results = await asyncio.gather(*(run_variant(n, m, g, clips) for n, (m, g) in VARIANTS.items()))
    print(f"{len(clips)} tests per model ({'REAL recordings' if '--real' in sys.argv else 'synthetic voices'})\n")
    for name, res, err in results:
        if err and not res: print(f"## {name}: FAILED - {err}\n"); continue
        def score(sel):
            rs = [sim(t, o) for (l, t, n, o, f) in res if sel(l, n)]
            return f"{100 * sum(rs) / max(1, len(rs)):4.0f}% similar, {sum(1 for r in rs if r >= 0.85)}/{len(rs)} exact-ish"
        lat = [f for (*_, f) in res if f]
        print(f"## {name}" + (f"   (stopped early: {err})" if err else ""))
        print(f"   all clean   : {score(lambda l, n: not n)}")
        print(f"   all noisy   : {score(lambda l, n: n)}")
        for lab in ("short", "sentence", "name", "number"):
            print(f"   {lab:9s}   : {score(lambda l, n, lab=lab: l == lab)}")
        print(f"   median time to first reply audio: {sorted(lat)[len(lat)//2]:.1f}s ({len(lat)} replies)\n" if lat else "   no replies\n")
    print("=== what each model repeated (per phrase) ===")
    for k, (label, text, noisy, pcm) in enumerate(clips):
        print(f"\n[{label}] {text}")
        for name, res, err in results:
            got = next((o for (l, t, n, o, f) in res if t == text and n == noisy), "(no reply)")
            print(f"   {name[:34]:34s} {sim(got, text)*100:3.0f}%  {got[:70]}")
    json.dump([{"model": n, "results": r} for n, r, e in results], open("/tmp/hearing_bench.json", "w"), ensure_ascii=False, indent=1)

if __name__ == "__main__":
    asyncio.run(main())
