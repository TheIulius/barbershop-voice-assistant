"""Pretends to be Twilio calling the local server: signed webhook -> TwiML -> media stream with 8 kHz mu-law audio.
Synthetic caller speech goes in; the assistant's reply audio comes back and is measured; the call record is printed.

    TWILIO_AUTH_TOKEN=test PUBLIC_BASE_URL=http://localhost:8001 .venv/bin/uvicorn app.main:app --port 8001   (one terminal)
    TWILIO_AUTH_TOKEN=test PUBLIC_BASE_URL=http://localhost:8001 .venv/bin/python scripts/phone_sim.py booking_ka   (another)
"""
import asyncio, base64, json, os, re, sys, time
from pathlib import Path
from urllib.parse import quote
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import httpx, numpy as np, websockets
from app import g711, telephony
from live_test import SCENARIOS, say_pcm

BASE = os.environ.get("PUBLIC_BASE_URL", "http://localhost:8001")
TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "test")


async def main(name, caller="+995555123457"):
    lang, lines = SCENARIOS[name]
    pairs = [("From", caller), ("To", "+15005550006"), ("CallSid", "CA" + str(int(time.time())))]
    body = "&".join(f"{k}={quote(v)}" for k, v in pairs)
    sig = telephony.twilio_signature(BASE + "/telephony/voice", pairs, TOKEN)
    r = httpx.post(BASE + "/telephony/voice", content=body, headers={"content-type": "application/x-www-form-urlencoded", "x-twilio-signature": sig})
    print("webhook ->", r.status_code, r.text[:150].replace("\n", " "), "...")
    assert r.status_code == 200, "webhook refused"
    url = re.search(r'Stream url="([^"]+)"', r.text).group(1); tok = re.search(r'value="([^"]+)"', r.text).group(1)
    out = {"audio": bytearray(), "clears": 0, "last": time.time(), "media": 0}
    t0 = time.time(); events = []
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({"event": "start", "streamSid": "MZsim", "start": {"streamSid": "MZsim", "callSid": pairs[2][1], "customParameters": {"token": tok}}}))
        pending = []
        async def sender():
            seq = 0
            while True:
                if pending: chunk = pending.pop(0)
                else: chunk = b"\xff" * 160                     # mu-law silence
                seq += 1
                await ws.send(json.dumps({"event": "media", "sequenceNumber": str(seq), "streamSid": "MZsim",
                                          "media": {"track": "inbound", "chunk": str(seq), "timestamp": str(seq * 20), "payload": base64.b64encode(chunk).decode()}}))
                await asyncio.sleep(0.02)
        async def receiver():
            try:
                async for m in ws:
                    e = json.loads(m)
                    if e["event"] == "media":
                        out.setdefault("t_first", time.time()); out["audio"] += g711.ulaw_to_pcm(base64.b64decode(e["media"]["payload"])).tobytes(); out["last"] = time.time(); out["media"] += 1
                    elif e["event"] == "clear": out["clears"] += 1
            except websockets.ConnectionClosed:
                pass
        s, rc = asyncio.create_task(sender()), asyncio.create_task(receiver())
        async def wait_reply(timeout=40):
            start = time.time(); n0 = out["media"]
            while time.time() - start < timeout and not rc.done():
                await asyncio.sleep(0.2)
                # a real phone plays audio in real time: the caller only answers once everything received has been *heard*
                heard_until = out.setdefault("t_first", time.time()) + len(out["audio"]) / 16000
                if out["media"] > n0 and time.time() - out["last"] > 1.0 and time.time() > heard_until + 1.2 and not pending: return
            events.append("(no complete reply within %ds)" % timeout)
        await wait_reply()
        for ln in lines:
            if rc.done(): break
            if isinstance(ln, tuple): pcm = await say_pcm(ln[0], ln[1], ln[2], ln[3]); ln = ln[0]
            else: ln = ln.replace("__AFTER_BOOK__", ""); pcm = await say_pcm(ln)
            x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            p8 = x[: x.size // 2 * 2].reshape(-1, 2).mean(axis=1).astype(np.int16)            # 16 kHz -> 8 kHz
            u = g711.pcm_to_ulaw(p8)
            pending.extend(u[i:i + 160].ljust(160, b"\xff") for i in range(0, len(u), 160))
            events.append(f"{time.time() - t0:5.1f}s CALLER SAYS: {ln}")
            await wait_reply()
        s.cancel()
        if not rc.done():
            await ws.send(json.dumps({"event": "stop", "streamSid": "MZsim"}))
        await asyncio.sleep(1.0)
        rc.cancel()
    print("\n".join(events))
    secs = len(out["audio"]) / 16000
    print(f"reply audio received: {secs:.1f}s  clear events: {out['clears']}  (call lasted {time.time() - t0:.0f}s)")
    await asyncio.sleep(1)
    try:
        last = httpx.get(BASE + "/api/calls?limit=1").json()[0]
        print(f"\ncall record: channel={last.get('channel')} reason={last['reason']} seconds={last['seconds']} lang={last['lang']} model={last['model']}")
        for t in last["trail"]:
            if t["what"] in ("caller", "assistant", "tool", "booked", "error", "hangup"):
                print(f"  {t['t']:5.1f}s {t['what']:10s} " + str({k: v for k, v in t.items() if k not in ('t', 'what')})[:150])
    except Exception as e:
        print("could not read /api/calls:", e)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "services_ka"))
