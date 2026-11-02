"""Phone line: lets a real phone call talk to the same live assistant as the web page.

How a call travels
    caller -> (barbershop's own number forwards, or people dial the new number) -> phone provider (Twilio)
    provider -> POST /telephony/voice      (we answer with a few lines of XML: "connect this call to our WebSocket")
    provider <-> WebSocket /telephony/stream   (8 kHz mu-law audio in both directions, JSON messages)
    our server <-> Gemini Live             (the same LiveCall code the web page uses)

Security (the two routes are open to the internet, so they check who is calling them):
  * /telephony/voice only accepts requests carrying a valid X-Twilio-Signature (HMAC of URL + form with your Twilio Auth Token);
  * it hands out a short-lived, single-use, signed token; /telephony/stream refuses any connection without one;
  * with TELEPHONY_ALLOWED_TO set, calls to any other number are refused;
  * the normal per-number / daily / simultaneous-call limits of app/live.py apply (a caller cannot burn the budget).
Nothing is enabled until TWILIO_AUTH_TOKEN is set.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from collections import deque
from typing import Callable, Optional
from urllib.parse import parse_qsl
from xml.sax.saxutils import quoteattr

import numpy as np
from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from . import g711
from .auth import _b64, _unb64

TOKEN_TTL = 120                      # seconds between answering the call and connecting the audio stream
FRAME_BYTES = 1280                   # 40 ms of 16 kHz 16-bit mono, the unit LiveCall works in
SILENT_RMS = 0.003                   # model audio below this is "padding" and may be skipped when we are ahead of real time
MAX_LEAD = 0.4                       # seconds of already-queued audio before padding is skipped
GREETING_MUTE_MAX = 15.0             # caller audio is muted while the greeting plays, but never longer than this

_used_tokens: dict[str, float] = {}
_hits: deque = deque(maxlen=200)


# ------------------------------------------------------------------ security helpers
def auth_token() -> str:
    return os.environ.get("TWILIO_AUTH_TOKEN", "")


def twilio_signature(url: str, pairs: list[tuple[str, str]], token: str) -> str:
    """Twilio's documented scheme: HMAC-SHA1(token, url + each POST field name+value, sorted by name), base64."""
    data = url + "".join(k + v for k, v in sorted(pairs))
    return base64.b64encode(hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()).decode()


def _stream_key() -> bytes:
    return hmac.new(auth_token().encode(), b"stream-token-v1", hashlib.sha256).digest()


def make_stream_token(caller: str, call_sid: str, ttl: int = TOKEN_TTL) -> str:
    payload = _b64(json.dumps({"c": caller, "s": call_sid, "exp": int(time.time()) + ttl, "n": secrets.token_hex(8)}).encode())
    return payload + "." + _b64(hmac.new(_stream_key(), payload.encode(), hashlib.sha256).digest())


def check_stream_token(token: str) -> Optional[dict]:
    """Returns the payload if the token is genuine, fresh and not used before."""
    try:
        payload, sig = token.split(".")
        if not auth_token() or not hmac.compare_digest(sig, _b64(hmac.new(_stream_key(), payload.encode(), hashlib.sha256).digest())):
            return None
        data = json.loads(_unb64(payload))
        now = time.time()
        for n in [n for n, e in _used_tokens.items() if e < now]:
            _used_tokens.pop(n, None)
        if data["exp"] < now or data["n"] in _used_tokens:
            return None
        _used_tokens[data["n"]] = data["exp"] + 60
        return data
    except Exception:
        return None


def local_number(raw: str) -> str:
    """'+995 555 12 34 56' -> '555123456'; anything that is not a Georgian number -> ''."""
    d = re.sub(r"\D", "", raw or "")
    if d.startswith("995") and len(d) == 12:
        d = d[3:]
    return d if len(d) == 9 and d.startswith(("5", "3", "4", "7")) else ""


def _public_base(request: Request) -> str:
    base = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
    if base:
        return base
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return f"{proto}://{request.headers.get('host', '')}"


# ------------------------------------------------------------------ the adapter
class PhoneSocket:
    """Looks like a browser WebSocket to LiveCall (accept/receive/send_bytes/send_text/close),
    but on the other side it talks Twilio Media Streams: 8 kHz mu-law in JSON."""

    is_phone = True

    def __init__(self, ws: WebSocket, stream_sid: str, caller: str, lang: str):
        self.ws, self.sid = ws, stream_sid
        self.caller_number = caller
        self.query_params = {"lang": lang}
        self.client = None
        self._frames: deque = deque(maxlen=25)      # caller audio waiting to be read (older audio is dropped, not queued up)
        self._ctrl: deque = deque()                 # control messages, never dropped
        self._wake = asyncio.Event()
        self._up = g711.Upsampler8to16()
        self._carry = np.zeros(0, dtype=np.int16)   # inbound 16 kHz samples not yet forming a full 40 ms frame
        self._rem = np.zeros(0, dtype=np.int16)     # outbound 24 kHz samples not yet a multiple of 3
        self.play_until = 0.0                       # when everything we sent so far has been spoken to the caller
        self._playing = False
        self._closed = False
        self._t0 = time.time()
        self._greeted_at: Optional[float] = None    # when the greeting finished playing
        self._tasks = [asyncio.create_task(self._pump()), asyncio.create_task(self._ticker())]

    # ----- what LiveCall calls -----
    async def accept(self) -> None:
        return None

    async def receive(self) -> dict:
        while True:
            if self._ctrl:
                return self._ctrl.popleft()
            if self._frames:
                return {"type": "websocket.receive", "bytes": self._frames.popleft()}
            self._wake.clear()
            await self._wake.wait()

    async def send_bytes(self, pcm24: bytes) -> None:
        if self._closed:
            return
        x = np.concatenate((self._rem, np.frombuffer(pcm24[: len(pcm24) // 2 * 2], dtype=np.int16)))
        n = x.size // 3 * 3
        x, self._rem = x[:n], x[n:]
        if n == 0:
            return
        now = time.time()
        lead = self.play_until - now
        if lead > MAX_LEAD and g711.rms(x) < SILENT_RMS:
            return                                   # Gemini pads with silence and sends faster than real time: skip it
        pcm8 = g711.down_24k_to_8k(x)
        await self._send_json({"event": "media", "streamSid": self.sid,
                               "media": {"payload": base64.b64encode(g711.pcm_to_ulaw(pcm8)).decode()}})
        self.play_until = max(now, self.play_until) + pcm8.size / 8000
        self._set_playing(True)

    async def send_text(self, text: str) -> None:
        try:
            ev = json.loads(text)
        except ValueError:
            return
        kind = ev.get("type")
        if kind == "interrupted":
            await self._send_json({"event": "clear", "streamSid": self.sid})   # drop what is still queued at the provider
            self.play_until = time.time()
            self._rem = np.zeros(0, dtype=np.int16)
            self._set_playing(False)
        elif kind == "turn_end" and self._greeted_at is None:
            self._greeted_at = max(self.play_until, time.time())

    async def close(self, code: int = 1000) -> None:
        if self._closed:
            return
        self._closed = True
        wait = min(10.0, self.play_until - time.time())
        if wait > 0 and self._tasks[0] and not self._tasks[0].done():
            await asyncio.sleep(wait + 0.3)          # let the goodbye finish before the line is dropped
        for t in self._tasks:
            t.cancel()
        try:
            await self.ws.close()
        except Exception:
            pass

    # ----- internals -----
    async def _send_json(self, obj: dict) -> None:
        try:
            await self.ws.send_text(json.dumps(obj))
        except Exception:
            self._closed = True

    def _set_playing(self, value: bool) -> None:
        if value != self._playing:
            self._playing = value
            self._ctrl.append({"type": "websocket.receive", "text": json.dumps({"type": "playing", "value": value})})
            self._wake.set()

    async def _ticker(self) -> None:
        while True:
            await asyncio.sleep(0.1)
            if self._playing and time.time() > self.play_until + 0.3:
                self._set_playing(False)

    def _muted(self) -> bool:
        """The caller's audio is replaced by silence while the greeting plays (so a 'hello?' or line echo cannot cut it off)."""
        now = time.time()
        if now - self._t0 > GREETING_MUTE_MAX:
            return False
        return self._greeted_at is None or now < self._greeted_at

    async def _pump(self) -> None:
        try:
            while True:
                msg = json.loads(await self.ws.receive_text())
                ev = msg.get("event")
                if ev == "media":
                    m = msg.get("media") or {}
                    if m.get("track", "inbound") != "inbound":
                        continue
                    pcm16 = self._up(g711.ulaw_to_pcm(base64.b64decode(m.get("payload", ""))))
                    self._carry = np.concatenate((self._carry, pcm16))
                    while self._carry.size >= FRAME_BYTES // 2:
                        frame, self._carry = self._carry[: FRAME_BYTES // 2], self._carry[FRAME_BYTES // 2:]
                        self._frames.append(bytes(FRAME_BYTES) if self._muted() else frame.tobytes())
                    self._wake.set()
                elif ev == "stop":
                    break
        except (WebSocketDisconnect, RuntimeError, ValueError):
            pass
        except asyncio.CancelledError:
            raise
        self._ctrl.append({"type": "websocket.disconnect"})
        self._wake.set()


# ------------------------------------------------------------------ routes
def make_router(get_agent: Callable[[], object], handle_call) -> APIRouter:
    router = APIRouter()

    @router.post("/telephony/voice")
    async def voice(request: Request):
        token = auth_token()
        if not token:
            return Response("telephony is not switched on", status_code=503, media_type="text/plain")
        now = time.time()
        while _hits and _hits[0] < now - 60:
            _hits.popleft()
        if len(_hits) >= 60:
            return Response("slow down", status_code=429, media_type="text/plain")
        _hits.append(now)
        body = (await request.body())[:20000].decode("utf-8", "replace")
        pairs = parse_qsl(body, keep_blank_values=True)
        url = _public_base(request) + request.url.path + (("?" + request.url.query) if request.url.query else "")
        given = request.headers.get("x-twilio-signature", "")
        if not given or not hmac.compare_digest(given, twilio_signature(url, pairs, token)):
            print("[phone] webhook refused: bad signature (is PUBLIC_BASE_URL exactly the address entered in Twilio?)", flush=True)
            return Response("forbidden", status_code=403, media_type="text/plain")
        form = dict(pairs)
        allowed = [x.strip() for x in os.environ.get("TELEPHONY_ALLOWED_TO", "").split(",") if x.strip()]
        if allowed and form.get("To", "") not in allowed:
            return Response("<Response><Reject/></Response>", media_type="text/xml")
        if get_agent() is None:
            return Response("<Response><Reject/></Response>", media_type="text/xml")
        caller = form.get("From", "")
        tok = make_stream_token(caller, form.get("CallSid", ""))
        ws_url = _public_base(request).replace("https://", "wss://").replace("http://", "ws://") + "/telephony/stream"
        xml = ('<?xml version="1.0" encoding="UTF-8"?><Response><Connect>'
               f'<Stream url={quoteattr(ws_url)}><Parameter name="token" value={quoteattr(tok)}/></Stream>'
               '</Connect><Hangup/></Response>')
        return Response(xml, media_type="text/xml")

    @router.websocket("/telephony/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        agent = get_agent()
        info = None
        sid = ""
        try:
            for _ in range(5):                      # 'connected', then 'start' (which carries the token)
                msg = json.loads(await asyncio.wait_for(ws.receive_text(), 10))
                if msg.get("event") == "start":
                    st = msg.get("start") or {}
                    sid = str(st.get("streamSid") or msg.get("streamSid") or "")
                    info = check_stream_token(str((st.get("customParameters") or {}).get("token", "")))
                    break
        except Exception:
            info = None
        if info is None or agent is None or not sid:
            print("[phone] stream refused (missing/invalid token)", flush=True)
            await ws.close(code=4401)
            return
        caller = local_number(info.get("c", ""))
        lang = os.environ.get("TELEPHONY_LANG", "ka")
        phone = PhoneSocket(ws, sid, caller, lang if lang in ("ka", "ru", "en") else "ka")
        key = "tel:" + hashlib.sha256((info.get("c", "") or "unknown").encode()).hexdigest()[:12]
        print(f"[phone] call started from ***{caller[-3:] if caller else '?'} stream={sid[:8]}", flush=True)
        try:
            await handle_call(phone, agent, False, key)
        finally:
            await phone.close()

    return router
