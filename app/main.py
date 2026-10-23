"""Web server: serves the test "phone" page and the JSON API.
Run:  .venv/bin/uvicorn app.main:app --port 8000   (binds localhost only by default)"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .actions import Actions
from .agent import Agent
from .calendars import GoogleCalendar, LocalCalendar
from .llm import LLMError, make_llm
from .shop import Shop
from . import tts
from .lab import router as lab_router
from .recorder import router as recorder_router
from .live import handle_call, CALL_LOG
from . import archive
from .auth import Auth, AuthMiddleware, make_router

ROOT = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()
DATA = ROOT / "data"
shop = Shop.load(DATA / "shop.json")
if os.environ.get("CALENDAR_BACKEND", "local") == "google":
    calendar = GoogleCalendar(os.environ["GOOGLE_SERVICE_ACCOUNT_FILE"], shop)
else:
    calendar = LocalCalendar(DATA / "local_calendar.json")
actions = Actions(shop, calendar, DATA)

llm_error = ""
try:
    llm = make_llm()
except LLMError as e:
    llm, llm_error = None, str(e)
agent = Agent(shop, actions, llm)

PUBLIC = os.environ.get("PUBLIC_MODE", "0") == "1"
auth = Auth.from_env()          # None = no password (local development only); PUBLIC_MODE refuses to start without one

app = FastAPI(title="Barbershop voice assistant (prototype)", docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(make_router(auth, ROOT / "static" / "login.html"))
app.include_router(lab_router)
app.include_router(recorder_router)
app.add_middleware(AuthMiddleware, auth=auth, public_mode=PUBLIC)

@app.on_event("startup")
async def _warm_voice_cache():
    if PUBLIC:
        return
    from .agent import T, LANG_NAMES
    texts = [agent.greeting(l) for l in LANG_NAMES]
    for l in LANG_NAMES:
        texts += [v for k, v in T[l].items() if "{" not in v]
    asyncio.create_task(tts.warm(texts))


_hits = defaultdict(lambda: deque(maxlen=120))


def _rate_limit(request: Request, per_minute: int = 40) -> None:
    ip = request.client.host if request.client else "?"
    q, now = _hits[ip], time.time()
    while q and q[0] < now - 60:
        q.popleft()
    if len(q) >= per_minute:
        raise HTTPException(429, "slow down")
    q.append(now)


class ChatIn(BaseModel):
    session_id: str = Field(max_length=64)
    text: str = Field(max_length=2000)


class TtsIn(BaseModel):
    text: str = Field(max_length=800)


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "live.html")


@app.get("/classic")
def classic():
    """The older turn-by-turn version (record, think, then speak). Kept for comparison."""
    return FileResponse(ROOT / "static" / "index.html")


@app.websocket("/ws/call")
async def ws_call(ws: WebSocket):
    if llm is None or getattr(llm, "name", "") != "gemini":
        await ws.accept()
        await ws.close(code=1011)
        return
    ip = auth.client_ip({k.decode().lower(): v.decode() for k, v in ws.scope["headers"]}, ws.client and (ws.client.host, 0)) if auth else None
    await handle_call(ws, agent, (not PUBLIC) and os.environ.get("DEBUG_UI", "1") == "1", ip)


@app.get("/api/calls")
def calls(limit: int = 20):
    """Recent calls with their event trail and the reason each one ended (memory only; resets when the server restarts)."""
    return list(CALL_LOG)[-max(1, min(limit, 40)):][::-1]


@app.get("/api/archive")
def archive_status():
    """Is saving call records to GitHub switched on, and did the last attempt work? (no secrets are shown)"""
    return archive.status()


@app.post("/api/archive/check")
def archive_check():
    """Try the GitHub connection right now and say exactly what is wrong, if anything."""
    return archive.check()


@app.get("/api/config")
def config():
    return {"public": PUBLIC}


@app.get("/api/health")
def health():
    return {"llm": llm.name if llm else None, "llm_error": llm_error,
            "calendar": type(calendar).__name__, "shop": shop.data["name"]["en"]}


@app.post("/api/start")
def start(request: Request, lang: str = "ka"):
    _rate_limit(request, 10)
    lang = lang if lang in ("ka", "en", "ru") else "ka"
    s = agent.new_session()
    s.lang = lang
    return {"session_id": s.id, "text": agent.greeting(lang), "lang": lang}


@app.post("/api/chat")
def chat(body: ChatIn, request: Request):
    _rate_limit(request)
    s = agent.get(body.session_id)
    if not s:
        raise HTTPException(404, "unknown session")
    if llm is None:
        return {"text": f"Assistant is not configured: {llm_error}", "lang": "en", "booked": False}
    t0 = time.time()
    r = agent.handle(s, body.text)
    print(f"[timing] chat {time.time() - t0:.1f}s", flush=True)
    return r.model_dump()


@app.post("/api/stt")
async def stt(request: Request):
    """Raw audio in (wav/mp3/ogg/flac), transcript out. Transcript is untrusted caller text like any other."""
    _rate_limit(request, 60)
    if llm is None or not hasattr(llm, "transcribe"):
        raise HTTPException(503, "speech-to-text needs LLM_PROVIDER=gemini")
    data = await request.body()
    mime = request.headers.get("content-type", "audio/wav").split(";")[0]
    if mime not in ("audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/flac"):
        raise HTTPException(415, "unsupported audio type")
    if not (1000 < len(data) < 2_000_000):
        raise HTTPException(413, "audio too short or too long")
    t0 = time.time()
    try:
        text = await run_in_threadpool(llm.transcribe, data, mime)
    except LLMError as e:
        print(f"[stt error] {e}", flush=True)
        raise HTTPException(502, "speech service error")
    print(f"[timing] stt {time.time() - t0:.1f}s  {len(data) // 1000} KB  -> {text[:60]!r}", flush=True)
    if text.strip().lower() in ("[silence]", "silence", ""):
        text = ""
    return {"text": text[:400]}


@app.post("/api/tts")
async def speak(body: TtsIn, request: Request):
    _rate_limit(request, 60)
    try:
        audio = await tts.synthesize(body.text)
    except Exception as e:  # network/voice service problem: the page falls back to the browser voice
        return JSONResponse({"error": str(e)[:200]}, status_code=502)
    return Response(content=audio, media_type="audio/mpeg")


@app.get("/api/bookings")
def bookings():
    """Dev-only view of the booking ledger (no login!). Disable with DEBUG_UI=0."""
    if not PUBLIC and os.environ.get("DEBUG_UI", "1") != "1":
        raise HTTPException(404)
    if not actions.ledger.exists():
        return []
    rows = [json.loads(x) for x in actions.ledger.read_text(encoding="utf-8").splitlines()]
    rows = rows[-20:][::-1]
    if PUBLIC:  # testers see their own test bookings, but never full phone numbers
        for r in rows:
            r["phone"] = "+995*****" + r["phone"][-3:]
    return rows
