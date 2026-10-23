"""Voice lab: hear the same sentence in different text-to-speech engines.
Dev tool only (disabled when DEBUG_UI=0). Not part of the assistant itself."""
from __future__ import annotations

import base64
import io
import os
import time
import wave
from pathlib import Path
from typing import Optional

import edge_tts
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

router = APIRouter()
ROOT = Path(__file__).resolve().parent.parent

EDGE_VOICES = {
    "ka": ["ka-GE-EkaNeural", "ka-GE-GiorgiNeural"],
    "ru": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
    "en": ["en-US-JennyNeural", "en-US-AriaNeural", "en-US-GuyNeural", "en-GB-SoniaNeural"],
}
GEMINI_VOICES = ["Kore", "Puck", "Zephyr", "Aoede", "Charon", "Leda", "Fenrir", "Sulafat"]
SAMPLES = {
    "ka": "გამარჯობა! ხვალ ხუთ საათზე თავისუფალი გვაქვს. გსურთ, დაგიჯავშნოთ თმის შეჭრა გიორგისთან? ფასი ოცდაათი ლარია.",
    "ru": "Здравствуйте! Завтра в пять часов есть свободное время. Хотите записаться на стрижку к Гиорги? Стоимость тридцать лари.",
    "en": "Hello! We have a free slot tomorrow at five o'clock. Would you like to book a haircut with Giorgi? It costs thirty lari.",
}


class LabIn(BaseModel):
    engine: str = Field(pattern="^(edge|gemini|eleven)$")
    voice: str = Field(max_length=80)
    text: str = Field(max_length=400)
    rate: str = Field(default="+0%", max_length=8)


def _wav(pcm: bytes, rate: int = 24000) -> bytes:
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return b.getvalue()


async def _edge(voice: str, text: str, rate: str) -> tuple:
    if voice not in sum(EDGE_VOICES.values(), []):
        raise HTTPException(400, "unknown voice")
    if len(rate) > 6 or not rate[1:-1].isdigit() or rate[0] not in "+-" or rate[-1] != "%":
        raise HTTPException(400, "bad rate")
    audio = bytearray()
    async for c in edge_tts.Communicate(text, voice, rate=rate).stream():
        if c["type"] == "audio":
            audio += c["data"]
    return bytes(audio), "audio/mpeg"


async def _gemini(voice: str, text: str) -> tuple:
    if voice not in GEMINI_VOICES:
        raise HTTPException(400, "unknown voice")
    key = os.environ.get("GEMINI_API_KEY", "")
    model = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")
    body = {"contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}}}
    async with httpx.AsyncClient(timeout=40) as c:
        r = await c.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                         json=body, headers={"x-goog-api-key": key})
    if r.status_code != 200:
        raise HTTPException(502, f"Gemini TTS http {r.status_code} (free quota? wait a minute)")
    pcm = base64.b64decode(r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
    return _wav(pcm), "audio/wav"


async def _eleven(voice: str, text: str) -> tuple:
    key = os.environ.get("ELEVENLABS_API_KEY", "")
    if not key:
        raise HTTPException(503, "Set ELEVENLABS_API_KEY in .env to try ElevenLabs")
    model = os.environ.get("ELEVENLABS_MODEL", "eleven_v3")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
                         json={"text": text, "model_id": model}, headers={"xi-api-key": key})
    if r.status_code != 200:
        raise HTTPException(502, f"ElevenLabs http {r.status_code}: {r.text[:140]}")
    return r.content, "audio/mpeg"


def _enabled():
    if os.environ.get("DEBUG_UI", "1") != "1":
        raise HTTPException(404)


@router.get("/lab")
def page():
    _enabled()
    return FileResponse(ROOT / "static" / "voices.html")


@router.get("/api/lab/options")
def options():
    _enabled()
    return {"edge": EDGE_VOICES, "gemini": GEMINI_VOICES, "samples": SAMPLES,
            "eleven_configured": bool(os.environ.get("ELEVENLABS_API_KEY")),
            "eleven_voice": os.environ.get("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            "gemini_model": os.environ.get("GEMINI_TTS_MODEL", "gemini-3.8-flash-lite-tts")}


@router.post("/api/lab/tts")
async def lab_tts(body: LabIn):
    _enabled()
    t0 = time.time()
    if body.engine == "edge":
        audio, mime = await _edge(body.voice, body.text, body.rate)
    elif body.engine == "gemini":
        audio, mime = await _gemini(body.voice, body.text)
    else:
        audio, mime = await _eleven(body.voice, body.text)
    return Response(content=audio, media_type=mime, headers={"X-Gen-Seconds": f"{time.time() - t0:.1f}"})
