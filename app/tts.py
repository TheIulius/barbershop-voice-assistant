"""Text-to-speech with Microsoft Edge neural voices via the unofficial `edge-tts`
package. FREE and fine for local testing, but NOT for production: it is an
undocumented endpoint that can change or stop without notice. For paying
customers switch to Azure AI Speech (same voices, official API)."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import edge_tts

from .textutil import detect_lang

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "tts_cache"

import os

VOICES = {
    "ka": os.environ.get("TTS_VOICE_KA", "ka-GE-EkaNeural"),    # or ka-GE-GiorgiNeural (male)
    "ru": os.environ.get("TTS_VOICE_RU", "ru-RU-SvetlanaNeural"),
    "en": os.environ.get("TTS_VOICE_EN", "en-US-JennyNeural"),
}
RATE = os.environ.get("TTS_RATE", "+22%")  # speaking speed; the default voice is slow for a phone call
MAX_CHARS = 400
_cache: dict = {}


async def synthesize(text: str) -> bytes:
    text = (text or "").strip()[:MAX_CHARS]
    if not text:
        return b""
    voice = VOICES[detect_lang(text, "ka")]
    key = (voice, RATE, text)
    if key in _cache:
        return _cache[key]
    f = CACHE_DIR / (hashlib.sha1(f"{voice}|{RATE}|{text}".encode()).hexdigest() + ".mp3")
    if f.exists():
        _cache[key] = f.read_bytes()
        return _cache[key]
    audio = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate=RATE).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
    if len(_cache) > 200:
        _cache.clear()
    _cache[key] = bytes(audio)
    if audio and len(text) < 200:  # fixed phrases are reused: keep them on disk (instant next time)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        f.write_bytes(bytes(audio))
    return bytes(audio)


async def warm(texts) -> None:
    """Pre-render the fixed phrases in the background so they play instantly."""
    sem = asyncio.Semaphore(2)

    async def one(t):
        async with sem:
            try:
                await synthesize(t)
            except Exception:
                pass
    await asyncio.gather(*(one(t) for t in texts))
