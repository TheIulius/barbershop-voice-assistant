"""Local-only voice recorder for benchmarks: you read a list of phrases, each recording is saved as a 16 kHz WAV in
data/voice_samples/ (git-ignored, never uploaded by this code). Disabled in PUBLIC_MODE and for non-local clients."""
from __future__ import annotations

import os
import wave
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

router = APIRouter()
ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "voice_samples"

PHRASES = [  # (category, text, digits_if_a_number)
    ("short", "ეგრე იყოს", ""), ("short", "დაიცა რა", ""), ("short", "კარგია", ""), ("short", "ორშაბათს", ""), ("short", "ნიკა", ""),
    ("sentence", "მინდა თმის შეჭრა ხვალ სამ საათზე", ""), ("sentence", "ხუთშაბათს თორმეტ საათზე შეიძლება", ""),
    ("sentence", "ფასი რამდენია წვერის მოწესრიგებაზე", ""), ("sentence", "პარკინგი გაქვთ", ""),
    ("name", "ილია შინჯიაშვილი", ""), ("name", "ნინო ჭყონია", ""), ("name", "ლევან თულაშვილი", ""), ("name", "ზურაბ ქარჩავა", ""),
    ("number", "ჩემი ნომერია ხუთას ორმოცდათხუთმეტი ას ოცდასამი ორმოცდაექვსი", "55512346"),
    ("number", "ხუთი ხუთი ხუთი ერთი ორი სამი ოთხი ხუთი ექვსი", "555123456"),
]


def _local_only(request: Request) -> None:
    if os.environ.get("PUBLIC_MODE", "0") == "1":
        raise HTTPException(404)
    host = request.client.host if request.client else ""
    if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
        raise HTTPException(403, "the recorder only works on the computer that runs the server")


@router.get("/record")
def page(request: Request):
    _local_only(request)
    return FileResponse(ROOT / "static" / "record.html")


@router.get("/api/record/list")
def listing(request: Request):
    _local_only(request)
    saved = {int(p.stem) for p in SAMPLES.glob("*.wav")} if SAMPLES.exists() else set()
    return {"phrases": [{"i": i, "category": c, "text": t, "saved": i in saved} for i, (c, t, _) in enumerate(PHRASES)]}


@router.post("/api/record/{idx}")
async def save(idx: int, request: Request):
    _local_only(request)
    if not (0 <= idx < len(PHRASES)):
        raise HTTPException(404)
    data = await request.body()
    if not (3000 < len(data) < 3_000_000):
        raise HTTPException(413, "recording too short or too long")
    SAMPLES.mkdir(parents=True, exist_ok=True)
    f = SAMPLES / f"{idx:02d}.wav"
    f.write_bytes(data)
    try:
        with wave.open(str(f), "rb") as w:
            ok = w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2
            secs = w.getnframes() / 16000
    except wave.Error:
        ok, secs = False, 0
    if not ok:
        f.unlink()
        raise HTTPException(415, "expected 16 kHz mono 16-bit WAV")
    return {"saved": idx, "seconds": round(secs, 1)}


@router.get("/api/record/audio/{idx}")
def audio(idx: int, request: Request):
    _local_only(request)
    f = SAMPLES / f"{idx:02d}.wav"
    if not (0 <= idx < len(PHRASES)) or not f.exists():
        raise HTTPException(404)
    return Response(f.read_bytes(), media_type="audio/wav")
