"""Voice recorder for benchmarks: you read a list of phrases, each recording is a 16 kHz mono WAV.

Two modes:
  * local (your computer): saved to data/voice_samples/ (git-ignored); only reachable from this computer;
  * PUBLIC_MODE (the password-protected site): saved to the GitHub data branch (voice_samples/NN.wav) through the same
    token as the call records, because the host's disk is wiped on every restart. The login gate still applies.
"""
from __future__ import annotations

import os
import time
import wave
from collections import deque
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

from . import archive

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


_saves = deque(maxlen=200)
REMOTE_DIR = "voice_samples"


def _public() -> bool:
    return os.environ.get("PUBLIC_MODE", "0") == "1"


def _local_only(request: Request) -> None:
    """Local mode: only this computer. Public mode: the login middleware already protects every route."""
    if _public():
        return
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
    note = ""
    if _public():
        if not archive.enabled():
            saved, note = set(), "Saving is not configured on the server (ARCHIVE_REPO / ARCHIVE_GITHUB_TOKEN), so recordings cannot be kept."
        else:
            try:
                saved = {int(n[:-4]) for n in archive.list_dir(REMOTE_DIR) if n.endswith(".wav") and n[:-4].isdigit()}
            except Exception as e:
                saved, note = set(), f"Could not read the saved recordings: {e}"
    else:
        saved = {int(p.stem) for p in SAMPLES.glob("*.wav")} if SAMPLES.exists() else set()
    return {"note": note, "phrases": [{"i": i, "category": c, "text": t, "saved": i in saved} for i, (c, t, _) in enumerate(PHRASES)]}


@router.post("/api/record/{idx}")
async def save(idx: int, request: Request):
    _local_only(request)
    if not (0 <= idx < len(PHRASES)):
        raise HTTPException(404)
    data = await request.body()
    if not (3000 < len(data) < 3_000_000):
        raise HTTPException(413, "recording too short or too long")
    import io
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            ok = w.getnchannels() == 1 and w.getframerate() == 16000 and w.getsampwidth() == 2
            secs = w.getnframes() / 16000
    except (wave.Error, EOFError):
        ok, secs = False, 0
    if not ok or secs > 60:
        raise HTTPException(415, "expected a 16 kHz mono 16-bit WAV of at most 60 s")
    if _public():
        now = time.time()
        while _saves and _saves[0] < now - 3600:
            _saves.popleft()
        if len(_saves) >= 60:
            raise HTTPException(429, "too many recordings this hour")
        _saves.append(now)
        try:
            await run_in_threadpool(archive.put_file, f"{REMOTE_DIR}/{idx:02d}.wav", data, f"voice sample {idx:02d}: {PHRASES[idx][1][:40]}")
        except Exception as e:
            print(f"[recorder] could not save sample {idx}: {e}", flush=True)
            raise HTTPException(502, f"could not save to GitHub: {str(e)[:200]}")
    else:
        SAMPLES.mkdir(parents=True, exist_ok=True)
        (SAMPLES / f"{idx:02d}.wav").write_bytes(data)
    return {"saved": idx, "seconds": round(secs, 1), "where": "github data branch" if _public() else "this computer"}


@router.get("/api/record/audio/{idx}")
def audio(idx: int, request: Request):
    _local_only(request)
    if not (0 <= idx < len(PHRASES)):
        raise HTTPException(404)
    if _public():
        blob = archive.get_file(f"{REMOTE_DIR}/{idx:02d}.wav")
        if not blob:
            raise HTTPException(404)
        return Response(blob, media_type="audio/wav")
    f = SAMPLES / f"{idx:02d}.wav"
    if not f.exists():
        raise HTTPException(404)
    return Response(f.read_bytes(), media_type="audio/wav")
