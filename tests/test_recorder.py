import io
import wave

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.recorder as rec

# an isolated app with only the recorder, so other tests' public-mode settings cannot interfere
app = FastAPI()
app.include_router(rec.router)


def wav_bytes(seconds=1.0, rate=16000, channels=1):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(rate); w.writeframes(b"\x01\x00" * int(rate * seconds) * channels)
    return b.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "SAMPLES", tmp_path)
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    return TestClient(app)


def test_save_list_and_play_back(client, tmp_path):
    r = client.post("/api/record/3", content=wav_bytes(), headers={"content-type": "audio/wav"})
    assert r.status_code == 200 and r.json()["saved"] == 3
    items = client.get("/api/record/list").json()["phrases"]
    assert items[3]["saved"] is True and items[2]["saved"] is False and len(items) == len(rec.PHRASES)
    assert client.get("/api/record/audio/3").status_code == 200


@pytest.mark.parametrize("bad", [wav_bytes(rate=44100), wav_bytes(channels=2), b"x" * 5000, b"tiny"])
def test_wrong_format_or_size_is_refused_and_not_kept(client, tmp_path, bad):
    assert client.post("/api/record/1", content=bad, headers={"content-type": "audio/wav"}).status_code in (413, 415)
    assert not list(tmp_path.glob("*.wav"))


def test_index_out_of_range_and_path_tricks(client):
    assert client.post("/api/record/99", content=wav_bytes()).status_code == 404
    assert client.post("/api/record/-1", content=wav_bytes()).status_code == 404
    assert client.get("/api/record/audio/..%2F..%2Fsecret").status_code in (404, 422)


def test_not_reachable_from_other_computers(client):
    c = TestClient(app, client=("203.0.113.9", 5000))
    assert c.get("/api/record/list").status_code == 403
    assert c.post("/api/record/0", content=wav_bytes()).status_code == 403


def test_the_phone_number_phrases_carry_the_digits_to_compare():
    nums = [(t, d) for c, t, d in rec.PHRASES if c == "number"]
    assert len(nums) == 2 and all(d.isdigit() for _, d in nums)
