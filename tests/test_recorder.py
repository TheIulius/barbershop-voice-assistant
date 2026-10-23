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



# ---------------- public mode: recordings go to the GitHub data branch ----------------------------------------------
@pytest.fixture
def pub(monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "1")
    store = {}
    monkeypatch.setattr(rec.archive, "enabled", lambda: True)
    monkeypatch.setattr(rec.archive, "put_file", lambda path, data, msg: store.__setitem__(path, data))
    monkeypatch.setattr(rec.archive, "list_dir", lambda path: sorted(p.split("/")[-1] for p in store))
    monkeypatch.setattr(rec.archive, "get_file", lambda path: store.get(path))
    rec._saves.clear()
    return TestClient(app, client=("203.0.113.9", 5000)), store     # a remote client: fine in public mode (login gate is separate)


def test_public_mode_saves_to_the_data_branch_not_to_disk(pub, tmp_path, monkeypatch):
    monkeypatch.setattr(rec, "SAMPLES", tmp_path)
    c, store = pub
    r = c.post("/api/record/4", content=wav_bytes(), headers={"content-type": "audio/wav"})
    assert r.status_code == 200 and r.json()["where"] == "github data branch"
    assert "voice_samples/04.wav" in store and not list(tmp_path.glob("*.wav"))
    items = c.get("/api/record/list").json()["phrases"]
    assert items[4]["saved"] and not items[3]["saved"]
    assert c.get("/api/record/audio/4").content == store["voice_samples/04.wav"]


def test_public_mode_without_a_token_says_so(monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "1")
    monkeypatch.setattr(rec.archive, "enabled", lambda: False)
    monkeypatch.delenv("ARCHIVE_GITHUB_TOKEN", raising=False)
    c = TestClient(app, client=("203.0.113.9", 5000))
    assert "not configured" in c.get("/api/record/list").json()["note"]
    assert c.post("/api/record/0", content=wav_bytes(), headers={"content-type": "audio/wav"}).status_code == 502


def test_public_mode_limits_saves_per_hour(pub):
    c, store = pub
    for _ in range(60):
        assert c.post("/api/record/0", content=wav_bytes(), headers={"content-type": "audio/wav"}).status_code == 200
    assert c.post("/api/record/0", content=wav_bytes(), headers={"content-type": "audio/wav"}).status_code == 429


def test_github_failure_is_reported_not_swallowed(pub, monkeypatch):
    c, store = pub
    def boom(*a, **k): raise RuntimeError("cannot write file: http 403 Resource not accessible")
    monkeypatch.setattr(rec.archive, "put_file", boom)
    r = c.post("/api/record/0", content=wav_bytes(), headers={"content-type": "audio/wav"})
    assert r.status_code == 502 and "403" in r.json()["detail"]


def test_archive_file_helpers_use_the_data_branch(monkeypatch):
    import base64
    from tests.test_archive import FakeGitHub, Resp
    from app import archive
    monkeypatch.setenv("ARCHIVE_GITHUB_TOKEN", "t0ken"); monkeypatch.setenv("ARCHIVE_REPO", "me/logs"); monkeypatch.setenv("ARCHIVE_BRANCH", "call-logs")
    archive._branch_ready["ok"] = True

    class GH(FakeGitHub):
        def get(self, url, **kw):
            self.calls.append(("GET", url, kw.get("params")))
            if "/contents/voice_samples/07.wav" in url: return Resp(200, {"sha": "oldsha"})
            return Resp(404)
    gh = GH(branch_exists=True)
    monkeypatch.setattr(archive, "requests", gh)
    archive.put_file("voice_samples/07.wav", b"RIFFdata", "sample")
    put = [c for c in gh.calls if c[0] == "PUT"][0]
    assert put[2]["branch"] == "call-logs" and put[2]["sha"] == "oldsha" and base64.b64decode(put[2]["content"]) == b"RIFFdata"
