"""The phone line must be impossible to use without Twilio's signature, and must convert audio correctly."""
import asyncio
import base64
import json
import time

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import g711, telephony
from app.auth import Auth, AuthMiddleware, hash_password

TOKEN = "unit-test-twilio-token"
BASE = "https://example.test"


# ---------------------------------------------------------------- audio
def test_ulaw_decoder_matches_the_standard_and_encoder_is_within_one_step():
    import audioop
    codes = bytes(range(256))
    assert g711.ulaw_to_pcm(codes).tobytes() == audioop.ulaw2lin(codes, 2)
    x = np.arange(-32768, 32768, dtype=np.int16)
    mine = np.frombuffer(g711.pcm_to_ulaw(x), np.uint8).astype(int)
    ref = np.frombuffer(audioop.lin2ulaw(x.tobytes(), 2), np.uint8).astype(int)
    assert np.abs(mine - ref).max() <= 1


def test_speech_survives_the_round_trip():
    t = np.arange(8000) / 8000
    x = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    back = g711.ulaw_to_pcm(g711.pcm_to_ulaw(x)).astype(float)
    assert np.sqrt(np.mean((back - x) ** 2)) / 8000 < 0.03          # < 3 % error


def test_resamplers_have_the_right_lengths():
    up = g711.Upsampler8to16()
    assert up(np.zeros(160, dtype=np.int16)).size == 320
    assert g711.down_24k_to_8k(np.zeros(720, dtype=np.int16)).size == 240


# ---------------------------------------------------------------- signature and tokens
def test_twilio_documented_signature_example():
    pairs = [("CallSid", "CA1234567890ABCDE"), ("Caller", "+14158675310"), ("Digits", "1234"),
             ("From", "+14158675310"), ("To", "+18005551212")]
    sig = telephony.twilio_signature("https://mycompany.com/myapp.php?foo=1&bar=2", pairs, "12345")
    assert sig == "GvWf1cFY/Q7PnoempGyD5oXAezc="


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", TOKEN)
    monkeypatch.setenv("PUBLIC_BASE_URL", BASE)
    monkeypatch.delenv("TELEPHONY_ALLOWED_TO", raising=False)
    telephony._used_tokens.clear()
    telephony._hits.clear()


def test_token_is_single_use_signed_and_expires():
    t = telephony.make_stream_token("+995555123456", "CA1")
    assert telephony.check_stream_token(t)["c"] == "+995555123456"
    assert telephony.check_stream_token(t) is None                                  # replay
    assert telephony.check_stream_token(telephony.make_stream_token("x", "y", ttl=-1)) is None
    good = telephony.make_stream_token("a", "b")
    assert telephony.check_stream_token(good[:-3] + "AAA") is None                  # tampered
    assert telephony.check_stream_token("garbage") is None


def test_token_from_another_account_is_refused(monkeypatch):
    t = telephony.make_stream_token("a", "b")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "someone-elses-token")
    assert telephony.check_stream_token(t) is None


def test_local_number():
    assert telephony.local_number("+995 555 12 34 56") == "555123456"
    assert telephony.local_number("+14158675310") == ""
    assert telephony.local_number("") == ""


# ---------------------------------------------------------------- routes (behind the real password gate)
calls = []


async def fake_handle_call(ws, agent, debug, ip):
    """Stands in for the live engine: echoes the first caller frame back as model audio, records what it saw."""
    await ws.accept()
    calls.append({"ip": ip, "lang": ws.query_params["lang"], "caller": ws.caller_number, "phone": ws.is_phone})
    got = []
    while len(got) < 3:
        m = await ws.receive()
        if m.get("type") == "websocket.disconnect":
            break
        if m.get("bytes") is not None:
            got.append(m["bytes"])
    calls[-1]["frames"] = got
    await ws.send_bytes(np.full(2400, 5000, dtype=np.int16).tobytes())            # 100 ms of tone at 24 kHz
    await ws.send_text(json.dumps({"type": "interrupted"}))
    await ws.send_text(json.dumps({"type": "hangup"}))


def build_app(agent=object()):
    app = FastAPI()
    app.include_router(telephony.make_router(lambda: agent, fake_handle_call))
    auth = Auth(hash_password("pw", iterations=1000), "s" * 40, True, False)
    app.add_middleware(AuthMiddleware, auth=auth, public_mode=True)
    return app


def signed_post(client, fields, token=TOKEN, url=BASE + "/telephony/voice"):
    pairs = list(fields.items())
    body = "&".join(f"{k}={v.replace('+', '%2B')}" for k, v in pairs)
    return client.post("/telephony/voice", content=body,
                       headers={"content-type": "application/x-www-form-urlencoded",
                                "x-twilio-signature": telephony.twilio_signature(url, pairs, token)})


def test_voice_webhook_needs_a_valid_signature():
    c = TestClient(build_app())
    assert c.post("/telephony/voice", content="From=%2B995555123456").status_code == 403
    r = signed_post(c, {"From": "+995555123456", "To": "+15005550006", "CallSid": "CA1"}, token="wrong-token")
    assert r.status_code == 403


def test_voice_webhook_answers_with_stream_twiml():
    c = TestClient(build_app())
    r = signed_post(c, {"From": "+995555123456", "To": "+15005550006", "CallSid": "CA1"})
    assert r.status_code == 200
    assert "<Stream url=\"wss://example.test/telephony/stream\"" in r.text and "<Connect>" in r.text and "<Hangup/>" in r.text
    assert "995555123456" not in r.text                     # the number travels only inside the signed token


def test_voice_webhook_refuses_other_numbers_when_a_list_is_set(monkeypatch):
    monkeypatch.setenv("TELEPHONY_ALLOWED_TO", "+15005550006")
    c = TestClient(build_app())
    ok = signed_post(c, {"From": "+995555123456", "To": "+15005550006", "CallSid": "CA1"})
    other = signed_post(c, {"From": "+995555123456", "To": "+19999999999", "CallSid": "CA2"})
    assert "<Stream" in ok.text and "<Reject/>" in other.text


def test_telephony_off_without_token(monkeypatch):
    monkeypatch.delenv("TWILIO_AUTH_TOKEN")
    assert TestClient(build_app()).post("/telephony/voice", content="").status_code == 503


def test_rest_of_the_site_still_needs_login():
    c = TestClient(build_app(), follow_redirects=False)
    assert c.get("/api/calls").status_code == 401


def start_msg(token, sid="MZ123"):
    return json.dumps({"event": "start", "streamSid": sid, "start": {"streamSid": sid, "customParameters": {"token": token}}})


def test_stream_refuses_without_a_token():
    c = TestClient(build_app())
    with c.websocket_connect("/telephony/stream") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(start_msg("nope"))
        with pytest.raises(WebSocketDisconnect) as e:
            ws.receive_text()
        assert e.value.code == 4401


def test_full_call_through_the_adapter():
    calls.clear()
    c = TestClient(build_app())
    tok = telephony.make_stream_token("+995555123456", "CA9")
    with c.websocket_connect("/telephony/stream") as ws:
        ws.send_text(json.dumps({"event": "connected"}))
        ws.send_text(start_msg(tok))
        ulaw = g711.pcm_to_ulaw(np.full(320, 1000, dtype=np.int16))                # 40 ms of caller audio
        for i in range(40):                                                         # > greeting-mute is not needed: fake engine never says turn_end
            ws.send_text(json.dumps({"event": "media", "media": {"track": "inbound", "payload": base64.b64encode(ulaw).decode()}}))
        out = []
        while True:
            try:
                out.append(json.loads(ws.receive_text()))
            except WebSocketDisconnect:
                break
            if out[-1].get("event") == "clear":
                break
    kinds = [m["event"] for m in out]
    assert kinds[:2] == ["media", "clear"]
    assert out[0]["streamSid"] == "MZ123"
    pcm8 = g711.ulaw_to_pcm(base64.b64decode(out[0]["media"]["payload"]))
    assert pcm8.size == 800 and abs(int(pcm8.mean()) - 5000) < 400                  # 100 ms at 8 kHz, level preserved
    seen = calls[0]
    assert seen["phone"] and seen["caller"] == "555123456" and seen["lang"] == "ka"
    assert seen["ip"].startswith("tel:") and "555123456" not in seen["ip"]
    assert all(len(f) == 1280 for f in seen["frames"])


def test_greeting_mutes_the_caller_until_it_has_played():
    async def run():
        class W:
            async def receive_text(self):
                await asyncio.sleep(10)
            async def send_text(self, t):
                pass
            async def close(self):
                pass
        p = telephony.PhoneSocket(W(), "MZ", "555123456", "ka")
        assert p._muted()                                   # greeting not finished
        await p.send_bytes(np.full(2400, 5000, dtype=np.int16).tobytes())
        await p.send_text(json.dumps({"type": "turn_end"}))
        assert p._muted()                                   # finished sending, but still playing for ~100 ms
        await asyncio.sleep(0.2)
        assert not p._muted()
        await p.close()
    asyncio.run(run())


def test_silence_padding_is_skipped_when_ahead_of_real_time():
    async def run():
        sent = []
        class W:
            async def receive_text(self):
                await asyncio.sleep(10)
            async def send_text(self, t):
                sent.append(json.loads(t))
            async def close(self):
                pass
        p = telephony.PhoneSocket(W(), "MZ", "", "ka")
        tone = np.full(24000, 5000, dtype=np.int16).tobytes()       # 1 s of speech: we are now 1 s ahead
        quiet = bytes(24000 * 2)                                     # 1 s of padding
        await p.send_bytes(tone)
        await p.send_bytes(quiet)
        assert len(sent) == 1                                        # padding dropped
        p.play_until = time.time()                                   # caught up: silence is real silence and must be kept
        await p.send_bytes(quiet)
        assert len(sent) == 2
        await p.close()
    asyncio.run(run())
