"""The password gate must hold for EVERY route, for HTTP and the live-call WebSocket alike."""
import os
import time

import pytest

os.environ["PUBLIC_MODE"] = "1"
os.environ["TRUST_PROXY"] = "0"
os.environ["SESSION_SECRET"] = "x" * 48
from app.auth import hash_password  # noqa: E402

PASSWORD = "correct horse battery staple"
os.environ["APP_PASSWORD_HASH"] = hash_password(PASSWORD, iterations=2000)

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

import app.main as main  # noqa: E402
import app.live as live  # noqa: E402


@pytest.fixture(autouse=True)
def fresh():
    main.auth.fails.clear()
    main.auth.global_fails.clear()
    live._budget.update({"day": "", "seconds": 0.0, "active": 0})
    live._active_by_ip.clear()
    live._recent_calls.clear()
    yield


@pytest.fixture
def client():
    return TestClient(main.app, follow_redirects=False)


def login(client, pw=PASSWORD):
    return client.post("/login", json={"password": pw})


PROTECTED_GET = ["/api/calls", "/", "/classic", "/lab", "/api/health", "/api/bookings", "/api/config", "/api/lab/options", "/docs", "/openapi.json", "/static/live.html"]
PROTECTED_POST = ["/api/start", "/api/chat", "/api/stt", "/api/tts", "/api/lab/tts", "/logout-not-a-route"]


@pytest.mark.parametrize("path", PROTECTED_GET)
def test_every_get_route_needs_login(client, path):
    r = client.get(path)
    assert r.status_code in (303, 401, 404), (path, r.status_code)
    if r.status_code == 303:
        assert r.headers["location"] == "/login"
    assert "Sample Barbershop" not in r.text


@pytest.mark.parametrize("path", PROTECTED_POST)
def test_every_post_route_needs_login(client, path):
    assert client.post(path, json={"text": "x", "session_id": "x"}).status_code in (401, 404)


def test_login_page_and_health_are_public_but_leak_nothing(client):
    assert client.get("/login").status_code == 200
    r = client.get("/healthz")
    assert r.status_code == 200 and r.text == "ok"


def test_wrong_password_rejected_then_locked_out(client):
    for _ in range(5):
        assert login(client, "wrong").status_code == 401
    r = login(client, PASSWORD)          # even the RIGHT password is refused while locked
    assert r.status_code == 429 and "Retry-After" in r.headers
    assert COOKIE_NAME not in r.headers.get("set-cookie", "")


COOKIE_NAME = "bs_session"


def test_correct_password_sets_hardened_cookie_and_unlocks(client):
    r = login(client)
    assert r.status_code == 200 and r.json()["ok"]
    sc = r.headers["set-cookie"].lower()
    assert "httponly" in sc and "samesite=strict" in sc and "path=/" in sc
    assert client.get("/api/config").json() == {"public": True}
    assert client.get("/").status_code == 200


def test_cookie_secure_flag_behind_https_proxy(client):
    r = client.post("/login", json={"password": PASSWORD}, headers={"x-forwarded-proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()


def test_tampered_and_expired_cookies_rejected(client):
    login(client)
    good = client.cookies.get(COOKIE_NAME)
    client.cookies.clear()
    for bad in (good[:-2] + "xx", "garbage", good.split(".")[0] + ".", ""):
        client.cookies.set(COOKIE_NAME, bad)
        assert client.get("/api/config").status_code == 401
    # a validly signed but expired token
    import base64, hashlib, hmac, json
    payload = base64.urlsafe_b64encode(json.dumps({"exp": int(time.time()) - 5, "n": "a"}).encode()).decode().rstrip("=")
    sig = base64.urlsafe_b64encode(hmac.new(b"x" * 48, payload.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    client.cookies.set(COOKIE_NAME, f"{payload}.{sig}")
    assert client.get("/api/config").status_code == 401


def test_cookie_signed_with_another_secret_is_rejected(client):
    import base64, hashlib, hmac, json
    payload = base64.urlsafe_b64encode(json.dumps({"exp": int(time.time()) + 999, "n": "a"}).encode()).decode().rstrip("=")
    sig = base64.urlsafe_b64encode(hmac.new(b"y" * 48, payload.encode(), hashlib.sha256).digest()).decode().rstrip("=")
    client.cookies.set(COOKIE_NAME, f"{payload}.{sig}")
    assert client.get("/api/config").status_code == 401


def test_developer_routes_do_not_exist_in_public_mode_even_when_logged_in(client):
    login(client)
    for path in ("/classic", "/lab", "/api/lab/options"):
        assert client.get(path).status_code == 404
    for path in ("/api/start", "/api/chat", "/api/stt", "/api/tts", "/api/lab/tts"):
        assert client.post(path, json={}).status_code == 404


def test_cross_site_post_refused_even_with_valid_cookie(client):
    login(client)
    r = client.post("/logout", headers={"origin": "https://evil.example"})
    assert r.status_code == 403


def test_security_headers_present(client):
    r = client.get("/login")
    h = {k.lower(): v for k, v in r.headers.items()}
    assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff"
    assert "default-src 'self'" in h["content-security-policy"] and h["cache-control"] == "no-store"
    assert h["referrer-policy"] == "no-referrer"


def test_bookings_masks_phone_numbers_in_public_mode(client, tmp_path, monkeypatch):
    ledger = tmp_path / "b.jsonl"
    ledger.write_text('{"id":"1","barber":"giorgi","service":"haircut","start":"2026-11-02T12:00:00+04:00","end":"x","name":"Nino","phone":"+995555123456"}\n')
    monkeypatch.setattr(main.actions, "ledger", ledger)
    login(client)
    rows = client.get("/api/bookings").json()
    assert rows[0]["phone"] == "+995*****456" and "555123" not in str(rows)


def test_login_rejects_oversized_and_malformed_bodies(client):
    assert client.post("/login", content=b"not json").status_code == 401
    assert client.post("/login", json={"password": "a" * 100000}).status_code == 401
    assert client.post("/login", json={}).status_code == 401


# ---------------- WebSocket (the live call) ------------------------------------------------
def test_websocket_refused_without_login(client):
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/ws/call?lang=ka"):
            pass
    assert e.value.code == 4401


def test_websocket_refused_with_forged_cookie(client):
    client.cookies.set(COOKIE_NAME, "forged.value")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/call?lang=ka"):
            pass


def test_websocket_refused_from_other_origin_even_when_logged_in(client):
    login(client)
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws/call?lang=ka", headers={"origin": "https://evil.example"}):
            pass


def test_websocket_accepted_with_login_and_same_origin(client, monkeypatch):
    async def fake(ws, agent, debug, ip=None):
        assert debug is False          # developer features are off in public mode
        await ws.accept()
        await ws.send_text('{"type":"ready"}')
        await ws.close()
    monkeypatch.setattr(main, "handle_call", fake)
    login(client)
    with client.websocket_connect("/ws/call?lang=ka", headers={"origin": "http://testserver"}) as ws:
        assert ws.receive_json()["type"] == "ready"


def test_daily_budget_stops_new_calls(client, monkeypatch):
    monkeypatch.setattr(live, "DAILY_CALL_MINUTES", 1.0)
    live._budget.update({"day": time.strftime("%Y-%m-%d", time.gmtime()), "seconds": 120.0})
    login(client)
    with client.websocket_connect("/ws/call?lang=ka", headers={"origin": "http://testserver"}) as ws:
        msg = ws.receive_json()
    assert msg["type"] == "error" and "budget" in msg["text"]


def test_concurrent_call_cap(client, monkeypatch):
    monkeypatch.setattr(live, "MAX_CONCURRENT_CALLS", 1)
    live._budget.update({"day": time.strftime("%Y-%m-%d", time.gmtime()), "active": 1})
    login(client)
    with client.websocket_connect("/ws/call?lang=ka", headers={"origin": "http://testserver"}) as ws:
        assert "right now" in ws.receive_json()["text"]


def test_public_mode_refuses_to_start_without_password(monkeypatch):
    from app.auth import Auth
    monkeypatch.delenv("APP_PASSWORD_HASH")
    with pytest.raises(RuntimeError):
        Auth.from_env()


def test_call_log_endpoint_needs_login_then_returns_list(client):
    assert client.get("/api/calls").status_code == 401
    login(client)
    r = client.get("/api/calls")
    assert r.status_code == 200 and isinstance(r.json(), list)
