"""Password gate for the whole app (pages, API and the live-call WebSocket).

Design:
  * the password is never stored: only a salted PBKDF2-SHA256 hash (600,000 rounds) (APP_PASSWORD_HASH);
  * a correct login gives a signed, expiring session cookie (HttpOnly, SameSite=Strict, Secure on https);
  * everything except /login and /healthz needs that cookie, including the WebSocket;
  * wrong guesses are throttled per client and globally, with a 1 s delay each;
  * same-origin checks on WebSockets and on POSTs, strict security headers, nothing cacheable.
This makes online guessing impractical; it cannot protect a password that is short or easy to guess,
so use a long passphrase for anything that matters.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from collections import defaultdict, deque
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

COOKIE = "bs_session"
SESSION_SECONDS = 12 * 3600
MAX_FAILS_PER_IP = 5
LOCK_SECONDS = 15 * 60
MAX_FAILS_GLOBAL = 40
GLOBAL_LOCK_SECONDS = 5 * 60

PUBLIC_PATHS = {"/login", "/healthz"}
# In public mode these developer/test routes do not exist at all (they would only burn your tokens).
PUBLIC_BLOCKED = ("/classic", "/lab", "/api/lab", "/record", "/api/record", "/api/start", "/api/chat", "/api/stt", "/api/tts")

CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline' blob:; style-src 'self' 'unsafe-inline'; "
       "connect-src 'self' ws: wss:; img-src 'self' data:; media-src 'self' blob:; worker-src 'self' blob:; "
       "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


ITERATIONS = 600_000  # PBKDF2-HMAC-SHA256, the OWASP-recommended work factor; stdlib, works on every Python build


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations, dklen=32)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iters, salt, dk = stored.split("$")
        if scheme != "pbkdf2_sha256" or not (1000 <= int(iters) <= 5_000_000):
            return False
        calc = hashlib.pbkdf2_hmac("sha256", password.encode(), _unb64(salt), int(iters), dklen=32)
        return hmac.compare_digest(calc, _unb64(dk))
    except Exception:
        return False


class Auth:
    def __init__(self, password_hash: str, secret: str, public_mode: bool, trust_proxy: bool):
        self.hash = password_hash
        self.secret = secret.encode()
        self.public = public_mode
        self.trust_proxy = trust_proxy
        self.fails = defaultdict(deque)       # ip -> failure timestamps
        self.global_fails = deque()

    @classmethod
    def from_env(cls) -> Optional["Auth"]:
        h = os.environ.get("APP_PASSWORD_HASH", "")
        public = os.environ.get("PUBLIC_MODE", "0") == "1"
        if not h:
            if public:
                raise RuntimeError("PUBLIC_MODE=1 requires APP_PASSWORD_HASH (refusing to start without a password)")
            return None
        secret = os.environ.get("SESSION_SECRET", "")
        if len(secret) < 32:
            raise RuntimeError("SESSION_SECRET must be a random string of at least 32 characters")
        return cls(h, secret, public, os.environ.get("TRUST_PROXY", "0") == "1")

    # ---- sessions ---------------------------------------------------------
    def make_token(self) -> str:
        payload = _b64(json.dumps({"exp": int(time.time()) + SESSION_SECONDS, "n": secrets.token_hex(8)}).encode())
        sig = _b64(hmac.new(self.secret, payload.encode(), hashlib.sha256).digest())
        return f"{payload}.{sig}"

    def valid_token(self, token: str) -> bool:
        try:
            payload, sig = token.split(".")
            good = _b64(hmac.new(self.secret, payload.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(sig, good):
                return False
            return json.loads(_unb64(payload))["exp"] > time.time()
        except Exception:
            return False

    def cookie_ok(self, cookie_header: str) -> bool:
        for part in (cookie_header or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == COOKIE and self.valid_token(v):
                return True
        return False

    # ---- client identity / throttling ----------------------------------------
    def client_ip(self, headers: dict, client) -> str:
        if self.trust_proxy:
            xff = headers.get("x-forwarded-for", "")
            if xff:
                return xff.split(",")[-1].strip()   # the entry added by our own proxy, not client-supplied ones
        return client[0] if client else "?"

    def locked(self, ip: str) -> int:
        """Seconds the caller must wait, or 0."""
        now = time.time()
        q = self.fails[ip]
        while q and q[0] < now - LOCK_SECONDS:
            q.popleft()
        g = self.global_fails
        while g and g[0] < now - GLOBAL_LOCK_SECONDS:
            g.popleft()
        if len(q) >= MAX_FAILS_PER_IP:
            return int(q[0] + LOCK_SECONDS - now) + 1
        if len(g) >= MAX_FAILS_GLOBAL:
            return int(g[0] + GLOBAL_LOCK_SECONDS - now) + 1
        return 0

    def record_fail(self, ip: str) -> None:
        now = time.time()
        self.fails[ip].append(now)
        self.global_fails.append(now)


# ---------------------------------------------------------------------------------
# ASGI middleware: runs before every route, for HTTP and WebSocket alike
# ---------------------------------------------------------------------------------
class AuthMiddleware:
    def __init__(self, app, auth: Optional[Auth], public_mode: bool):
        self.app, self.auth, self.public = app, auth, public_mode

    @staticmethod
    def _headers(scope) -> dict:
        return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}

    def _same_origin(self, headers: dict) -> bool:
        origin = headers.get("origin")
        if not origin:
            return True
        return urlparse(origin).netloc == headers.get("host", "")

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        headers = self._headers(scope)

        if scope["type"] == "websocket":
            ok = (self.auth is None or self.auth.cookie_ok(headers.get("cookie", ""))) and self._same_origin(headers)
            if self.public and path.startswith(PUBLIC_BLOCKED):
                ok = False
            if not ok:
                await receive()                       # the websocket.connect message
                await send({"type": "websocket.close", "code": 4401})
                return
            return await self.app(scope, receive, send)

        async def reply(status: int, body: bytes = b"", ctype: str = "application/json", extra=()):
            await send({"type": "http.response.start", "status": status,
                        "headers": self._sec_headers(headers, [(b"content-type", ctype.encode()), *extra])})
            await send({"type": "http.response.body", "body": body})

        if self.public and path.startswith(PUBLIC_BLOCKED):
            return await reply(404, b'{"detail":"Not Found"}')
        if scope["method"] not in ("GET", "HEAD", "OPTIONS") and not self._same_origin(headers):
            return await reply(403, b'{"detail":"cross-site request refused"}')

        if self.auth is not None and path not in PUBLIC_PATHS and not self.auth.cookie_ok(headers.get("cookie", "")):
            if path.startswith("/api") or scope["method"] != "GET":
                return await reply(401, b'{"detail":"login required"}')
            return await reply(303, b"", "text/plain", extra=[(b"location", b"/login")])

        async def send_wrapped(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = self._sec_headers(headers, list(message.get("headers", [])), replace=True)
            await send(message)
        await self.app(scope, receive, send_wrapped)

    def _sec_headers(self, req_headers: dict, hdrs: list, replace: bool = False) -> list:
        have = {k.lower() for k, _ in hdrs}
        extra = [
            (b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"),
            (b"referrer-policy", b"no-referrer"), (b"permissions-policy", b"microphone=(self), camera=(), geolocation=()"),
            (b"content-security-policy", CSP.encode()), (b"cache-control", b"no-store"),
            (b"cross-origin-opener-policy", b"same-origin"),
        ]
        if req_headers.get("x-forwarded-proto") == "https":
            extra.append((b"strict-transport-security", b"max-age=31536000"))
        return hdrs + [(k, v) for k, v in extra if k not in have]


# ---------------------------------------------------------------------------------
# /login and /logout
# ---------------------------------------------------------------------------------
def make_router(auth: Optional[Auth], login_page_path) -> APIRouter:
    router = APIRouter()

    @router.get("/login")
    def login_page():
        from fastapi.responses import FileResponse
        return FileResponse(login_page_path)

    @router.get("/healthz")
    def healthz():
        return Response("ok", media_type="text/plain")

    @router.post("/login")
    async def login(request: Request):
        if auth is None:
            return JSONResponse({"ok": True})
        ip = auth.client_ip(dict(request.headers), (request.client.host, 0) if request.client else None)
        wait = auth.locked(ip)
        if wait:
            return JSONResponse({"ok": False, "error": f"Too many wrong attempts. Try again in {wait // 60 + 1} min."}, status_code=429,
                                headers={"Retry-After": str(wait)})
        try:
            raw = await request.body()
            pw = str(json.loads(raw[:2000]).get("password", ""))[:200] if raw else ""
        except Exception:
            pw = ""
        ok = await asyncio.to_thread(verify_password, pw, auth.hash)
        if not ok:
            auth.record_fail(ip)
            await asyncio.sleep(1.0)
            return JSONResponse({"ok": False, "error": "Wrong password."}, status_code=401)
        secure = request.headers.get("x-forwarded-proto", request.url.scheme) == "https"
        resp = JSONResponse({"ok": True})
        resp.set_cookie(COOKIE, auth.make_token(), max_age=SESSION_SECONDS, httponly=True, secure=secure,
                        samesite="strict", path="/")
        return resp

    @router.post("/logout")
    def logout():
        resp = JSONResponse({"ok": True})
        resp.delete_cookie(COOKIE, path="/")
        return resp

    return router
