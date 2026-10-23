"""LLM connectors. Both return ONE JSON object (constrained by a schema) so the
model can only *choose an action*; it never calls code directly."""
from __future__ import annotations

import copy
import os
import time
from typing import Dict, List

import httpx

ACTIONS = ["none", "check_availability", "book", "confirm", "cancel_pending", "request_callback"]

_STR = {"type": "string"}
SCHEMA = {
    "type": "object",
    "properties": {
        "reply": _STR,
        "action": {"type": "string", "enum": ACTIONS},
        "service": _STR, "barber": _STR, "date": _STR, "time": _STR,
        "customer_name": _STR, "customer_phone": _STR, "note": _STR,
    },
    # every field is required so the model must fill (or blank) each one instead of silently dropping the date
    "required": ["reply", "action", "service", "barber", "date", "time", "customer_name", "customer_phone", "note"],
}


def _gemini_schema(node):
    """Gemini wants OpenAPI-style uppercase type names."""
    node = copy.deepcopy(node)

    def walk(n):
        if isinstance(n, dict):
            if "type" in n and isinstance(n["type"], str):
                n["type"] = n["type"].upper()
            for v in n.values():
                walk(v)
    walk(node)
    return node


class LLMError(Exception):
    pass


def thinking_config(model: str):
    """Turn model 'thinking' off/minimal: it adds seconds of delay and can eat the output budget."""
    if "2.5" in model or "3.5" in model:
        return {"thinkingBudget": 0}
    if "gemini-3" in model:
        return {"thinkingLevel": "minimal"}
    return None


class GeminiLLM:
    """Gemini over plain REST. Free-tier quotas differ per model and run out, so every call walks a
    chain of models: when one answers 429/503 (or times out) it is put on a short cool-down and the
    next model is tried immediately."""
    name = "gemini"
    COOLDOWN_S = 120

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.key, self.model = api_key, model
        self.http = httpx.Client(timeout=25)
        self.tokens_in = self.tokens_out = self.calls = 0  # running totals, to measure real cost
        self.cooldown: Dict[str, float] = {}
        self.last_model = ""

    # -- model chains -----------------------------------------------------
    def _chain(self, primary: str, extra_env: str, default_extra: str) -> List[str]:
        extra = [m.strip() for m in os.environ.get(extra_env, default_extra).split(",") if m.strip()]
        chain, seen = [], set()
        for m in [primary] + extra:
            if m and m not in seen:
                chain.append(m); seen.add(m)
        return chain

    def _generate(self, models: List[str], make_body, what: str) -> dict:
        errors = []
        for round_ in range(2):
            live = [m for m in models if self.cooldown.get(m, 0) < time.time()] or models
            for m in live:
                try:
                    r = self.http.post(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent",
                                       json=make_body(m), headers={"x-goog-api-key": self.key})
                except httpx.HTTPError as e:
                    errors.append(f"{m}: {type(e).__name__}")
                    self.cooldown[m] = time.time() + 20
                    continue
                if r.status_code == 200:
                    self.last_model = m
                    u = r.json().get("usageMetadata", {})
                    self.calls += 1
                    self.tokens_in += u.get("promptTokenCount", 0)
                    self.tokens_out += u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)
                    return r.json()
                errors.append(f"{m}: http {r.status_code}")
                if r.status_code in (429, 500, 503, 403, 404):
                    self.cooldown[m] = time.time() + (self.COOLDOWN_S if r.status_code in (429, 404) else 20)
                    continue
                raise LLMError(f"{what}: gemini http {r.status_code}: {r.text[:300]}")
            if round_ == 0:
                time.sleep(1.5)
        raise LLMError(f"{what}: all models failed ({'; '.join(errors)})")

    # -- chat (returns one JSON action) ------------------------------------
    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        def body(model):
            gen = {"temperature": 0.2, "maxOutputTokens": 700, "responseMimeType": "application/json",
                   "responseSchema": _gemini_schema(SCHEMA)}
            tc = thinking_config(model)
            if tc:
                gen["thinkingConfig"] = tc  # faster, cheaper for phone use
            return {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["text"]}]}
                                 for m in messages],
                    "generationConfig": gen}
        data = self._generate(self._chain(self.model, "GEMINI_FALLBACK_MODELS", "gemini-3-flash-preview"), body, "chat")
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            raise LLMError(f"gemini returned no text: {str(data)[:300]}")

    # -- speech to text -----------------------------------------------------
    def transcribe(self, audio: bytes, mime: str) -> str:
        """Speech-to-text with Gemini's audio input (ka / ru / en, language auto-detected)."""
        import base64
        prompt = ("Transcribe this phone-call audio exactly as spoken. The speaker uses Georgian, Russian or English. "
                  "Write Georgian in Georgian script and Russian in Cyrillic. Write numbers as digits. "
                  "Output ONLY the transcript. If there is no clear human speech, output exactly: [silence]")
        b64 = base64.b64encode(audio).decode()

        def body(model):
            gen = {"temperature": 0, "maxOutputTokens": 600}
            tc = thinking_config(model)
            if tc:
                gen["thinkingConfig"] = tc
            return {"contents": [{"role": "user", "parts": [{"text": prompt}, {"inlineData": {"mimeType": mime, "data": b64}}]}],
                    "generationConfig": gen}
        primary = os.environ.get("GEMINI_STT_MODEL") or "gemini-3-flash-preview"
        data = self._generate(self._chain(primary, "GEMINI_STT_FALLBACK_MODELS", "gemini-flash-lite-latest"), body, "stt")
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            return ""


class OllamaLLM:
    name = "ollama"

    def __init__(self, url: str, model: str):
        if not model:
            raise LLMError("OLLAMA_MODEL is not set")
        self.url, self.model = url.rstrip("/"), model
        self.http = httpx.Client(timeout=120)

    def complete(self, system: str, messages: List[Dict[str, str]]) -> str:
        body = {
            "model": self.model,
            "stream": False,
            "format": SCHEMA,
            "options": {"temperature": 0.2},
            "messages": [{"role": "system", "content": system}]
            + [{"role": m["role"], "content": m["text"]} for m in messages],
        }
        try:
            r = self.http.post(f"{self.url}/api/chat", json=body)
        except httpx.HTTPError as e:
            raise LLMError(f"cannot reach Ollama at {self.url}: {e}")
        if r.status_code != 200:
            raise LLMError(f"ollama http {r.status_code}: {r.text[:300]}")
        return r.json()["message"]["content"]


def make_llm():
    provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        return GeminiLLM(os.environ.get("GEMINI_API_KEY", ""), os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    if provider == "ollama":
        return OllamaLLM(os.environ.get("OLLAMA_URL", "http://localhost:11434"), os.environ.get("OLLAMA_MODEL", ""))
    raise LLMError(f"unknown LLM_PROVIDER {provider!r}")
