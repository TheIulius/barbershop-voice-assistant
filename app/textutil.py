"""Untrusted-text handling: sanitising caller input, cleaning model output,
validating the few free-text fields (name, phone) that reach the calendar."""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

MAX_USER_CHARS = 400
MAX_REPLY_CHARS = 320

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f​-‏‪-‮⁦-⁩﻿]")
_URL = re.compile(r"(https?://\S+|www\.\S+|\S+@\S+\.\S+)", re.I)
_MARKDOWN = re.compile(r"[*_`#>|~\[\]{}<>\\]+")
_SPACES = re.compile(r"\s+")


def sanitize_user_text(text: str, tag: str) -> str:
    """Normalise caller text. Removes invisible/control characters and any text
    that imitates our internal tool-result marker, then truncates."""
    text = unicodedata.normalize("NFC", text or "")
    text = _CONTROL.sub(" ", text)
    text = text.replace(tag, " ")
    text = re.sub(r"<<\s*TR[-\w]*\s*>>", " ", text, flags=re.I)
    text = _SPACES.sub(" ", text).strip()
    return text[:MAX_USER_CHARS]


def detect_lang(text: str, default: str = "ka") -> str:
    """Pick ka / ru / en from the script of the text (deterministic, no model)."""
    ka = ru = en = 0
    for ch in text:
        o = ord(ch)
        if 0x10D0 <= o <= 0x10FF or 0x1C90 <= o <= 0x1CBF:
            ka += 1
        elif 0x0400 <= o <= 0x04FF:
            ru += 1
        elif ch.isascii() and ch.isalpha():
            en += 1
    best = max((ka, "ka"), (ru, "ru"), (en, "en"))
    return best[1] if best[0] > 0 else default


def clean_reply(text: str, canary: str, forbidden_fragments: Optional[list] = None) -> Optional[str]:
    """Make a model reply safe to speak. Returns None if it must be discarded."""
    if not isinstance(text, str):
        return None
    low = text.lower()
    if canary.lower() in low:
        return None
    for frag in forbidden_fragments or []:
        if frag.lower() in low:
            return None
    text = _CONTROL.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _MARKDOWN.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    if not text:
        return None
    if len(text) > MAX_REPLY_CHARS:
        cut = text[:MAX_REPLY_CHARS]
        end = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), cut.rfind("።"))
        text = cut[: end + 1] if end > 40 else cut.rstrip()
    return text


def valid_name(raw: str) -> Optional[str]:
    """1-3 words, letters only (any script), each at most 20 chars."""
    raw = _SPACES.sub(" ", _CONTROL.sub(" ", raw or "")).strip()
    words = raw.split(" ")
    if not (1 <= len(words) <= 3):
        return None
    for w in words:
        core = w.replace("-", "").replace("'", "")
        if not (1 <= len(w) <= 20) or not core or not all(c.isalpha() for c in core):
            return None
    return " ".join(words)


def normalize_phone(raw: str) -> Optional[str]:
    """Return +995XXXXXXXXX for plausible Georgian mobile/landline numbers."""
    digits = re.sub(r"[ \-().]", "", raw or "")
    if digits.startswith("+995"):
        digits = digits[4:]
    elif digits.startswith("00995"):
        digits = digits[5:]
    elif digits.startswith("995") and len(digits) == 12:
        digits = digits[3:]
    if not digits.isdigit() or len(digits) != 9:
        return None
    if digits.startswith("5") or digits.startswith("32"):
        return "+995" + digits
    return None
