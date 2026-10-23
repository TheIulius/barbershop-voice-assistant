"""Calendar backends. The agent never talks to these directly; only the
server-side `Actions` class does, after validation."""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from .shop import Booking, Reject, Shop

Interval = Tuple[datetime, datetime]


def event_id(barber: str, start: datetime) -> str:
    """Deterministic id (hex only, valid for Google): the same barber+start can
    never be created twice."""
    return hashlib.sha1(f"{barber}|{start.isoformat()}".encode()).hexdigest()


class Calendar:
    def busy(self, barber: str, start: datetime, end: datetime) -> List[Interval]:
        raise NotImplementedError

    def create(self, b: Booking, shop: Shop) -> None:
        raise NotImplementedError


class LocalCalendar(Calendar):
    """JSON-file calendar. Works with zero setup; used for tests and demos."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def _read(self) -> list:
        if not self.path.exists():
            return []
        return json.loads(self.path.read_text(encoding="utf-8"))

    def busy(self, barber, start, end):
        out = []
        for e in self._read():
            if e["barber"] != barber:
                continue
            s, f = datetime.fromisoformat(e["start"]), datetime.fromisoformat(e["end"])
            if s < end and f > start:
                out.append((s, f))
        return out

    def create(self, b, shop):
        with self.lock:
            events = self._read()
            eid = event_id(b.barber, b.start)
            if any(e["id"] == eid for e in events):
                raise Reject("slot_taken")
            events.append({
                "id": eid, "barber": b.barber, "service": b.service,
                "start": b.start.isoformat(), "end": b.end.isoformat(),
                "name": b.name, "phone": b.phone,
            })
            self.path.write_text(json.dumps(events, ensure_ascii=False, indent=1), encoding="utf-8")


class GoogleCalendar(Calendar):
    """Google Calendar through a service account (REST, no heavy client lib).

    Setup: create a service account, download its JSON key, create one Google
    Calendar per barber and share each with the service account's email using
    'Make changes to events'. Put each calendar's ID in shop.json."""

    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    BASE = "https://www.googleapis.com/calendar/v3"

    def __init__(self, key_file: str, shop: Shop):
        from google.oauth2 import service_account
        from google.auth.transport.requests import AuthorizedSession

        creds = service_account.Credentials.from_service_account_file(key_file, scopes=self.SCOPES)
        self.http = AuthorizedSession(creds)
        self.shop = shop

    def _cal(self, barber: str) -> str:
        cid = self.shop.barbers[barber].get("calendar_id", "")
        if not cid:
            raise Reject("calendar_not_configured", barber)
        return cid

    def busy(self, barber, start, end):
        cid = self._cal(barber)
        r = self.http.post(f"{self.BASE}/freeBusy", json={
            "timeMin": start.isoformat(), "timeMax": end.isoformat(),
            "timeZone": self.shop.data["timezone"], "items": [{"id": cid}],
        }, timeout=10)
        r.raise_for_status()
        cal = r.json()["calendars"][cid]
        if cal.get("errors"):
            raise Reject("calendar_error", str(cal["errors"]))
        return [(datetime.fromisoformat(x["start"].replace("Z", "+00:00")),
                 datetime.fromisoformat(x["end"].replace("Z", "+00:00"))) for x in cal.get("busy", [])]

    def create(self, b, shop):
        cid = self._cal(b.barber)
        tz = shop.data["timezone"]
        name_en = shop.services[b.service]["en"]
        body = {
            "id": event_id(b.barber, b.start),
            "summary": f"{name_en} - {b.name}",
            "description": f"Booked by AI assistant\nPhone: {b.phone}\nBooking ID: {b.id}",
            "start": {"dateTime": b.start.isoformat(), "timeZone": tz},
            "end": {"dateTime": b.end.isoformat(), "timeZone": tz},
            "extendedProperties": {"private": {"source": "ai-assistant", "booking_id": b.id}},
        }
        r = self.http.post(f"{self.BASE}/calendars/{cid}/events", json=body, timeout=10)
        if r.status_code == 409:
            raise Reject("slot_taken")
        r.raise_for_status()
