"""Shop configuration (the only source of business facts) and slot maths."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
LANGS = ("ka", "en", "ru")


class Reject(Exception):
    """A request the server refuses. `code` is a stable machine-readable reason."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(code)
        self.code = code
        self.detail = detail


def _hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


@dataclass(frozen=True)
class Booking:
    id: str
    barber: str
    service: str
    start: datetime
    end: datetime
    name: str
    phone: str


class Shop:
    def __init__(self, data: dict):
        self.data = data
        self.tz = ZoneInfo(data["timezone"])
        self.slot = int(data["slot_minutes"])
        self.lead = int(data["min_lead_minutes"])
        self.max_days = int(data["max_days_ahead"])
        self.services: Dict[str, dict] = data["services"]
        self.barbers: Dict[str, dict] = data["barbers"]
        self.hours: Dict[str, Optional[List[str]]] = data["hours"]

    @classmethod
    def load(cls, path: Path) -> "Shop":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def now(self) -> datetime:
        return datetime.now(self.tz)

    # --- opening hours -------------------------------------------------
    def day_window(self, barber_id: str, d: date) -> Optional[Tuple[time, time]]:
        key = DAYS[d.weekday()]
        h = self.hours.get(key)
        if not h or key in self.barbers[barber_id].get("days_off", []):
            return None
        return _hhmm(h[0]), _hhmm(h[1])

    def at(self, d: date, t: time) -> datetime:
        return datetime(d.year, d.month, d.day, t.hour, t.minute, tzinfo=self.tz)

    # --- validation (raise Reject) -------------------------------------
    def check_service(self, sid: str) -> dict:
        if sid not in self.services:
            raise Reject("bad_service")
        return self.services[sid]

    def check_barber(self, bid: str) -> None:
        if bid != "any" and bid not in self.barbers:
            raise Reject("bad_barber")

    def check_date(self, s: str, now: datetime) -> date:
        try:
            d = date.fromisoformat((s or "").strip())
        except ValueError:
            raise Reject("bad_date")
        today = now.date()
        if d < today or d > today + timedelta(days=self.max_days):
            raise Reject("date_out_of_range")
        return d

    def check_time(self, s: str) -> time:
        try:
            t = _hhmm((s or "").strip())
        except (ValueError, AttributeError):
            raise Reject("bad_time")
        if (t.hour * 60 + t.minute) % self.slot != 0:
            raise Reject("bad_time_grid")
        return t

    # --- availability --------------------------------------------------
    def starts_for(self, barber_id: str, service_id: str, d: date, now: datetime) -> List[datetime]:
        win = self.day_window(barber_id, d)
        if not win:
            return []
        dur = timedelta(minutes=self.services[service_id]["minutes"])
        out: List[datetime] = []
        cur = self.at(d, win[0])
        close = self.at(d, win[1])
        earliest = now + timedelta(minutes=self.lead)
        while cur + dur <= close:
            if cur >= earliest:
                out.append(cur)
            cur += timedelta(minutes=self.slot)
        return out

    # --- text for the prompt ------------------------------------------
    def info_block(self) -> str:
        d = self.data
        lines = [
            "name: " + " / ".join(d["name"][k] for k in LANGS),
            "address: " + " / ".join(d["address"][k] for k in LANGS),
            "public_phone: " + d.get("phone_public", ""),
            "opening_hours (Asia/Tbilisi): "
            + ", ".join(f"{k} {'closed' if not v else v[0] + '-' + v[1]}" for k, v in d["hours"].items()),
            "services (id: ka / en / ru, price GEL, minutes):",
        ]
        for sid, s in self.services.items():
            lines.append(f"  {sid}: {s['ka']} / {s['en']} / {s['ru']}, {s['price_gel']} GEL, {s['minutes']} min")
        lines.append("barbers (id: names, days off):")
        for bid, b in self.barbers.items():
            lines.append(f"  {bid}: {b['name']['ka']} / {b['name']['en']} / {b['name']['ru']}, off {','.join(b.get('days_off', [])) or 'none'}")
        lines.append("faq:")
        for f in d.get("faq", []):
            lines.append(f"  {f['topic']}: {f['en']} | {f['ka']} | {f['ru']}")
        return "\n".join(lines)
