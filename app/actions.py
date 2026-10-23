"""The ONLY code that can change anything. The language model can merely
*request* one of these actions; every argument is validated here, and the
model's text is never used for state changes or confirmations."""
from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .calendars import Calendar
from .shop import Booking, Reject, Shop
from .textutil import normalize_phone, valid_name

MAX_BOOKINGS_PER_SESSION = 2
MAX_FUTURE_BOOKINGS_PER_PHONE = 3
MAX_ACTIONS_PER_SESSION = 30
MAX_CALLBACKS_PER_SESSION = 1
MAX_TIMES_LISTED = 5


@dataclass
class Session:
    id: str
    lang: str = "ka"
    history: list = field(default_factory=list)  # [{"role": "user"|"assistant", "text": str}]
    pending: Optional[Booking] = None
    bookings_made: int = 0
    callbacks: int = 0
    actions: int = 0
    turns: int = 0
    last_seen: float = 0.0


class Actions:
    def __init__(self, shop: Shop, calendar: Calendar, data_dir: Path):
        self.shop = shop
        self.cal = calendar
        self.dir = Path(data_dir)
        self.ledger = self.dir / "bookings.jsonl"
        self.callbacks_file = self.dir / "callbacks.jsonl"
        self.lock = threading.Lock()

    # --- helpers -------------------------------------------------------
    def _count(self, s: Session) -> None:
        s.actions += 1
        if s.actions > MAX_ACTIONS_PER_SESSION:
            raise Reject("session_limit")

    def _barbers(self, barber: str) -> List[str]:
        self.shop.check_barber(barber)
        return list(self.shop.barbers) if barber == "any" else [barber]

    def _free(self, barber: str, service: str, day, now: datetime) -> List[datetime]:
        starts = self.shop.starts_for(barber, service, day, now)
        if not starts:
            return []
        dur = timedelta(minutes=self.shop.services[service]["minutes"])
        busy = self.cal.busy(barber, self.shop.at(day, datetime.min.time()), self.shop.at(day, datetime.min.time()) + timedelta(days=1))
        return [s for s in starts if not any(s < be and s + dur > bs for bs, be in busy)]

    def _future_by_phone(self, phone: str, now: datetime) -> int:
        if not self.ledger.exists():
            return 0
        n = 0
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["phone"] == phone and datetime.fromisoformat(r["start"]) > now:
                n += 1
        return n

    # --- actions -------------------------------------------------------
    def check_availability(self, s: Session, service: str, barber: str, date_s: str, now: datetime,
                           time_s: str = "") -> str:
        self._count(s)
        self.shop.check_service(service)
        day = self.shop.check_date(date_s, now)
        times = set()
        for b in self._barbers(barber):
            times.update(t.strftime("%H:%M") for t in self._free(b, service, day, now))
        times = sorted(times)
        head = f"{day.isoformat()} {service}"
        if not times:
            return f"{head}: NO free times at all that day (closed, day off, or fully booked)"
        want = None
        try:
            want = self.shop.check_time(time_s).strftime("%H:%M") if time_s else None
        except Reject:
            want = None  # ranges or garbage: ignore and just list times
        if want:
            if want in times:
                return f"{head}: {want} IS FREE. All free times that day: {', '.join(times)}"
            near = sorted(times, key=lambda t: abs(_mins(t) - _mins(want)))[:3]
            return f"{head}: {want} is NOT free. Nearest free times: {', '.join(sorted(near))}"
        if len(times) <= MAX_TIMES_LISTED:
            return f"{head}: free times: {', '.join(times)} (complete list)"
        shown = _spread(times, MAX_TIMES_LISTED)
        return (f"{head}: {len(times)} free times between {times[0]} and {times[-1]}. Sample: {', '.join(shown)}. "
                f"This is only a sample: other times may be free; to test a time the caller asks for, "
                f"call check_availability again with that time")

    def propose(self, s: Session, service: str, barber: str, date_s: str, time_s: str,
                name: str, phone: str, now: datetime) -> Booking:
        self._count(s)
        svc = self.shop.check_service(service)
        day = self.shop.check_date(date_s, now)
        t = self.shop.check_time(time_s)
        nm = valid_name(name)
        if not nm:
            raise Reject("bad_name")
        ph = normalize_phone(phone)
        if not ph:
            raise Reject("bad_phone")
        if s.bookings_made >= MAX_BOOKINGS_PER_SESSION:
            raise Reject("session_limit")
        if self._future_by_phone(ph, now) >= MAX_FUTURE_BOOKINGS_PER_PHONE:
            raise Reject("too_many_for_phone")
        start = self.shop.at(day, t)
        chosen = None
        for b in self._barbers(barber):
            if start in self._free(b, service, day, now):
                chosen = b
                break
        if not chosen:
            raise Reject("slot_taken")
        b = Booking(id=secrets.token_hex(4), barber=chosen, service=service, start=start,
                    end=start + timedelta(minutes=svc["minutes"]), name=nm, phone=ph)
        s.pending = b
        return b

    def confirm(self, s: Session, now: datetime) -> Booking:
        self._count(s)
        b = s.pending
        if b is None:
            raise Reject("nothing_pending")
        with self.lock:
            day = b.start.date()
            if b.start not in self._free(b.barber, b.service, day, now):
                s.pending = None
                raise Reject("slot_taken")
            self.cal.create(b, self.shop)
            self.dir.mkdir(parents=True, exist_ok=True)
            with self.ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "id": b.id, "barber": b.barber, "service": b.service, "start": b.start.isoformat(),
                    "end": b.end.isoformat(), "name": b.name, "phone": b.phone, "created": now.isoformat(),
                }, ensure_ascii=False) + "\n")
        s.pending = None
        s.bookings_made += 1
        return b

    def cancel_pending(self, s: Session) -> None:
        self._count(s)
        s.pending = None

    def request_callback(self, s: Session, name: str, phone: str, note: str, now: datetime) -> None:
        self._count(s)
        if s.callbacks >= MAX_CALLBACKS_PER_SESSION:
            raise Reject("session_limit")
        ph = normalize_phone(phone)
        if not ph:
            raise Reject("bad_phone")
        nm = valid_name(name) or ""
        note = " ".join((note or "").split())[:200]
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.lock, self.callbacks_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": now.isoformat(), "name": nm, "phone": ph, "note": note}, ensure_ascii=False) + "\n")
        s.callbacks += 1


def _spread(items: List[str], n: int) -> List[str]:
    """Pick n items spread across the list (so the caller hears morning/afternoon/evening)."""
    if len(items) <= n:
        return items
    step = (len(items) - 1) / (n - 1)
    return [items[round(i * step)] for i in range(n)]


def _mins(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)
