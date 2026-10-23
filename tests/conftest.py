import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.actions import Actions  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.calendars import LocalCalendar  # noqa: E402
from app.shop import Shop  # noqa: E402

TBS = ZoneInfo("Asia/Tbilisi")
NOW = datetime(2026, 10, 26, 9, 0, tzinfo=TBS)  # a Monday, 09:00
TUESDAY = "2026-10-27"


class ScriptedLLM:
    """Stands in for the model. Each call pops the next scripted JSON action.
    Used to simulate a model that has been fully jailbroken."""

    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, system, messages):
        self.calls.append((system, messages))
        out = self.outputs.pop(0)
        return out if isinstance(out, str) else json.dumps(out)


@pytest.fixture
def shop():
    return Shop.load(ROOT / "data" / "shop.json")


@pytest.fixture
def env(shop, tmp_path):
    cal = LocalCalendar(tmp_path / "cal.json")
    actions = Actions(shop, cal, tmp_path)
    return shop, cal, actions, tmp_path


def make_agent(env, outputs):
    shop, cal, actions, _ = env
    llm = ScriptedLLM(outputs)
    return Agent(shop, actions, llm, now_fn=lambda: NOW), llm, cal


def book_args(**kw):
    a = {"reply": "", "action": "book", "service": "haircut", "barber": "giorgi", "date": TUESDAY,
         "time": "15:00", "customer_name": "Nino", "customer_phone": "555123456", "note": ""}
    a.update(kw)
    return a
