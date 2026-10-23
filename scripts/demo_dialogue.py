"""Scripted multi-turn booking conversations against the REAL model (throw-away calendar).
    .venv/bin/python scripts/demo_dialogue.py"""
import sys, tempfile, time
from datetime import timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.main import _load_env
_load_env()
from app.actions import Actions
from app.agent import Agent
from app.calendars import LocalCalendar
from app.llm import make_llm
from app.shop import Shop

CALLS = {
    "en": ["Hi, I'd like a haircut tomorrow with Giorgi", "Around 3 pm please", "My name is Nino and my number is 555 123 456", "Yes, that's right"],
    "ka": ["გამარჯობა, მინდა თმის შეჭრა ხვალ გიორგისთან", "15:00-ზე თუ შეიძლება", "მე ვარ ნინო, ნომერი 555123457", "დიახ, სწორია"],
    "ru": ["Здравствуйте, хочу записаться на стрижку завтра к Нике", "Лучше после обеда, в 16:00", "Меня зовут Анна, телефон 555 123 458", "Да, всё верно"],
    "info": ["How much is a haircut and where are you?", "Do you have parking?", "Are you open on Sunday?", "Do you take cards?"],
}
tmp = Path(tempfile.mkdtemp())
shop = Shop.load(ROOT / "data" / "shop.json")
cal = LocalCalendar(tmp / "cal.json")
llm = make_llm()
agent = Agent(shop, Actions(shop, cal, tmp), llm)
for name, lines in CALLS.items():
    print(f"\n===== {name} =====")
    s = agent.new_session()
    for ln in lines:
        time.sleep(3)
        r = agent.handle(s, ln)
        print(f"  caller: {ln}\n  bot   : {r.text}{'   [BOOKED]' if r.booked else ''}")
print("\ncalendar events:", [(e[0].isoformat(), b) for b in ('giorgi', 'nika') for e in cal.busy(b, shop.now(), shop.now() + timedelta(days=30))])

if hasattr(llm, 'calls'):
    print(f'\nmodel usage: {llm.calls} calls, {llm.tokens_in} input tokens, {llm.tokens_out} output tokens (4 conversations x 4 turns)')
