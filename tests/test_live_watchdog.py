"""The call must never hang up silently or nudge the caller while the assistant is busy."""
import asyncio
import json
import time

import pytest

from app.live import LiveCall
from app.agent import Agent
from app.actions import Actions
from app.calendars import LocalCalendar
from app.shop import Shop
from pathlib import Path


class FakeWS:
    def __init__(self):
        self.events = []

    async def send_text(self, t):
        self.events.append(json.loads(t))


class FakeG:
    def __init__(self):
        self.sent = []

    async def send(self, t):
        self.sent.append(json.loads(t))


@pytest.fixture
def call(tmp_path):
    root = Path(__file__).resolve().parent.parent
    shop = Shop.load(root / "data" / "shop.json")
    agent = Agent(shop, Actions(shop, LocalCalendar(tmp_path / "c.json"), tmp_path), llm=None)
    c = LiveCall(FakeWS(), agent, "ka", False)
    return c


async def run_watchdog(call, g, seconds):
    t = asyncio.create_task(call.watchdog(g))
    await asyncio.sleep(seconds)
    done = t.done()
    t.cancel()
    return done


def texts(g):
    return [m["realtimeInput"]["text"] for m in g.sent if "realtimeInput" in m]


def test_model_silent_after_tool_result_gets_nudged_to_answer(call):
    g = FakeG()
    call.pending_since = time.time() - 9
    assert not asyncio.run(run_watchdog(call, g, 1.5))
    assert any("lookup finished" in t for t in texts(g))
    assert not any("still there" in t for t in texts(g))       # must NOT ask the caller if they are there


def test_model_that_never_answers_ends_call_with_visible_error(call):
    g = FakeG()
    call.pending_since = time.time() - 30
    assert asyncio.run(run_watchdog(call, g, 1.5))
    assert any(e["type"] == "error" and "stopped answering" in e["text"] for e in call.ws.events)


def test_no_nudge_while_assistant_is_talking_or_playing(call):
    g = FakeG()
    call.waiting_for_caller = True
    call.last_activity = time.time() - 60
    call.playing = True
    assert not asyncio.run(run_watchdog(call, g, 1.5))
    assert texts(g) == []


def test_no_nudge_when_it_is_not_the_callers_turn(call):
    g = FakeG()
    call.waiting_for_caller = False         # e.g. the assistant is thinking
    call.last_activity = time.time() - 60
    assert not asyncio.run(run_watchdog(call, g, 1.5))
    assert texts(g) == []


def test_caller_silence_gets_gentle_prompt_then_hangup(call):
    g = FakeG()
    call.waiting_for_caller = True
    call.last_activity = call.last_voice_at = time.time() - 11
    asyncio.run(run_watchdog(call, g, 1.5))
    assert any("still there" in t for t in texts(g))
    call.nudges = 2
    call.last_activity = call.last_voice_at = time.time() - 11
    assert asyncio.run(run_watchdog(call, g, 1.5))
    assert any(e["type"] == "hangup" for e in call.ws.events)
    assert call.end_reason == "caller_silent_after_2_prompts"


def test_unexpected_google_disconnect_is_reported(call):
    import websockets

    class ClosingG:
        close_code = 1011
        close_reason = "internal"

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration
    asyncio.run(call.from_gemini(ClosingG()))
    assert any(e["type"] == "error" and "1011" in e["text"] for e in call.ws.events)


def test_sound_from_the_caller_counts_as_presence_no_hangup(call):
    g = FakeG()
    call.waiting_for_caller = True
    call.nudges = 2
    call.last_activity = time.time() - 60
    call.last_voice_at = time.time()          # the caller is making sound right now
    assert not asyncio.run(run_watchdog(call, g, 1.5))
    assert texts(g) == [] and not any(e["type"] == "hangup" for e in call.ws.events)


def test_noisy_line_gets_a_quieter_spot_request_not_a_hangup(call):
    g = FakeG()
    call.waiting_for_caller = True
    call.loud.extend([True] * 200)             # loud audio the whole time, but no words recognised
    call.last_user_text_at = time.time() - 20
    call.last_activity = time.time() - 60
    assert not asyncio.run(run_watchdog(call, g, 1.5))
    assert any("background noise" in t for t in texts(g))
    assert not any("still there" in t for t in texts(g))
    assert not any(e["type"] == "hangup" for e in call.ws.events)


def test_line_that_stays_too_noisy_ends_with_a_clear_message(call):
    g = FakeG()
    call.waiting_for_caller = True
    call.loud.extend([True] * 200)
    call.last_user_text_at = time.time() - 60
    call.noise_notes = 3
    call.last_noise_note = time.time() - 30
    assert asyncio.run(run_watchdog(call, g, 1.5))
    assert any(e["type"] == "error" and "too noisy" in e["text"] for e in call.ws.events)
    assert call.end_reason == "noisy_line_no_speech_recognised"


def test_call_summary_records_reason_and_trail(call, capsys):
    from app import live
    live.CALL_LOG.clear()
    call.note("caller", text="გამარჯობა")
    call.end_reason = "caller_left (browser closed or connection dropped)"
    call._record_call()
    rec = live.CALL_LOG[-1]
    assert rec["reason"].startswith("caller_left") and rec["trail"][0]["what"] == "caller"
    assert "[calllog]" in capsys.readouterr().out
