import json
import base64

import pytest

from app import archive


class Resp:
    def __init__(self, code, body=None):
        self.status_code, self._b = code, body or {}

    def json(self):
        return self._b


class FakeGitHub:
    """Stands in for the `requests` module: records calls, simulates a repo whose log branch does not exist yet."""

    def __init__(self, branch_exists=False, put_codes=(201,)):
        self.calls, self.branch_exists, self.put_codes = [], branch_exists, list(put_codes)

    def get(self, url, **kw):
        self.calls.append(("GET", url))
        if url.endswith("/git/ref/heads/call-logs"):
            return Resp(200 if self.branch_exists else 404)
        if url.endswith("/git/ref/heads/main"):
            return Resp(200, {"object": {"sha": "abc123"}})
        if url.endswith("/repos/me/logs"):
            return Resp(200, {"default_branch": "main"})
        return Resp(404)

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw["json"]))
        self.branch_exists = True
        return Resp(201)

    def put(self, url, **kw):
        self.calls.append(("PUT", url, kw["json"]))
        return Resp(self.put_codes.pop(0) if len(self.put_codes) > 1 else self.put_codes[0])


SUMMARY = {"id": "ab12cd34", "started_utc": "2026-10-24 14:05:09", "seconds": 61.2, "reason": "caller_left",
           "trail": [{"t": 3, "what": "caller", "text": "ნომერია 555 123 457 და სახელი ნინო"}]}


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("ARCHIVE_GITHUB_TOKEN", "t0ken")
    monkeypatch.setenv("ARCHIVE_REPO", "me/logs")
    monkeypatch.setenv("ARCHIVE_BRANCH", "call-logs")
    monkeypatch.setattr(archive.time, "sleep", lambda s: None)
    archive._branch_ready["ok"] = False


def test_disabled_without_settings(monkeypatch):
    monkeypatch.delenv("ARCHIVE_GITHUB_TOKEN")
    assert not archive.enabled() and archive.archive_call(SUMMARY) is None


def test_creates_branch_then_commits_one_file_per_call(monkeypatch):
    gh = FakeGitHub()
    monkeypatch.setattr(archive, "requests", gh)
    archive._put(SUMMARY)
    kinds = [c[0] + " " + c[1].split("github.com")[1] for c in gh.calls]
    assert any(k.startswith("POST /repos/me/logs/git/refs") for k in kinds)          # branch created from main
    put = [c for c in gh.calls if c[0] == "PUT"][0]
    assert put[1].endswith("/contents/calls/2026-10-24/140509-ab12cd34.json") and put[2]["branch"] == "call-logs"
    saved = json.loads(base64.b64decode(put[2]["content"]))
    assert saved["id"] == "ab12cd34"


def test_phone_numbers_are_masked_before_upload(monkeypatch):
    gh = FakeGitHub(branch_exists=True)
    monkeypatch.setattr(archive, "requests", gh)
    archive._put(SUMMARY)
    put = [c for c in gh.calls if c[0] == "PUT"][0]
    text = base64.b64decode(put[2]["content"]).decode()
    assert "555 123 457" not in text and "•••" in text and "ნინო" in text         # digits hidden, the rest kept


def test_github_trouble_never_raises(monkeypatch, capsys):
    gh = FakeGitHub(branch_exists=True, put_codes=(500,))
    monkeypatch.setattr(archive, "requests", gh)
    archive._put(SUMMARY)                                   # must not raise
    assert "FAILED" in capsys.readouterr().out
    assert len([c for c in gh.calls if c[0] == "PUT"]) == 3  # retried


def test_retry_then_success(monkeypatch, capsys):
    gh = FakeGitHub(branch_exists=True, put_codes=(502, 201))
    monkeypatch.setattr(archive, "requests", gh)
    archive._put(SUMMARY)
    assert "saved calls/2026-10-24" in capsys.readouterr().out


def test_masking_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("ARCHIVE_MASK_NUMBERS", "0")
    assert archive.scrub(SUMMARY)["trail"][0]["text"] == SUMMARY["trail"][0]["text"]
