"""Keep call records after a server restart by committing each finished call to a GitHub repository.

Why: the free hosting plan loses local files and in-memory logs on every restart or sleep.
How: after each call a background thread writes one small JSON file through GitHub's contents API, on its own branch:
    calls/YYYY-MM-DD/HHMMSS-<id>.json
It never blocks or breaks a call: failures are printed to the server log and dropped.

Safety notes (read these before turning it on):
  * Use a PRIVATE repository that holds only logs. A token that can write to your *code* repository could be abused to
    push code that the host then deploys. Create a fine-grained token limited to that one logs repo, with
    "Contents: Read and write" and nothing else.
  * The records contain what callers said. Long digit sequences (phone numbers) are masked by default.
Settings (environment): ARCHIVE_GITHUB_TOKEN, ARCHIVE_REPO ("owner/name"), ARCHIVE_BRANCH (default "call-logs"),
ARCHIVE_MASK_NUMBERS (default "1").
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from typing import Optional

import requests

API = "https://api.github.com"
_branch_ready = {"ok": False}
_lock = threading.Lock()


def _cfg():
    return (os.environ.get("ARCHIVE_GITHUB_TOKEN", ""), os.environ.get("ARCHIVE_REPO", ""),
            os.environ.get("ARCHIVE_BRANCH", "call-logs"))


def enabled() -> bool:
    token, repo, _ = _cfg()
    return bool(token and repo and "/" in repo)


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "barbershop-call-archive"}


def mask(text: str) -> str:
    """Hide long digit runs (phone numbers) in free text."""
    return re.sub(r"(?<!\d)\d[\d\s\-]{4,}\d(?!\d)", "•••", text)


def scrub(summary: dict) -> dict:
    if os.environ.get("ARCHIVE_MASK_NUMBERS", "1") != "1":
        return summary
    out = json.loads(json.dumps(summary, ensure_ascii=False))
    for e in out.get("trail", []):
        for k, v in list(e.items()):
            if isinstance(v, str) and k in ("text", "name"):
                e[k] = mask(v)
    return out


def _ensure_branch(token: str, repo: str, branch: str) -> None:
    if _branch_ready["ok"]:
        return
    h = _headers(token)
    r = requests.get(f"{API}/repos/{repo}/git/ref/heads/{branch}", headers=h, timeout=15)
    if r.status_code == 200:
        _branch_ready["ok"] = True
        return
    if r.status_code != 404:
        raise RuntimeError(f"cannot check branch: http {r.status_code}")
    info = requests.get(f"{API}/repos/{repo}", headers=h, timeout=15)
    if info.status_code != 200:
        raise RuntimeError(f"cannot read repo: http {info.status_code} (token lacks access?)")
    default = info.json().get("default_branch", "main")
    base = requests.get(f"{API}/repos/{repo}/git/ref/heads/{default}", headers=h, timeout=15)
    if base.status_code == 404 or base.status_code == 409:
        # empty repository: create a first file so a branch exists, then branch from it
        c = requests.put(f"{API}/repos/{repo}/contents/README.md", headers=h, timeout=20,
                         json={"message": "init", "content": base64.b64encode(b"# call logs\n").decode()})
        if c.status_code not in (200, 201):
            raise RuntimeError(f"cannot initialise empty repo: http {c.status_code}")
        base = requests.get(f"{API}/repos/{repo}/git/ref/heads/{default}", headers=h, timeout=15)
    if base.status_code != 200:
        raise RuntimeError(f"cannot read default branch: http {base.status_code}")
    sha = base.json()["object"]["sha"]
    mk = requests.post(f"{API}/repos/{repo}/git/refs", headers=h, timeout=15, json={"ref": f"refs/heads/{branch}", "sha": sha})
    if mk.status_code not in (200, 201, 422):       # 422 = it already exists (another call created it first)
        raise RuntimeError(f"cannot create branch: http {mk.status_code}")
    _branch_ready["ok"] = True


def _put(summary: dict) -> None:
    token, repo, branch = _cfg()
    data = scrub(summary)
    path = f"calls/{data['started_utc'][:10]}/{data['started_utc'][11:].replace(':', '')}-{data['id']}.json"
    body = {"message": f"call {data['id']}: {data.get('reason', '')[:60]}",
            "content": base64.b64encode(json.dumps(data, ensure_ascii=False, indent=1).encode()).decode(), "branch": branch}
    last = ""
    for attempt in range(3):
        try:
            with _lock:
                _ensure_branch(token, repo, branch)
            r = requests.put(f"{API}/repos/{repo}/contents/{path}", headers=_headers(token), json=body, timeout=20)
            if r.status_code in (200, 201):
                print(f"[archive] saved {path}", flush=True)
                return
            last = f"http {r.status_code}"
        except Exception as e:  # network trouble, bad token, ...
            last = f"{type(e).__name__}: {str(e)[:100]}"
        time.sleep(2 * (attempt + 1))
    print(f"[archive] FAILED to save call {data['id']}: {last}", flush=True)


def archive_call(summary: dict) -> Optional[threading.Thread]:
    """Fire and forget: never raises, never blocks the call."""
    if not enabled():
        return None
    t = threading.Thread(target=_put, args=(summary,), daemon=True)
    t.start()
    return t
