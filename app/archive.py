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


STATUS = {"last_attempt_utc": "", "last_ok": None, "last_error": "", "saved": 0, "failed": 0}


def _cfg():
    token = os.environ.get("ARCHIVE_GITHUB_TOKEN", "").strip()
    repo = os.environ.get("ARCHIVE_REPO", "").strip()
    for junk in ("https://github.com/", "http://github.com/", "github.com/"):
        if repo.startswith(junk):
            repo = repo[len(junk):]
    repo = repo.strip("/").removesuffix(".git") if hasattr(str, "removesuffix") else repo.strip("/")
    if repo.endswith(".git"):
        repo = repo[:-4]
    branch = os.environ.get("ARCHIVE_BRANCH", "call-logs").strip() or "call-logs"
    return token, repo, branch


def _why(r) -> str:
    """HTTP status plus GitHub's own explanation (never contains our secrets)."""
    try:
        msg = r.json().get("message", "")
    except Exception:
        msg = ""
    return f"http {r.status_code} {msg}".strip()


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
        raise RuntimeError(f"cannot check branch: {_why(r)}")
    info = requests.get(f"{API}/repos/{repo}", headers=h, timeout=15)
    if info.status_code != 200:
        raise RuntimeError(f"cannot read repo {repo}: {_why(info)} (wrong name, or the token was not given access to this repository)")
    default = info.json().get("default_branch", "main")
    base = requests.get(f"{API}/repos/{repo}/git/ref/heads/{default}", headers=h, timeout=15)
    if base.status_code == 404 or base.status_code == 409:
        # empty repository: create a first file so a branch exists, then branch from it
        c = requests.put(f"{API}/repos/{repo}/contents/README.md", headers=h, timeout=20,
                         json={"message": "init", "content": base64.b64encode(b"# call logs\n").decode()})
        if c.status_code not in (200, 201):
            raise RuntimeError(f"cannot initialise empty repo: {_why(c)}")
        base = requests.get(f"{API}/repos/{repo}/git/ref/heads/{default}", headers=h, timeout=15)
    if base.status_code != 200:
        raise RuntimeError(f"cannot read default branch: {_why(base)}")
    sha = base.json()["object"]["sha"]
    mk = requests.post(f"{API}/repos/{repo}/git/refs", headers=h, timeout=15, json={"ref": f"refs/heads/{branch}", "sha": sha})
    if mk.status_code not in (200, 201, 422):       # 422 = it already exists (another call created it first)
        raise RuntimeError(f"cannot create branch: {_why(mk)} (the token needs Contents: Read and write)")
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
                STATUS.update(last_attempt_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), last_ok=True, last_error="", saved=STATUS["saved"] + 1)
                print(f"[archive] saved {path}", flush=True)
                return
            last = f"writing the file: {_why(r)}"
        except Exception as e:  # network trouble, bad token, ...
            last = f"{type(e).__name__}: {str(e)[:160]}"
        time.sleep(2 * (attempt + 1))
    STATUS.update(last_attempt_utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), last_ok=False, last_error=last, failed=STATUS["failed"] + 1)
    print(f"[archive] FAILED to save call {data['id']}: {last}", flush=True)


def archive_call(summary: dict) -> Optional[threading.Thread]:
    """Fire and forget: never raises, never blocks the call."""
    if not enabled():
        return None
    t = threading.Thread(target=_put, args=(summary,), daemon=True)
    t.start()
    return t


def status() -> dict:
    token, repo, branch = _cfg()
    return {"enabled": enabled(), "repo": repo or "(not set)", "branch": branch,
            "token_present": bool(token), "token_length": len(token), "token_kind": token.split("_")[0] + "_…" if "_" in token else "(unrecognised format)",
            **STATUS}


def check() -> dict:
    """Try the whole chain without saving a call: token valid? repo reachable? branch exists or can be created?"""
    token, repo, branch = _cfg()
    if not enabled():
        return {"ok": False, "problem": "ARCHIVE_GITHUB_TOKEN and/or ARCHIVE_REPO (owner/name) are not set on the server"}
    try:
        h = _headers(token)
        me = requests.get(f"{API}/repos/{repo}", headers=h, timeout=15)
        if me.status_code != 200:
            return {"ok": False, "problem": f"cannot read repo {repo}: {_why(me)}",
                    "hint": "check the spelling owner/name; the token must be a fine-grained token that includes THIS repository"}
        perms = me.json().get("permissions", {})
        _branch_ready["ok"] = False
        with _lock:
            _ensure_branch(token, repo, branch)
        return {"ok": True, "repo": repo, "branch": branch, "permissions": perms, "message": "branch exists or was just created"}
    except Exception as e:
        return {"ok": False, "problem": str(e)[:300]}


# ---- generic files on the same branch (used by the voice recorder) -------------------------------------------------
def put_file(path: str, content: bytes, message: str) -> None:
    """Create or overwrite one file on the data branch. Raises RuntimeError with GitHub's reason on failure."""
    token, repo, branch = _cfg()
    if not enabled():
        raise RuntimeError("GitHub saving is not configured on the server (ARCHIVE_REPO / ARCHIVE_GITHUB_TOKEN)")
    h = _headers(token)
    with _lock:
        _ensure_branch(token, repo, branch)
    sha = None
    cur = requests.get(f"{API}/repos/{repo}/contents/{path}", headers=h, params={"ref": branch}, timeout=20)
    if cur.status_code == 200:
        sha = cur.json().get("sha")
    elif cur.status_code != 404:
        raise RuntimeError(f"cannot check existing file: {_why(cur)}")
    body = {"message": message, "content": base64.b64encode(content).decode(), "branch": branch}
    if sha:
        body["sha"] = sha
    r = requests.put(f"{API}/repos/{repo}/contents/{path}", headers=h, json=body, timeout=30)
    if r.status_code not in (200, 201):
        raise RuntimeError(f"cannot write file: {_why(r)}")


def list_dir(path: str) -> list:
    """File names inside a folder on the data branch ([] if the folder or branch does not exist yet)."""
    token, repo, branch = _cfg()
    if not enabled():
        return []
    r = requests.get(f"{API}/repos/{repo}/contents/{path}", headers=_headers(token), params={"ref": branch}, timeout=20)
    if r.status_code == 404:
        return []
    if r.status_code != 200:
        raise RuntimeError(f"cannot list folder: {_why(r)}")
    return [x["name"] for x in r.json() if x.get("type") == "file"]


def get_file(path: str) -> Optional[bytes]:
    token, repo, branch = _cfg()
    if not enabled():
        return None
    h = {**_headers(token), "Accept": "application/vnd.github.raw+json"}
    r = requests.get(f"{API}/repos/{repo}/contents/{path}", headers=h, params={"ref": branch}, timeout=30)
    return r.content if r.status_code == 200 else None
