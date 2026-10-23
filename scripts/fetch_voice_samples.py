"""Download the recordings made on the website (GitHub data branch) into data/voice_samples/ for the benchmark.
    GH=/path/to/gh .venv/bin/python scripts/fetch_voice_samples.py [owner/repo] [branch]
Uses your GitHub login (`gh auth token`) or the GITHUB_TOKEN environment variable."""
import base64, json, os, subprocess, sys
from pathlib import Path
import requests

repo = sys.argv[1] if len(sys.argv) > 1 else "TheIulius/barbershop-voice-assistant"
branch = sys.argv[2] if len(sys.argv) > 2 else "call-logs"
token = os.environ.get("GITHUB_TOKEN") or subprocess.run([os.environ.get("GH", "gh"), "auth", "token"], capture_output=True, text=True).stdout.strip()
if not token: sys.exit("no GitHub login found: set GITHUB_TOKEN or log in with gh")
h = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
out = Path(__file__).resolve().parent.parent / "data" / "voice_samples"; out.mkdir(parents=True, exist_ok=True)
r = requests.get(f"https://api.github.com/repos/{repo}/contents/voice_samples", headers=h, params={"ref": branch}, timeout=30)
if r.status_code == 404: sys.exit("no recordings on the data branch yet")
r.raise_for_status()
n = 0
for f in r.json():
    if f["name"].endswith(".wav"):
        blob = requests.get(f["download_url"], headers=h, timeout=60).content if f.get("download_url") else b""
        if not blob:
            blob = requests.get(f["url"], headers={**h, "Accept": "application/vnd.github.raw+json"}, timeout=60).content
        (out / f["name"]).write_bytes(blob); n += 1
print(f"downloaded {n} recordings to {out}")
