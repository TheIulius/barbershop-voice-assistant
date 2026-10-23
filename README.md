# Barbershop voice assistant — free test prototype

Talk to an AI receptionist (Georgian / English / Russian) in your browser. It answers questions from
`data/shop.json` and books appointments into a calendar. Everything runs on your Mac; the only
outside services are the free language model and the free Microsoft voices.

> This tests the **brain** (conversation, safety, booking). A real phone number comes later
> (see STARTUP_ANALYSIS.md, telephony section). Use **fake names/numbers only** while testing on a free
> Gemini key: Google may use free-tier prompts for training.

## Deploy online so friends can test (free, password-protected)

Everything is behind a login (`app/auth.py`): pages, API and the live-call WebSocket. In `PUBLIC_MODE` the
developer pages (`/lab`, `/classic`, typed-text channel, voice saving) do not exist, bookings show masked phone
numbers, and a daily call-minute budget plus a concurrent-call cap protect your Google quota.

1. **New Gemini key.** The old one was pasted into a chat, so create a fresh key at aistudio.google.com/apikey and
   delete the old one. In Google AI Studio, set a spending limit / quota if the page offers one.
2. **GitHub** (private repo): create an empty repo on github.com, then
   `git remote add origin https://github.com/<you>/<repo>.git && git push -u origin main`
3. **Render** (render.com, free): New > Blueprint > pick the repo (it reads `render.yaml`), then in the service's
   Environment add the three secrets: `GEMINI_API_KEY` (new key), `APP_PASSWORD_HASH` and `SESSION_SECRET`
   (both are in the git-ignored file `DEPLOY_SECRETS.local.txt`).
4. Open the `https://<name>.onrender.com` address, sign in, press Call. Send friends the address + password.

Free-tier behaviour: the service sleeps after ~15 min without traffic and wakes in about a minute; local files
(test bookings) are lost on restart. Change the password any time with `scripts/make_password_hash.py` and
paste the new `APP_PASSWORD_HASH` into Render. Use a long passphrase: online guessing is limited to 5 tries per
15 minutes per connection, but nothing protects a short or guessable password if someone knows it.

## Keep call records after restarts (optional, recommended while testing)

Call records (what was said, tool results, why the call ended) live in server memory and vanish on every restart.
To keep them, the server can commit one small JSON file per call to a GitHub repo, on its own branch (`call-logs`):

1. Create a **private** repository that holds only logs (e.g. `barbershop-call-logs`). Do not reuse the code repo:
   a token that can write to the code repo could be abused to push code that Render then deploys.
2. GitHub → Settings → Developer settings → Fine-grained tokens → new token, **only that repo**,
   permission **Contents: Read and write**, nothing else.
3. In Render → Environment add `ARCHIVE_REPO` (`owner/name`) and `ARCHIVE_GITHUB_TOKEN` (the token). Optional:
   `ARCHIVE_BRANCH` (default `call-logs`), `ARCHIVE_MASK_NUMBERS=0` to keep phone numbers (default hides long digit runs).

Failures never affect a call (they are printed as `[archive] FAILED ...` in the Render logs). Records contain what
callers said: keep the repo private and use fake data while testing.

## 0. The live call (recommended) — natural talking, interruptions, Kore voice

`http://localhost:8000/` is a real-time voice call (Gemini Live, model `gemini-3.8-live`, voice **Kore**).
You can interrupt the assistant mid-sentence; it answers ~0.3–1 s after you stop speaking. Use headphones.

| Page | What |
|---|---|
| `/` | live call (this section) |
| `/classic` | older turn-by-turn version (record → think → speak) |
| `/lab` | compare voices side by side |

How it stays controlled: the browser talks to **our server**, which talks to Gemini. The AI can only act through
tools (`check_availability`, `book`, `confirm_booking`, `cancel_pending`, `request_callback`, `end_call`); every
tool call is validated by the same code as before (`app/actions.py`). `confirm_booking` is refused unless the
caller's own transcribed words after the read-back contain a "yes" (or the model quotes a clear yes and the
transcript has no "no"). The page's "Bookings made" list comes from the server, not from what the AI says.
Limits: 6-minute calls, 30 tool calls, 12 calls / 10 min per IP (set `LIVE_MAX_CALLS_10MIN`), 10 s silence nudge.

```bash
.venv/bin/python scripts/live_test.py booking_en   # synthetic caller books by voice (also booking_ka, booking_ru, interrupt)
.venv/bin/python scripts/live_redteam.py           # 15 injection attacks against the live assistant
```

Language policy: only Georgian, Russian and English are ever spoken. The call starts in the language chosen on the
page's buttons and stays there; it switches only when the caller says a full clear sentence in another of the three
(or asks). Unclear audio that "sounds like" Spanish/German/etc. gets a short "sorry, could you repeat?" in the call
language. A server-side guard cuts the audio and corrects the assistant if it ever drifts into another language.
Booking confirmations are written by the server in the language the assistant says it is speaking.

Known weak spots: Georgian phone digits are sometimes misheard (the assistant reads them back; always check),
the on-screen caller transcript is sometimes wrong in Georgian even when the AI understood, and the live voice
needs the Google key's quota (free tier can run out).

## 1. Run it (5 minutes)

```bash
cd barber-assistant
cp .env.example .env            # then open .env and paste your key after GEMINI_API_KEY=
.venv/bin/uvicorn app.main:app --port 8000
```

Free key: https://aistudio.google.com/apikey (sign in with a Google account → "Create API key").
Open http://localhost:8000 **in Chrome**, press **Call**, allow the microphone, and talk. You can also type.

- Speech-to-text is Chrome's built-in recognizer (free, sends audio to Google). Pick the language button
  (ქართ / EN / RU) matching what you will speak; the page then follows the language the assistant answers in.
- Voices: Microsoft `ka-GE-EkaNeural`, `ru-RU-SvetlanaNeural`, `en-US-JennyNeural` via `edge-tts`
  (unofficial, **test only**).
- Bookings go to `data/local_calendar.json` + `data/bookings.jsonl` and show in the page's bottom table.

## 2. Put bookings in Google Calendar

1. https://console.cloud.google.com → new project → enable **Google Calendar API**.
2. IAM & Admin → Service Accounts → create one → Keys → Add key → JSON. Save it somewhere safe
   (never commit it; `.gitignore` already blocks `service-account*.json`).
3. In Google Calendar create **one calendar per barber** (e.g. "Giorgi"). Settings → *Share with specific people* →
   add the service account's email → permission **Make changes to events**.
4. Copy each calendar's ID (Settings → *Integrate calendar*) into `data/shop.json` → `barbers.<id>.calendar_id`.
5. In `.env`: `CALENDAR_BACKEND=google` and `GOOGLE_SERVICE_ACCOUNT_FILE=/full/path/key.json`, restart.

The Google backend is written but **not yet tested against a real calendar** — do a test booking and check it appears.

## 3. Open-source / local model option

Set `LLM_PROVIDER=ollama`, install Ollama, `ollama pull <model>`, set `OLLAMA_MODEL`.
Caveats for this Mac (16 GB RAM, ~10 GB free disk): only small models (≈4 B parameters, ~3 GB) fit, and small
open models are usually weak at Georgian — test before trusting. Keep Gemini as the quality reference.

## 4. Safety design (what is and isn't guaranteed)

Nobody can make prompt injection *impossible* for a language model. This project instead makes it **harmless**:

| Layer | What it does |
|---|---|
| Action-only model | The model returns one JSON "request" from a fixed list; it can't run code or touch the calendar. |
| Server validation (`app/actions.py`) | Service/barber ids, dates (≤30 days), opening hours, 30-min grid, name (1–3 letters-only words), Georgian phone number are all checked by code. Invalid → refused. |
| Server-written confirmations | Read-back and "you are booked" come from templates after the calendar write succeeds; the model can't fake them. |
| Limits | 2 bookings per call, 3 future bookings per phone, 40 turns, 30 actions per call, rate limit per IP. |
| Double-booking | Slot re-checked under a lock at confirm time; Google event IDs are derived from barber+start so duplicates are rejected. |
| Input/output cleaning | Control/invisible characters stripped, fake tool-result markers removed, replies stripped of links/markdown, capped at 320 chars. |
| Canary | A secret token in the prompt; if the reply contains it (or prompt text) it is replaced by a refusal. |

Worst case if the model is fully jailbroken: it says something off-topic, or makes a *valid* booking the
caller could have made anyway. Remaining risks: spam bookings from many phone numbers (add caller-ID
checks on the real phone line), and wrong Georgian understanding (needs real testing).

## 5. Test it

```bash
.venv/bin/python -m pytest -q                # 41 offline tests, incl. simulated jailbroken models
.venv/bin/python scripts/redteam.py --ping   # does the real model answer?
.venv/bin/python scripts/redteam.py          # 23 attacks in ka/en/ru against the REAL model
.venv/bin/python scripts/demo_dialogue.py    # scripted booking calls in ka/en/ru + token usage
```

## Layout

`app/agent.py` conversation + prompt · `app/actions.py` the only code that changes state ·
`app/shop.py` business data + slots · `app/calendars.py` local/Google · `app/llm.py` Gemini/Ollama ·
`app/tts.py` voices · `app/main.py` web API · `static/index.html` the "phone" page · `data/shop.json` sample shop.
