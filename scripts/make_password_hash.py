"""Create the values for the password gate.
    .venv/bin/python scripts/make_password_hash.py            (asks for the password, nothing is echoed)
Prints APP_PASSWORD_HASH and a fresh SESSION_SECRET to paste into your hosting provider's secret settings.
Never commit these, and never put the plain password in a file."""
import getpass, os, secrets, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.auth import hash_password

pw = os.environ.get("NEW_PASSWORD") or getpass.getpass("New password: ")
if len(pw) < 12:
    print("Warning: use at least 12 characters, ideally a long passphrase of several random words.", file=sys.stderr)
print("APP_PASSWORD_HASH=" + hash_password(pw))
print("SESSION_SECRET=" + secrets.token_urlsafe(48))
