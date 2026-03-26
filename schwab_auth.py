"""
One-time Schwab OAuth setup — run this locally to create schwab_token.json.

    .venv\Scripts\python.exe schwab_auth.py

A browser window will open for you to log in to your Schwab account.
After authorization, the token is saved and the app can use the Schwab API.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import schwab

APP_KEY = os.getenv("SCHWAB_APP_KEY", "")
APP_SECRET = os.getenv("SCHWAB_APP_SECRET", "")
CALLBACK_URL = "https://127.0.0.1"
TOKEN_PATH = str(Path(__file__).parent / "schwab_token.json")

if not APP_KEY or APP_KEY in ("your_key_here", "your_app_key_here"):
    print("ERROR: SCHWAB_APP_KEY not set. Add it to .env first.")
    raise SystemExit(1)

c = schwab.auth.easy_client(
    api_key=APP_KEY,
    app_secret=APP_SECRET,
    callback_url=CALLBACK_URL,
    token_path=TOKEN_PATH,
)

print("Authentication successful!")
print(f"Token saved to {TOKEN_PATH}")
