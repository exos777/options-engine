"""
Configuration and secrets loader.

Reads from environment variables (Railway) or .env (local dev).
Keys are exposed as module-level constants — import what you need:

    from config import SCHWAB_APP_KEY, SCHWAB_APP_SECRET
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to real env vars

SCHWAB_APP_KEY: str    = os.getenv("SCHWAB_APP_KEY", "")
SCHWAB_APP_SECRET: str = os.getenv("SCHWAB_APP_SECRET", "")


def schwab_available() -> bool:
    """Return True if Schwab credentials are configured and a token exists."""
    from data.schwab_provider import _get_credentials
    app_key, app_secret, token_json = _get_credentials()
    if not app_key or app_key in ("your_key_here", "your_app_key_here"):
        return False
    token_path = Path(__file__).parent / "schwab_token.json"
    return bool(token_json or token_path.exists())
