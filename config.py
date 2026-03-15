"""
Configuration and secrets loader.

Reads from .env (local dev) via python-dotenv.
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
