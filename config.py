"""
Configuration and secrets loader.

Reads from .env (local dev) via python-dotenv, then Streamlit secrets.
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

# Try Streamlit secrets as fallback
try:
    import streamlit as st
    os.environ.setdefault("SCHWAB_APP_KEY", st.secrets.get("SCHWAB_APP_KEY", ""))
    os.environ.setdefault("SCHWAB_APP_SECRET", st.secrets.get("SCHWAB_APP_SECRET", ""))
except Exception:
    pass

SCHWAB_APP_KEY: str    = os.getenv("SCHWAB_APP_KEY", "")
SCHWAB_APP_SECRET: str = os.getenv("SCHWAB_APP_SECRET", "")


def schwab_available() -> bool:
    """Return True if Schwab credentials are configured and a token exists."""
    # Resolve app key: module-level var, env, or st.secrets
    app_key = SCHWAB_APP_KEY
    if not app_key or app_key in ("your_key_here", "your_app_key_here"):
        app_key = os.getenv("SCHWAB_APP_KEY", "")
    if not app_key or app_key in ("your_key_here", "your_app_key_here"):
        try:
            import streamlit as st
            app_key = st.secrets.get("SCHWAB_APP_KEY", "")
        except Exception:
            pass
    if not app_key or app_key in ("your_key_here", "your_app_key_here"):
        return False

    # Local: check for token file on disk
    token_path = Path(__file__).parent / "schwab_token.json"
    if token_path.exists():
        return True
    # Streamlit Cloud: check for token in secrets
    try:
        import streamlit as st
        if st.secrets.get("SCHWAB_TOKEN_JSON", ""):
            return True
    except Exception:
        pass
    return False
