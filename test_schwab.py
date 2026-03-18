"""
Quick test: authenticate with Schwab and fetch a live options chain.

Run locally first to generate schwab_token.json via browser OAuth:
    .venv\Scripts\python.exe test_schwab.py

After token is generated, subsequent runs use the saved token automatically.
"""
import os
import sys
from pathlib import Path

# Load .env for local dev
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

# Also try st.secrets if running inside Streamlit
try:
    import streamlit as st
    os.environ.setdefault("SCHWAB_APP_KEY", st.secrets.get("SCHWAB_APP_KEY", ""))
    os.environ.setdefault("SCHWAB_APP_SECRET", st.secrets.get("SCHWAB_APP_SECRET", ""))
except Exception:
    pass

APP_KEY    = os.getenv("SCHWAB_APP_KEY", "")
APP_SECRET = os.getenv("SCHWAB_APP_SECRET", "")
TOKEN_PATH = Path(__file__).parent / "schwab_token.json"
CALLBACK   = "https://127.0.0.1"
TEST_SYMBOL = "SPY"

def main():
    if not APP_KEY or APP_KEY in ("your_key_here", "your_app_key_here"):
        print("ERROR: SCHWAB_APP_KEY not set. Add it to .env or Streamlit secrets.")
        sys.exit(1)
    if not APP_SECRET or APP_SECRET in ("your_key_here", "your_app_secret_here"):
        print("ERROR: SCHWAB_APP_SECRET not set. Add it to .env or Streamlit secrets.")
        sys.exit(1)

    print(f"APP_KEY loaded:    {APP_KEY[:6]}...{APP_KEY[-4:]}")
    print(f"TOKEN_PATH:        {TOKEN_PATH}")
    print(f"Token file exists: {TOKEN_PATH.exists()}")
    print()

    import schwab

    print("Authenticating with Schwab (browser will open if no token exists)...")
    try:
        c = schwab.auth.easy_client(
            api_key=APP_KEY,
            app_secret=APP_SECRET,
            callback_url=CALLBACK,
            token_path=str(TOKEN_PATH),
        )
        print("Authentication OK")
    except Exception as e:
        print(f"Auth FAILED: {e}")
        sys.exit(1)

    print(f"\nFetching quote for {TEST_SYMBOL}...")
    try:
        resp = c.get_quote(TEST_SYMBOL)
        resp.raise_for_status()
        data = resp.json()
        price = data[TEST_SYMBOL]["quote"]["lastPrice"]
        print(f"  {TEST_SYMBOL} last price: ${price:.2f}  -- Schwab quote OK")
    except Exception as e:
        print(f"Quote fetch FAILED: {e}")
        sys.exit(1)

    print(f"\nFetching option chain for {TEST_SYMBOL}...")
    try:
        resp = c.get_option_chain(
            TEST_SYMBOL,
            contract_type=schwab.client.Client.Options.ContractType.ALL,
            strike_count=5,
        )
        resp.raise_for_status()
        chain = resp.json()
        exp_dates = list(chain.get("callExpDateMap", {}).keys())
        print(f"  Expirations available: {exp_dates[:3]} ...")
        print("  Option chain OK")
    except Exception as e:
        print(f"Option chain FAILED: {e}")
        sys.exit(1)

    print("\nAll Schwab connectivity tests PASSED.")

if __name__ == "__main__":
    main()
