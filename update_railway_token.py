"""
Push schwab_token.json to Railway as SCHWAB_TOKEN_JSON env var.

Run after schwab_auth.py to update Railway without manual copy/paste:

    .venv\\Scripts\\python.exe update_railway_token.py

Requires:
  - RAILWAY_TOKEN env var (railway.app → Account Settings → API Tokens)
  - RAILWAY_PROJECT_ID, RAILWAY_SERVICE_ID, RAILWAY_ENVIRONMENT_ID in .env
    or as env vars.  Find them in the Railway dashboard URL:
      railway.app/project/<PROJECT_ID>/service/<SERVICE_ID>
    Environment ID: Settings → Environments → click → ID in URL.
"""

import json
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import requests

RAILWAY_TOKEN = os.environ.get("RAILWAY_TOKEN", "")
PROJECT_ID = os.environ.get("RAILWAY_PROJECT_ID", "")
SERVICE_ID = os.environ.get("RAILWAY_SERVICE_ID", "")
ENVIRONMENT_ID = os.environ.get("RAILWAY_ENVIRONMENT_ID", "")

TOKEN_PATH = Path(__file__).parent / "schwab_token.json"

RAILWAY_API = "https://backboard.railway.app/graphql/v2"

UPSERT_MUTATION = """
mutation variableUpsert($input: VariableUpsertInput!) {
    variableUpsert(input: $input)
}
"""


def update_railway_variable(name: str, value: str) -> dict:
    headers = {
        "Authorization": f"Bearer {RAILWAY_TOKEN}",
        "Content-Type": "application/json",
    }
    variables = {
        "input": {
            "projectId": PROJECT_ID,
            "serviceId": SERVICE_ID,
            "environmentId": ENVIRONMENT_ID,
            "name": name,
            "value": value,
        }
    }
    resp = requests.post(
        RAILWAY_API,
        headers=headers,
        json={"query": UPSERT_MUTATION, "variables": variables},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        print(f"ERROR: Railway API returned errors: {data['errors']}")
        sys.exit(1)
    return data


def main():
    missing = []
    if not RAILWAY_TOKEN:
        missing.append("RAILWAY_TOKEN")
    if not PROJECT_ID:
        missing.append("RAILWAY_PROJECT_ID")
    if not SERVICE_ID:
        missing.append("RAILWAY_SERVICE_ID")
    if not ENVIRONMENT_ID:
        missing.append("RAILWAY_ENVIRONMENT_ID")
    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}")
        print("Add them to .env or set as environment variables.")
        sys.exit(1)

    if not TOKEN_PATH.exists():
        print(f"ERROR: {TOKEN_PATH} not found. Run schwab_auth.py first.")
        sys.exit(1)

    token_json = TOKEN_PATH.read_text().strip()
    json.loads(token_json)  # validate it's valid JSON

    print(f"Uploading {len(token_json)} bytes to Railway SCHWAB_TOKEN_JSON...")
    result = update_railway_variable("SCHWAB_TOKEN_JSON", token_json)
    print(f"Railway updated successfully: {result.get('data', {})}")


if __name__ == "__main__":
    main()
