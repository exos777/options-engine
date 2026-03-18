import schwab
import os

APP_KEY = "your_app_key_here"
APP_SECRET = "your_app_secret_here"
CALLBACK_URL = "https://127.0.0.1"
TOKEN_PATH = "schwab_token.json"

c = schwab.auth.easy_client(
    api_key=APP_KEY,
    app_secret=APP_SECRET,
    callback_url=CALLBACK_URL,
    token_path=TOKEN_PATH,
)

print("Authentication successful!")
print(f"Token saved to {TOKEN_PATH}")
