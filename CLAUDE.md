# Options Engine — Claude Guidelines

## Core Product Philosophy

This app is designed for option premium selling, not option buying.
Optimize all recommendations for income generation, risk-adjusted premium
capture, theta decay, acceptable assignment outcomes, and liquidity.

Penalize event risk, excessive vega exposure, poor break-even levels, and
strikes inside the expected move. For covered calls, prefer strikes above
cost basis unless explicitly allowed otherwise. For cash-secured puts,
prefer strikes at prices the user would be willing to own.

**Core product rule:** "What strike gives me the best premium income for
this week, with acceptable risk if I get assigned?"

## Scoring Philosophy

- **Premium income** and **theta decay rate** are the primary value drivers.
- **Delta** bounds assignment risk.
- **Liquidity** (OI + tight spread) ensures fills are achievable.
- **Chart alignment** provides regime context.
- **Expected move** acts as a risk modifier: outside EM = safer.
- **Vega** acts as an IV-expansion risk modifier: high vega = riskier.
- **Earnings in window** is an event-risk penalty — always applied.

## Architecture

- `data/provider.py` — yfinance wrapper, SSL bypass for corporate proxy
- `indicators/technical.py` — SMA, EMA, Bollinger Bands, MACD, RSI, ATR
- `indicators/support_resistance.py` — pivot-based S/R detection
- `scoring/regime.py` — 6-regime classifier
- `scoring/covered_call.py` — CC scoring engine
- `scoring/cash_secured_put.py` — CSP scoring engine
- `scoring/engine.py` — recommendation selection + explanations
- `ui/chart.py` — Plotly 3-panel chart
- `ui/recommendations.py` — Streamlit card rendering
- `app/main.py` — Streamlit entry point
- `config.py` — secrets loader (dotenv)

## Python Environment

- Python 3.14.3 via uv at `C:\Users\leeha\AppData\Roaming\uv\python\cpython-3.14.3-windows-x86_64-none\python.exe`
- Always use `.venv\Scripts\python.exe` (never bare `python` or `streamlit`)
- Run app: `.venv\Scripts\python.exe -m streamlit run app/main.py`
- Run tests: `.venv\Scripts\pytest tests/ -q`
