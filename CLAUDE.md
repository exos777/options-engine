# Options Engine — Claude Guidelines

This app is designed for a disciplined wheel strategy trader who
systematically sells cash-secured puts (CSPs) and covered calls (CCs)
to generate consistent income over short timeframes (7–14 DTE).

## Core Product Rule

"Recommend the best strike to sell premium this week that either
expires worthless for income, or results in an assignment or exit
price aligned with my long-term willingness to own or sell the stock."

## Primary Objective

Maximize risk-adjusted income while ensuring all assignments occur
at prices the user is comfortable holding or exiting.

## Wheel Strategy Phases

### Phase 1 — Cash-Secured Put (Entry)

Goal: Collect premium OR acquire shares at a desirable price.
Core Question: "Would I confidently buy this stock at this strike?"
Hard Requirement: Strike must be at or below user-defined desired
ownership price.

Optimize for:
- Strike aligned with desired buy price
- Strong support confluence
- Outside expected move preferred
- 7-14 DTE default window
- High theta decay relative to time
- Attractive premium relative to capital at risk
- Favorable break-even vs support
- High liquidity (tight spreads, strong OI)

Penalize:
- Strike above desired buy price
- Weak or no support structure
- Inside expected move
- Earnings within expiration window
- IV percentile < 25 (premium too thin)
- Excessive vega risk before events

### Phase 2 — Covered Call (Exit)

Goal: Generate income while reducing cost basis OR exit at a profit.
Core Question: "Does this strike provide income AND allow profitable exit?"
Hard Requirement: Strike MUST be above cost basis.

Optimize for:
- Strike above cost basis with profit margin
- Resistance alignment
- Premium meaningfully reduces cost basis
- 7-14 DTE preferred
- Consistent income generation over cycles
- Balanced assignment probability vs upside capture

Penalize:
- Any strike below cost basis (hard filter)
- Selling calls too close in strong uptrends
- Poor liquidity or wide spreads
- Low premium environments

## Multi-Timeframe Regime

Weekly = Ownership decision ("Should I own this?")
Daily  = Entry timing ("Is now the right time?")

### Regime-Based Actions

CSP:
- Weekly Bullish + Daily Pullback → Best setup.
  Allow delta 0.25-0.30, full size.
- Weekly Bullish + Daily Overextended →
  Sell farther OTM, reduce size.
- Weekly Bearish → Avoid CSP unless:
  far OTM + strong support + exceptional premium.

CC:
- Weekly Bullish → Sell farther OTM (preserve upside).
- Weekly Bearish → Sell closer OTM (maximize income).
- Weekly + Daily Bearish → High probability CC cycles.

## Scoring Weights

### CSP Phase (Balanced)

| Factor              | Weight |
|---------------------|--------|
| Premium/Theta       | 25%    |
| Assignment quality  | 20%    |
| Delta/assignment    | 15%    |
| Expected move       | 15%    |
| Liquidity           | 10%    |
| Support alignment   | 10%    |
| Regime alignment    |  5%    |

### CC Phase (Balanced)

| Factor              | Weight |
|---------------------|--------|
| Premium/Theta       | 25%    |
| Cost basis          | 25%    |
| Delta/assignment    | 15%    |
| Expected move       | 10%    |
| Liquidity           | 10%    |
| Resistance          | 10%    |
| Regime alignment    |  5%    |

## Hard Trade Filters (No Exceptions)

- CSP strike above desired buy price → NO TRADE
- CC strike below cost basis → NO TRADE
- Earnings inside DTE window → NO TRADE
- Spread > 15% of premium → NO TRADE
- DTE < 5 → NO TRADE
- DTE > 21 → NO TRADE
- Weekly regime strongly bearish (CSP) → NO TRADE

## Position Sizing

| Score   | Size          |
|---------|---------------|
| >= 75   | Full Size ✅  |
| >= 60   | Half Size ⚡  |
| >= 45   | Quarter ⚠️    |
| < 45    | No Trade ⛔   |

## Final Decision Engine

App must always answer:
1. Should I sell this option?
2. At what strike?
3. At what size?
4. What happens next in the wheel?

## Architecture

- `data/provider.py` — yfinance wrapper, SSL bypass for corporate proxy
- `data/common.py` — shared data utilities (expiration, earnings, helpers)
- `indicators/technical.py` — SMA, EMA, Bollinger Bands, MACD, RSI, ATR
- `indicators/support_resistance.py` — pivot-based S/R detection
- `scoring/regime.py` — 6-regime classifier
- `scoring/common.py` — shared scoring utilities
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
