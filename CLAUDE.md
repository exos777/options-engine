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

## Daily Scanner — Same-Day CSP Mode

The Daily Scanner tab is a separate mode for selling a put and buying
it back THE SAME DAY. Assignment is only the worst case if the trade
can't be closed, so the watchlist holds only quality stocks the trader
would own. Its rules override the wheel filters above for this tab only:

- CSP only (no covered calls).
- Earliest expiration only — a daily if listed, else the nearest weekly;
  never more than 7 DTE. Today's expiration counts until the 4:00 pm ET
  close. (Overrides "DTE < 5 → NO TRADE".)
- Ranked by what a same-day round trip should earn
  (`scoring/csp_premium.py`):
  - decay today = |theta| (Tradier greeks, per day) × session hours
    left ÷ 6.5, capped at 70% of the credit
  - net edge = decay today − bid/ask spread (paid in and out)
  - puts whose decay can't beat the spread are dropped
- Score: net edge per $ of collateral 35%, liquidity 25% (spread 50%,
  today's volume 30%, open interest 20%), delta 20% (sweet spot
  0.10–0.20), distance vs expected move 10%, IV rank 5%, support 5%.
- Earnings are ALLOWED but flagged, and halve the decay estimate: event
  premium doesn't bleed intraday. (Overrides the earnings hard filter.)
- Downtrend regime is a −10 score penalty and a badge, not a block.
- Hard filters: OTM only, |delta| 0.05–0.35, bid ≥ $0.05, spread ≤ 10%
  of mid, OI or volume ≥ 100.
- Each pick shows a plan: target price if decay holds, stop at 2× credit,
  close by 3:45 pm ET, cost if assigned, cash required, contracts that
  fit the Ledger's limits; plus a "Compare strikes" table (top 3).
- The Ledger raises an action alert after 3:30 pm ET for any put opened
  today that is still open.

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
- `data/store.py` — persistence (SQLite locally, Postgres via `DATABASE_URL` on Railway): prefs, trades, IV history
- `portfolio/ledger.py` — replays trades into wheel cycles (adjusted basis, premium, realized P&L)
- `portfolio/alerts.py` — exit alerts (same-day close, 50% profit take, strike tested, ITM, earnings, CC below basis)
- `portfolio/exposure.py` — capital at risk vs account size / per-ticker limits
- `indicators/technical.py` — SMA, EMA, Bollinger Bands, MACD, RSI, ATR
- `indicators/support_resistance.py` — pivot-based S/R detection
- `scoring/regime.py` — 6-regime classifier
- `scoring/common.py` — shared scoring utilities
- `scoring/covered_call.py` — CC scoring engine
- `scoring/cash_secured_put.py` — CSP scoring engine
- `scoring/engine.py` — recommendation selection + explanations
- `ui/chart.py` — Plotly 3-panel chart
- `ui/recommendations.py` — Streamlit card rendering
- `scoring/csp_premium.py` — same-day CSP scorer (Daily Scanner)
- `ui/scanner.py` — Daily Scanner tab, same-day CSP mode (records ATM IV daily for true IV rank)
- `ui/ledger.py` — Ledger tab (journal, alerts, exposure, performance)
- `app/main.py` — Streamlit entry point
- `config.py` — secrets loader (dotenv)

## Python Environment

- Python 3.14.3 via uv at `C:\Users\leeha\AppData\Roaming\uv\python\cpython-3.14.3-windows-x86_64-none\python.exe`
- Always use `.venv\Scripts\python.exe` (never bare `python` or `streamlit`)
- Run app: `.venv\Scripts\python.exe -m streamlit run app/main.py`
- Run tests: `.venv\Scripts\pytest tests/ -q`
