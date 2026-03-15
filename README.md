# Options Screener

A local Streamlit app for screening weekly covered calls and cash-secured puts.
Enter a ticker, select a strategy, and get ranked strike recommendations backed by
real-time option chain data, technical chart analysis, and a weighted scoring engine.

---

## Features

| Section | Details |
|---|---|
| **Market Overview** | Live price, % move, RSI, ATR, expected move, earnings warning |
| **Technical Chart** | Candlestick + SMA 20/50 + support/resistance + strike overlays + volume |
| **Recommendations** | Top 3 strikes labelled Safest Income / Best Balance / Max Premium with plain-English explanations |
| **Ranked Table** | All qualifying strikes scored and sortable |

**Scoring weights:** Premium · Delta · Liquidity · Chart Alignment · Cost Basis / Buy Price.
All weights shift by risk profile (Conservative / Balanced / Aggressive).

---

## Requirements

- Python 3.11+
- Internet connection (Yahoo Finance data, 15-min delayed)

---

## Setup

```bash
# 1. Clone / open the project folder
cd options-engine

# 2. Create a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Run the App

```bash
streamlit run app/main.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Run Tests

```bash
# From the project root
pytest tests/ -v
```

All tests are pure unit tests — no network calls required.

---

## Project Structure

```
options-engine/
│
├── app/
│   └── main.py                  # Streamlit UI entry point
│
├── data/
│   └── provider.py              # yfinance wrapper (quotes, chains, history)
│
├── indicators/
│   ├── technical.py             # SMA, RSI, ATR, composite builder
│   └── support_resistance.py    # Pivot-based S/R detection
│
├── scoring/
│   ├── regime.py                # Chart regime classifier
│   ├── covered_call.py          # Covered call scoring engine
│   ├── cash_secured_put.py      # CSP scoring engine
│   └── engine.py                # Recommendation engine + ScreenerResult builder
│
├── strategies/
│   └── models.py                # Shared dataclasses and enumerations
│
├── ui/
│   ├── chart.py                 # Plotly figure builder
│   └── recommendations.py       # Streamlit rendering helpers
│
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_indicators.py
│   ├── test_support_resistance.py
│   ├── test_regime.py
│   ├── test_covered_call.py
│   ├── test_cash_secured_put.py
│   ├── test_engine.py
│   └── test_edge_cases.py
│
├── requirements.txt
└── README.md
```

---

## How the Scoring Works

### Covered Calls

| Component | Weight (Balanced) | Notes |
|---|---|---|
| Premium / Ann. Return | 20% | Logistic curve; target ~25% APY for Balanced |
| Delta | 25% | Lower delta → higher score; hard filter above max |
| Liquidity (OI + spread) | 20% | Log-scaled OI + linear spread penalty |
| Chart Alignment | 20% | Regime multiplier × sweet-spot OTM distance |
| Cost Basis | 15% | Full score above basis; zero if below break-even (unless allowed) |

### Cash-Secured Puts

| Component | Weight (Balanced) | Notes |
|---|---|---|
| Premium / Ann. Return | 20% | Same logistic curve |
| Delta | 25% | Low abs(delta) preferred |
| Liquidity | 20% | Same as above |
| Chart Alignment | 20% | Bonus for strikes at/below support |
| Buy Price Proximity | 15% | Net cost vs. desired acquisition price |

### Chart Regime Rules

| Regime | CC Trade Bias | CSP Trade Bias |
|---|---|---|
| Bullish | Go — favor farther OTM | Go |
| Neutral | Caution — closer OTM OK | Caution |
| Near Resistance | Caution — strikes just above R score well | Neutral |
| Near Support | Go | Go — strikes near support score highest |
| Overextended | Caution | Caution |
| Bearish | **Skip** — scores reduced 65% | **Skip** |

### Risk Profiles

| Profile | Max Delta | Weight Shift |
|---|---|---|
| Conservative | 0.30 | More weight on delta + chart |
| Balanced | 0.40 | Even distribution |
| Aggressive | 0.50 | More weight on premium |

---

## Data Source

All market data comes from **Yahoo Finance** via the `yfinance` library.
Quotes are approximately 15 minutes delayed. Option Greeks (delta, IV) are
included when Yahoo provides them; if missing, neutral fallback values are used.

---

## Limitations & Notes

- **Not financial advice.** This tool is for educational and analytical purposes only.
- Greeks are not always available from Yahoo Finance, especially for less liquid names.
- Expected move is estimated from ATM implied volatility — it is approximate.
- Earnings dates are sourced from Yahoo Finance and may occasionally be inaccurate.
- Weekly options may not exist for all tickers; the app falls back to the nearest available expiration.
