"""
Options Screener — Streamlit entry point.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import importlib
import sys
import os

# Ensure project root is on the path when running from the app/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Force-reload all project modules to bust Streamlit Cloud's stale module cache.
# Without this, cached .pyc from a prior deploy can shadow the current source.
#
# IMPORTANT: modules that define dataclasses crossing the @st.cache_data
# boundary (strategies.models → Quote, OptionContract, …) MUST be reloaded
# FIRST. st.cache_data pickles return values, and pickle verifies that an
# instance's class *is* the class currently bound in its module. If a
# provider module is reloaded before strategies.models, it binds the old
# Quote class while strategies.models holds the new one — pickle then fails
# with "Cannot serialize the return value (of type strategies.models.Quote)".
# Reloading models first means every dependent re-binds to the fresh class.
_MODEL_PREFIXES = ("strategies.", "portfolio.models")
_PROJECT_PREFIXES = (
    "scoring.", "indicators.", "data.", "ui.", "greeks.", "portfolio.", "config",
)


def _matches(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == p or name.startswith(p) for p in prefixes)


for _mod_name in list(sys.modules):
    if _matches(_mod_name, _MODEL_PREFIXES):
        importlib.reload(sys.modules[_mod_name])
for _mod_name in list(sys.modules):
    if _matches(_mod_name, _PROJECT_PREFIXES) and not _matches(_mod_name, _MODEL_PREFIXES):
        importlib.reload(sys.modules[_mod_name])

import streamlit as st

from data import provider as _yf_provider
from data.common import pick_wheel_expiration
from data.tradier_provider import (
    tradier_available,
    tradier_status,
    tradier_token_fingerprint as _tradier_token_fingerprint,
)
from indicators.technical import calculate_full_indicators
from indicators.support_resistance import find_support_resistance
from scoring.regime import classify_regime
from scoring.covered_call import score_covered_calls
from scoring.cash_secured_put import score_cash_secured_puts
from scoring.engine import run_screener
from strategies.models import FilterParams, RiskProfile, Strategy
from ui.chart import build_price_chart
from portfolio.exposure import compute_exposure, max_new_csp_contracts
from ui import ledger as ledger_ui
from ui.assignment_view import render_assignment_panel
from ui import recommendations as rec_ui
from ui.position_management import render_position_manager
from ui.scanner import render_scanner


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Options Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults (prevents KeyError crashes)
# ---------------------------------------------------------------------------
_SESSION_DEFAULTS = {
    "result": None,
    "hist_df": None,
    "full_ind": None,
    "expected_move": 0.0,
    "expirations": [],
    "default_exp": "",
    "last_ticker": "",
    "ticker_history": ["TSLA", "MSFT", "NVDA"],
    "auto_run": False,
}
for _k, _v in _SESSION_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.title("📈 Weekly Options Screener")
st.caption("Covered Calls & Cash-Secured Puts")

st.markdown("""
<style>
/* Quick ticker buttons */
div[data-testid="column"] > div > div > div > button {
    background-color: #1a4a2e;
    color: #3fb950;
    border: 1px solid #3fb950;
    border-radius: 6px;
    font-weight: 700;
    font-size: 13px;
}
div[data-testid="column"] > div > div > div > button:hover {
    background-color: #3fb950;
    color: #0d1117;
    border-color: #3fb950;
}
div[data-testid="column"] > div > div > div > button:focus {
    background-color: #3fb950;
    color: #0d1117;
    border-color: #3fb950;
    box-shadow: 0 0 0 2px #1a4a2e;
}

/* Main tabs: larger and spread evenly across the full width. The :not()
   skips any tab bar sitting inside a nested stTabs, so in-tab sub-tabs
   (e.g. the Ledger's "Single fill / Roll") keep Streamlit's default. */
[data-testid="stTabs"] [role="tablist"]:not([data-testid="stTabs"] [data-testid="stTabs"] *) {
    gap: 0.5rem;
}
[data-testid="stTabs"] [role="tablist"]:not([data-testid="stTabs"] [data-testid="stTabs"] *) > [role="tab"] {
    flex: 1 1 0;
    justify-content: center;
    padding: 0.85rem 0.5rem;
    border-radius: 8px 8px 0 0;
}
[data-testid="stTabs"] [role="tablist"]:not([data-testid="stTabs"] [data-testid="stTabs"] *) > [role="tab"]:hover {
    background-color: rgba(255, 255, 255, 0.05);
}
[data-testid="stTabs"] [role="tablist"]:not([data-testid="stTabs"] [data-testid="stTabs"] *) > [role="tab"] p {
    font-size: 1.2rem;
    font-weight: 600;
}
@media (max-width: 640px) {
    /* Phones: let labels wrap so all four tabs fit without scrolling. */
    [data-testid="stTabs"] [role="tablist"]:not([data-testid="stTabs"] [data-testid="stTabs"] *) {
        gap: 0.25rem;
    }
    [data-testid="stTabs"] [role="tablist"]:not([data-testid="stTabs"] [data-testid="stTabs"] *) > [role="tab"] {
        min-width: 0;
        padding: 0.6rem 0.25rem;
    }
    [data-testid="stTabs"] [role="tablist"]:not([data-testid="stTabs"] [data-testid="stTabs"] *) > [role="tab"] p {
        font-size: 0.9rem;
        line-height: 1.25;
        white-space: normal;
        text-align: center;
    }
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Input Panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    # Data comes from Tradier, falling back to Yahoo Finance automatically.
    # Validate the token against the live API rather than merely checking
    # that one is set — a stale key otherwise shows a false "connected"
    # badge and then 401s on the first real request. Cached per-token so
    # this costs one call per key, not one per rerun.
    os.environ.pop("OPTIONS_FORCE_YFINANCE", None)
    _tok_fingerprint = _tradier_token_fingerprint()
    if st.session_state.get("_tradier_probe_key") != _tok_fingerprint:
        st.session_state["_tradier_probe_key"] = _tok_fingerprint
        st.session_state["_tradier_probe"] = tradier_status()
    _tradier_live, _tradier_msg = st.session_state.get(
        "_tradier_probe", (False, "Not checked.")
    )

    if _tradier_live:
        st.success("📡 Tradier connected", icon="✅")
    else:
        st.warning(
            f"Tradier not connected — using Yahoo Finance (15-min delayed). "
            f"{_tradier_msg}",
            icon="⚠️",
        )
        if st.button("↻ Re-check Tradier", use_container_width=True):
            st.session_state.pop("_tradier_probe_key", None)
            st.rerun()

    st.divider()

    # --- Ticker history ---
    if "ticker_history" not in st.session_state:
        st.session_state["ticker_history"] = ["TSLA", "MSFT", "NVDA"]

    history = st.session_state.get("ticker_history", ["TSLA", "MSFT", "NVDA"])
    active_ticker = st.session_state.get("last_ticker", "TSLA")

    label1 = f"✓ {history[0]}" if len(history) > 0 and history[0] == active_ticker else (history[0] if len(history) > 0 else "TSLA")
    label2 = f"✓ {history[1]}" if len(history) > 1 and history[1] == active_ticker else (history[1] if len(history) > 1 else "MSFT")
    label3 = f"✓ {history[2]}" if len(history) > 2 and history[2] == active_ticker else (history[2] if len(history) > 2 else "NVDA")

    st.caption("Recent tickers:")
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        if st.button(label1, use_container_width=True, key="quick_ticker_1"):
            st.session_state["quick_select"] = history[0] if len(history) > 0 else "TSLA"
            st.session_state["auto_run"] = True
    with btn_col2:
        if st.button(label2, use_container_width=True, key="quick_ticker_2"):
            st.session_state["quick_select"] = history[1] if len(history) > 1 else "MSFT"
            st.session_state["auto_run"] = True
    with btn_col3:
        if st.button(label3, use_container_width=True, key="quick_ticker_3"):
            st.session_state["quick_select"] = history[2] if len(history) > 2 else "NVDA"
            st.session_state["auto_run"] = True

    default_ticker = st.session_state.pop(
        "quick_select",
        st.session_state.get("last_ticker", "TSLA"),
    )
    ticker_input = st.text_input(
        "Ticker Symbol",
        value=default_ticker,
        max_chars=10,
        help="Enter a US stock ticker (e.g. TSLA, MSFT, NVDA)",
    ).upper().strip()

    run_button = st.button("🔍 Run Screener", type="primary", use_container_width=True)

    # The Max Premium panel screens covered calls AND puts. The single-side
    # sections below it follow the wheel phase: covered calls when the
    # Ledger holds 100+ shares of this ticker, cash-secured puts otherwise.
    # Covered calls also use the Ledger's premium-adjusted basis as the
    # cost-basis hard filter.
    _ledger_pos = ledger_ui.position_for(ticker_input)
    _ledger_has_shares = _ledger_pos is not None and _ledger_pos.shares > 0
    strategy = (
        Strategy.COVERED_CALL
        if _ledger_has_shares and _ledger_pos.shares >= 100
        else Strategy.CASH_SECURED_PUT
    )
    if _ledger_has_shares:
        st.caption(
            f"📒 Ledger: {_ledger_pos.shares:,} {ticker_input} shares · "
            f"adjusted basis ${_ledger_pos.adjusted_basis:,.2f}"
        )


# ---------------------------------------------------------------------------
# Data provider: Tradier-first with automatic yfinance fallback
# ---------------------------------------------------------------------------
# data.provider tries Tradier and falls back to yfinance per call; gate the
# "live" label on the *validated* token status, not merely "a token is set".
_use_tradier = _tradier_live
dp = _yf_provider
_source_label = "📡 Tradier (Live)" if _use_tradier else "📊 yfinance (Delayed)"

# ---------------------------------------------------------------------------
# Build FilterParams — standard wheel defaults (the sidebar no longer has
# risk / filter controls). Cost basis comes from the ledger when held.
# ---------------------------------------------------------------------------

_ledger_basis = (
    round(max(_ledger_pos.adjusted_basis, 0.0), 2) if _ledger_has_shares else None
)
filter_params = FilterParams(
    strategy=strategy,
    risk_profile=RiskProfile.BALANCED,
    max_delta=0.40,
    min_open_interest=50,
    max_spread_pct=0.30,
    min_premium=0.10,
    shares_owned=_ledger_pos.shares if _ledger_has_shares else 0,
    cost_basis=_ledger_basis if strategy == Strategy.COVERED_CALL else None,
    allow_below_basis=False,
    cash_available=None,
    desired_buy_price=None,
)



# ---------------------------------------------------------------------------
# Cached data-fetching helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300, show_spinner=False)
def cached_quote(ticker: str):
    return dp.get_quote(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_expirations(ticker: str):
    return dp.get_expirations(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_chain(ticker: str, expiration: str):
    return dp.get_option_chain(ticker, expiration)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_earnings(ticker: str):
    if not hasattr(dp, "get_earnings_date"):
        return None
    try:
        return dp.get_earnings_date(ticker)
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_history(ticker: str):
    return dp.get_historical(ticker, months=6)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(ticker: str, expiration: str, params: FilterParams):
    """
    Execute the full screening pipeline for one ticker / expiration and
    return a ScreenerResult.  All exceptions are caught and displayed.
    """
    progress = st.progress(0, text="Fetching quote…")

    quote = cached_quote(ticker)
    progress.progress(15, text="Fetching expirations…")

    progress.progress(25, text=f"Fetching option chain for {expiration}…")
    calls, puts = cached_chain(ticker, expiration)

    progress.progress(40, text="Fetching historical prices…")
    hist_df = cached_history(ticker)

    progress.progress(55, text="Calculating technical indicators…")
    full_ind = calculate_full_indicators(hist_df)
    indicators = full_ind.snapshot

    progress.progress(65, text="Detecting support / resistance…")
    support_lvls, resistance_lvls = find_support_resistance(
        hist_df, quote.price
    )

    progress.progress(75, text="Classifying chart regime…")
    regime = classify_regime(indicators, support_lvls, resistance_lvls)

    progress.progress(82, text="Scoring options…")
    dte = dp.days_to_expiration(expiration)

    # Compute single expected move from avg IV of the relevant chain
    import math as _math
    _chain = calls if params.strategy == Strategy.COVERED_CALL else puts
    _ivs = [c.implied_volatility for c in _chain if c.implied_volatility]
    _avg_iv = sum(_ivs) / len(_ivs) if _ivs else None
    expected_move = (
        quote.price * _avg_iv * _math.sqrt(max(dte, 1) / 365)
        if _avg_iv
        else indicators.atr_14 * _math.sqrt(max(dte, 1))
    )

    # If user explicitly selected a short-dated expiration, relax the min_dte
    # guard so strikes are still scored (user chose the expiration intentionally).
    effective_min_dte = 0 if dte < 5 else 5

    if params.strategy == Strategy.COVERED_CALL:
        scored = score_covered_calls(
            calls, quote.price, dte,
            indicators, regime,
            support_lvls, resistance_lvls,
            params,
            earnings_date=quote.earnings_date,
            expected_move=expected_move,
            min_dte=effective_min_dte,
        )
    else:
        scored = score_cash_secured_puts(
            puts, quote.price, dte,
            indicators, regime,
            support_lvls, resistance_lvls,
            params,
            earnings_date=quote.earnings_date,
            expected_move=expected_move,
            min_dte=effective_min_dte,
        )

    progress.progress(92, text="Building recommendations…")
    warnings: list[str] = []
    ew = dp.earnings_warning(quote, expiration)
    if ew:
        warnings.append(ew)

    result = run_screener(
        quote=quote,
        expiration=expiration,
        dte=dte,
        scored_options=scored,
        indicators=indicators,
        regime=regime,
        support_levels=support_lvls,
        resistance_levels=resistance_lvls,
        params=params,
        warnings=warnings,
    )

    progress.progress(100, text="Done.")
    progress.empty()
    return result, hist_df, full_ind, expected_move


# ---------------------------------------------------------------------------
# Fetch expirations on first load or ticker change
# ---------------------------------------------------------------------------

auto_run = st.session_state.pop("auto_run", False)

# A blank ticker must never reach the provider — Run Screener used to
# short-circuit this check and request `?symbol=`, which returns HTTP 400.
if (run_button or auto_run) and not ticker_input:
    st.warning("Enter a ticker symbol to run the screener.", icon="👈")
    st.stop()

if ticker_input and (
    run_button
    or auto_run
    or ticker_input != st.session_state.get("last_ticker")
):
    with st.spinner(f"Loading expirations for {ticker_input}…"):
        try:
            exps = cached_expirations(ticker_input)
            default = pick_wheel_expiration(list(exps)) or dp.get_nearest_weekly_expiration(ticker_input)
            st.session_state["expirations"] = list(exps)
            st.session_state["default_exp"] = default
            st.session_state["last_ticker"] = ticker_input
            # Trigger rerun so the selectbox populates before we run the pipeline
            if not run_button and not auto_run:
                st.rerun()
        except ValueError as e:
            st.error(f"Could not load expirations: {e}", icon="🚨")
            st.stop()
        except Exception as e:  # network / HTTP / provider failure
            st.error(
                f"Could not load expirations for '{ticker_input}': {e}\n\n"
                "Check the ticker symbol, or switch Data Source to "
                "Yahoo Finance in the sidebar.",
                icon="🚨",
            )
            st.stop()

# Resolve the expiration to use
active_expiration = (
    st.session_state.get("default_exp")
    or ""
)

# ---------------------------------------------------------------------------
# Run pipeline when button pressed
# ---------------------------------------------------------------------------

if (run_button or auto_run) and ticker_input and active_expiration:
    try:
        with st.spinner(""):
            result, hist_df, full_ind, expected_move = run_pipeline(ticker_input, active_expiration, filter_params)
        st.session_state["result"] = result
        st.session_state["hist_df"] = hist_df
        st.session_state["full_ind"] = full_ind
        st.session_state["expected_move"] = expected_move

        # Update ticker history
        _hist = st.session_state.get("ticker_history", ["TSLA", "MSFT", "NVDA"])
        if ticker_input not in _hist:
            _hist.insert(0, ticker_input)
        elif _hist[0] != ticker_input:
            _hist.remove(ticker_input)
            _hist.insert(0, ticker_input)
        st.session_state["ticker_history"] = _hist[:3]
    except ValueError as e:
        st.error(str(e), icon="🚨")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}", icon="🚨")
        st.stop()

# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------

def _render_csp_capacity(result) -> None:
    """How many contracts of each recommended put fit the ledger's limits."""
    settings = ledger_ui.load_settings()
    account = float(settings.get("account_size") or 0)
    if account <= 0:
        st.caption(
            "📒 Set your account size in the Ledger tab to see how many "
            "contracts fit your exposure limits."
        )
        return
    try:
        _, ledger = ledger_ui.load_ledger()
    except Exception:
        return
    report = compute_exposure(
        list(ledger.positions.values()), account,
        settings["max_ticker_pct"], settings["max_total_pct"],
    )
    parts = []
    for rec in result.recommendations:
        strike = rec.option.contract.strike
        n = max_new_csp_contracts(
            report, result.quote.ticker, strike,
            settings["max_ticker_pct"], settings["max_total_pct"],
        )
        parts.append(f"${strike:g} put → **{n}** contract{'s' if n != 1 else ''}")
    st.caption(
        "📒 Capacity within your limits "
        f"({settings['max_ticker_pct']:.0%}/ticker, "
        f"{settings['max_total_pct']:.0%} total): " + "  ·  ".join(parts)
    )


result: object = st.session_state.get("result")
hist_df = st.session_state.get("hist_df")
full_ind = st.session_state.get("full_ind")
expected_move: float = st.session_state.get("expected_move", 0.0)

tab_scanner, tab_screener, tab_ledger, tab_position = st.tabs([
    "☀️ Daily Scanner",
    "\U0001f4ca Screener",
    "📒 Ledger",
    "\U0001f504 Manage Position",
])

with tab_scanner:
    render_scanner(dp=dp)

# Rendered before Manage Position: the ledger's "Manage" button queues a
# pre-fill that the position form applies later in this same run.
with tab_ledger:
    ledger_ui.render_ledger(dp=dp)

with tab_screener:
    if result is None:
        st.info(
            "Enter a ticker and click **Run Screener** to get started.",
            icon="👈",
        )
    else:
        # A. Market Overview
        st.divider()
        st.subheader(f"Market Overview — {result.quote.ticker}")
        st.caption(
            f"🗓️ Expiration {result.expiration} · {result.dte} DTE — picked "
            f"automatically: the earliest 7–14 DTE (wheel window), else 5–21."
        )
        rec_ui.render_market_overview(result, strategy, full_ind)

        # A2. Price Forecast (2-zone bias bar)
        rec_ui.render_price_forecast(result, strategy, full_ind)

        # A3. CC vs CSP — most premium under an assignment-probability limit
        st.divider()
        try:
            _calls, _puts = cached_chain(result.quote.ticker, result.expiration)
        except Exception:
            _calls, _puts = [], []
        render_assignment_panel(
            result, full_ind, _calls, _puts,
            min_open_interest=filter_params.min_open_interest,
            desired_buy_price=filter_params.desired_buy_price,
            cost_basis=filter_params.cost_basis,
            earnings_date=(
                result.quote.earnings_date
                or cached_earnings(result.quote.ticker)
            ),
        )

        # B. Technical Chart
        st.divider()
        st.subheader("Technical Chart")
        if full_ind is not None and hist_df is not None and not hist_df.empty:
            fig = build_price_chart(
                full_ind=full_ind,
                support_levels=result.support_levels,
                resistance_levels=result.resistance_levels,
                recommendations=result.recommendations,
                ticker=result.quote.ticker,
                expiration=result.expiration,
                current_price=result.quote.price,
                expected_move=expected_move,
            )
            st.plotly_chart(fig, use_container_width=True)

        # C. Recommendations
        st.divider()
        st.caption(
            "Showing **covered calls** — the Ledger holds your shares."
            if strategy == Strategy.COVERED_CALL else
            "Showing **cash-secured puts** — no shares of this ticker in the "
            "Ledger. Both sides are compared in 🎯 Max Premium above."
        )
        rec_ui.render_recommendations(result, strategy, expected_move=expected_move)
        if strategy == Strategy.CASH_SECURED_PUT and result.recommendations:
            _render_csp_capacity(result)

        # D. Ranked Option Table
        st.divider()
        rec_ui.render_option_table(result)

    # Footer
    st.divider()
    _footer_source = "Tradier (real-time)" if _use_tradier else "Yahoo Finance (15-min delayed)"
    st.caption(
        f"Data provided by {_footer_source}. "
        "This tool is for educational and informational purposes only. "
        "Not financial advice."
    )

with tab_position:
    render_position_manager(dp=dp)
