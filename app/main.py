"""
Options Screener — Streamlit entry point.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on the path when running from the app/ subdirectory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

from data import provider as dp
from indicators.technical import calculate_full_indicators
from indicators.support_resistance import find_support_resistance
from scoring.regime import classify_regime
from scoring.covered_call import score_covered_calls
from scoring.cash_secured_put import score_cash_secured_puts
from scoring.engine import run_screener
from strategies.models import FilterParams, RiskProfile, Strategy
from ui.chart import build_price_chart
from ui import recommendations as rec_ui


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Options Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📈 Weekly Options Screener")
st.caption("Covered Calls & Cash-Secured Puts — powered by yfinance")


# ---------------------------------------------------------------------------
# Sidebar — Input Panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    run_button = st.button("🔍 Run Screener", type="primary", use_container_width=True)

    st.divider()

    ticker_input = st.text_input(
        "Ticker Symbol",
        value="AAPL",
        max_chars=10,
        help="Enter a US stock ticker (e.g. AAPL, MSFT, SPY)",
    ).upper().strip()

    strategy_choice = st.radio(
        "Strategy",
        options=[s.value for s in Strategy],
        index=0,
        horizontal=True,
    )
    strategy = Strategy(strategy_choice)

    st.divider()
    st.subheader("Expiration")

    # We fetch expirations after the run button; store in session state
    expiration_options: list[str] = st.session_state.get("expirations", [])
    default_exp: str = st.session_state.get("default_exp", "")

    if expiration_options:
        exp_index = (
            expiration_options.index(default_exp)
            if default_exp in expiration_options
            else 0
        )
        selected_expiration = st.selectbox(
            "Expiration Date",
            options=expiration_options,
            index=exp_index,
            key="expiration_select",
        )
    else:
        selected_expiration = ""
        st.info("Run screener to load expirations.")

    st.divider()
    st.subheader("Position Details")

    if strategy == Strategy.COVERED_CALL:
        shares_owned = st.number_input("Shares Owned", min_value=0, value=100, step=100)
        cost_basis = st.number_input(
            "Cost Basis per Share ($)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            help="Leave 0 to ignore cost basis filtering",
        )
        cost_basis_val = cost_basis if cost_basis > 0 else None
        allow_below_basis = st.checkbox(
            "Allow strikes below cost basis",
            value=False,
            help="If unchecked, strikes below break-even are filtered out",
        )
        cash_available = None
        desired_buy_price = None
    else:
        cash_available = st.number_input(
            "Cash Available ($)",
            min_value=0.0,
            value=10000.0,
            step=500.0,
        )
        desired_buy_price_input = st.number_input(
            "Desired Buy Price ($)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            help="Leave 0 to ignore buy price preference",
        )
        desired_buy_price = desired_buy_price_input if desired_buy_price_input > 0 else None
        shares_owned = 0
        cost_basis_val = None
        allow_below_basis = False

    st.divider()
    st.subheader("Risk Profile")

    risk_choice = st.select_slider(
        "Risk Tolerance",
        options=[r.value for r in RiskProfile],
        value=RiskProfile.BALANCED.value,
    )
    risk_profile = RiskProfile(risk_choice)

    st.divider()
    st.subheader("Filters")

    max_delta = st.slider(
        "Max Delta",
        min_value=0.05,
        max_value=0.60,
        value={"Conservative": 0.30, "Balanced": 0.40, "Aggressive": 0.50}[risk_choice],
        step=0.05,
        help="Maximum absolute delta to consider",
    )
    min_oi = st.number_input(
        "Min Open Interest",
        min_value=0,
        value=50,
        step=10,
    )
    max_spread_pct = st.slider(
        "Max Bid-Ask Spread %",
        min_value=5,
        max_value=80,
        value=30,
        step=5,
        help="Maximum bid-ask spread as % of mid price",
    )
    min_premium = st.number_input(
        "Min Premium ($)",
        min_value=0.0,
        value=0.10,
        step=0.05,
    )



# ---------------------------------------------------------------------------
# Build FilterParams from sidebar inputs
# ---------------------------------------------------------------------------

filter_params = FilterParams(
    strategy=strategy,
    risk_profile=risk_profile,
    max_delta=max_delta,
    min_open_interest=min_oi,
    max_spread_pct=max_spread_pct / 100,
    min_premium=min_premium,
    shares_owned=shares_owned,
    cost_basis=cost_basis_val,
    allow_below_basis=allow_below_basis,
    cash_available=cash_available,
    desired_buy_price=desired_buy_price,
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

    if params.strategy == Strategy.COVERED_CALL:
        scored = score_covered_calls(
            calls, quote.price, dte,
            indicators, regime,
            support_lvls, resistance_lvls,
            params,
        )
    else:
        scored = score_cash_secured_puts(
            puts, quote.price, dte,
            indicators, regime,
            support_lvls, resistance_lvls,
            params,
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
    return result, hist_df, full_ind


# ---------------------------------------------------------------------------
# Fetch expirations on first load or ticker change
# ---------------------------------------------------------------------------

if run_button or (
    ticker_input
    and ticker_input != st.session_state.get("last_ticker")
):
    with st.spinner(f"Loading expirations for {ticker_input}…"):
        try:
            exps = cached_expirations(ticker_input)
            default = dp.get_nearest_weekly_expiration(ticker_input)
            st.session_state["expirations"] = list(exps)
            st.session_state["default_exp"] = default
            st.session_state["last_ticker"] = ticker_input
            # Trigger rerun so the selectbox populates before we run the pipeline
            if not run_button:
                st.rerun()
        except ValueError as e:
            st.error(f"Could not load expirations: {e}", icon="🚨")
            st.stop()

# Resolve the expiration to use
active_expiration = (
    st.session_state.get("expiration_select")
    or st.session_state.get("default_exp")
    or ""
)

# ---------------------------------------------------------------------------
# Run pipeline when button pressed
# ---------------------------------------------------------------------------

if run_button and ticker_input and active_expiration:
    try:
        with st.spinner(""):
            result, hist_df, full_ind = run_pipeline(ticker_input, active_expiration, filter_params)
        st.session_state["result"] = result
        st.session_state["hist_df"] = hist_df
        st.session_state["full_ind"] = full_ind
    except ValueError as e:
        st.error(str(e), icon="🚨")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}", icon="🚨")
        st.stop()

# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------

result: object = st.session_state.get("result")
hist_df = st.session_state.get("hist_df")
full_ind = st.session_state.get("full_ind")

if result is None:
    st.info(
        "Enter a ticker and click **Run Screener** to get started.",
        icon="👈",
    )
    st.stop()

# A. Market Overview
st.divider()
st.subheader(f"Market Overview — {result.quote.ticker}")
rec_ui.render_market_overview(result, strategy)

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
    )
    st.plotly_chart(fig, use_container_width=True)

# C. Recommendations
st.divider()
rec_ui.render_recommendations(result, strategy)

# D. Ranked Option Table
st.divider()
rec_ui.render_option_table(result)

# Footer
st.divider()
st.caption(
    "Data provided by Yahoo Finance (15-min delayed). "
    "This tool is for educational and informational purposes only. "
    "Not financial advice."
)
