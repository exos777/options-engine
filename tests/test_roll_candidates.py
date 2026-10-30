"""Roll candidates must net the new leg's credit against closing the old one."""

from __future__ import annotations

from datetime import date, timedelta

from scoring.position_decision import OpenPosition
from strategies.models import OptionContract
from ui.position_management import _fetch_roll_candidates

LEG2 = (date.today() + timedelta(days=8)).isoformat()


class _DP:
    def get_option_chain(self, ticker, exp):
        put = lambda k, bid: OptionContract(
            strike=k, expiration=exp, option_type="put", bid=bid, ask=bid + 0.2,
            last=bid, volume=100, open_interest=500, delta=-0.25,
        )
        return [], [put(450, 11.00), put(440, 6.00)]


def test_roll_credit_is_sell_to_open_bid_minus_buy_to_close_ask():
    pos = OpenPosition(strategy="CSP", strike=460, original_premium=4.50,
                       current_bid=9.80, current_ask=10.10)
    cands = {c.strike: c for c in _fetch_roll_candidates(_DP(), "MSFT", pos, LEG2)}
    assert cands[450].buy_to_close == 10.10
    assert cands[450].sell_to_open == 11.00
    assert round(cands[450].roll_credit, 2) == 0.90
    assert round(cands[440].roll_credit, 2) == -4.10       # a debit, not a credit


class _WideDP:
    def get_option_chain(self, ticker, exp):
        put = lambda k: OptionContract(
            strike=k, expiration=exp, option_type="put", bid=1.0, ask=1.1,
            last=1.0, volume=100, open_interest=500, delta=-0.10,
        )
        return [], [put(k) for k in (360, 380, 400, 470, 520)]


def test_csp_search_reaches_20pct_below_the_strike():
    pos = OpenPosition(strategy="CSP", strike=460, original_premium=4.50,
                       current_bid=0.9, current_ask=1.0)
    strikes = {c.strike for c in _fetch_roll_candidates(_WideDP(), "MSFT", pos, LEG2)}
    assert strikes == {380, 400, 470}      # 368–506: 360 and 520 out of range
