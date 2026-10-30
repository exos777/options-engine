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
