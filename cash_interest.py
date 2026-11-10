"""#8 Interest on waiting cash: lump sum vs spreading it, and the bear-pause rule, redone with dividends and with
3-month Treasury bill interest on money that waits.

Three versions each: price only with 0% cash (as earlier), with dividends and 0% cash, with dividends and bill interest.
(a) lump vs spread uses monthly data (monthly.py): buys at month-average levels, cash earns that month's bill rate.
(b) the pause rule uses daily data: signals on the price index as before; holdings valued with daily_tr();
    each skipped month's $BUY earns bills from its month's first trading day until the catch-up.
"""

import numpy as np
import pandas as pd

from bear_pause import bear_pause
from dca import BUY
from lump_vs_spread import SPREADS
from monthly import daily_close, daily_tr, monthly
from pause_check import STARTS, month_starts, vs_dca
from report import section as make_section, table, text


def spread_ratio(level, cash_growth, s, n):
    """Spread over n months vs at once, on the lump. level[t] = price paid in month t; cash_growth[t] = growth of
    $1 of cash from the start of the data to the start of month t."""
    k = np.arange(s, s + n)
    return np.mean(cash_growth[k] / cash_growth[s] * level[s] / level[k]) - 1


def pause_vs_dca(price, tr, cash, months, start):
    """Final value of the pause rule relative to plain monthly DCA, both valued on tr; cash is the growth index of
    money in bills (None = 0%)."""
    price, tr = price[start:], tr[start:]
    cash = pd.Series(1.0, index=price.index) if cash is None else cash[start:]
    starts = month_starts(price)
    dca = BUY * np.sum(tr.iloc[-1] / tr[starts].to_numpy())
    pauses = bear_pause(price, months)[1]
    # each skipped month waits in cash from its own first trading day until the catch-up (or the end)
    value, skipped_all = 0.0, set()
    for _, s, r, k in pauses:
        sk = [d for d in starts if s <= d and (r is None or d < r)]
        assert len(sk) == k, (s, r, len(sk), k)
        skipped_all |= set(sk)
        end = r if r is not None else price.index[-1]
        cash_out = BUY * np.sum(cash[end] / cash[sk].to_numpy())
        value += cash_out * (tr.iloc[-1] / tr[end] if r is not None else 1)
    value += BUY * np.sum(tr.iloc[-1] / tr[[d for d in starts if d not in skipped_all]].to_numpy())
    return value / dca - 1


# self-checks: no change in prices or cash means no difference; a halving price makes spreading over 2 months +50%
assert np.isclose(spread_ratio(np.ones(5), np.ones(5), 0, 3), 0)
assert np.isclose(spread_ratio(np.array([100.0, 50]), np.ones(2), 0, 2), 0.5)
assert np.isclose(spread_ratio(np.array([100.0, 100]), np.array([1.0, 1.1]), 0, 2), 0.05)  # 10% interest on half

def section():
    df = monthly()
    tr_level = (1 + df["stock_ret"]).cumprod().to_numpy()
    cash_growth = np.concatenate([[1.0], (1 + df["cash_ret"]).cumprod().to_numpy()[:-1]])  # at the start of each month
    variants = [("price only, 0% cash", df["price"].to_numpy(), np.ones(len(df))),
                ("with dividends, 0% cash", tr_level, np.ones(len(df))),
                ("with dividends, bills on cash", tr_level, cash_growth)]
    rows_a, a12 = [], {}
    for n in SPREADS:
        cells = []
        for label, level, growth in variants:
            r = np.array([spread_ratio(level, growth, s, n) for s in range(len(df) - n + 1)])
            cells.append([f"{np.median(r):+.1%}", f"at once won {(r < 0).mean():.0%}"])
            if n == 12:
                a12[label] = (np.median(r), (r < 0).mean())
        rows_a.append([f"{n} months"] + cells)

    price, tr = daily_close("sp500_full"), daily_tr()
    days = price.index.to_period("M")
    n_days = pd.Series(1, index=price.index).groupby(days).transform("size").to_numpy()
    # the daily data starts 1927-12-30, before the monthly data: that one day earns nothing
    cash = pd.Series(np.cumprod((1 + df["cash_ret"].reindex(days).fillna(0).to_numpy()) ** (1 / n_days)), index=price.index)
    assert cash.notna().all()
    # cross-check: price only with 0% cash must reproduce pause_check.py
    for m in [12, 31]:
        assert np.isclose(pause_vs_dca(price, price, None, m, STARTS[0]), vs_dca(price, m)), m
    rows_b, b31 = [], {}
    for m in [6, 12, 18, 24, 31, 36]:
        cells = []
        for st in STARTS:
            v = [pause_vs_dca(price, price, None, m, st), pause_vs_dca(price, tr, None, m, st), pause_vs_dca(price, tr, cash, m, st)]
            cells.append([f"{v[2]:+.1%}", f"{v[0]:+.1%} price only · {v[1]:+.1%} dividends, 0% cash"])
            if m == 31:
                b31[st[:4]] = v
        rows_b.append([f"{m} months"] + cells)

    with_bills, no_cash = a12["with dividends, bills on cash"], a12["with dividends, 0% cash"]
    assert with_bills[0] < 0 and no_cash[0] < 0  # the sentence below says "behind"
    lede = (f"Earlier results assumed money waiting to be invested earned 0%. With dividends and 3-month Treasury bill interest "
            f"on that cash, investing a lump sum at once still beat spreading it over 12 months in {with_bills[1]:.0%} of start months; "
            f"the median 12-month spread ended {-with_bills[0]:.1%} behind on the lump instead of {-no_cash[0]:.1%}. The 31-month pause rule "
            f"still ends {b31['1954'][2]:+.1%} against plain DCA from 1954.")
    return make_section(
        "cash-interest", "Interest on waiting cash", "Does interest rescue waiting?", lede,
        [table("Spreading a lump sum over N months vs investing it at once (median, on the lump)",
               ["Spread over"] + [v[0].capitalize() for v in variants], rows_a,
               note="Every start month from 1928; monthly data, buying at month-average levels."),
         table("Bear-pause rule vs plain monthly DCA, final value (with dividends and bill interest)",
               ["Pause"] + [("All years" if i == 0 else f"From {st[:4]}") for i, st in enumerate(STARTS)], rows_b,
               note="Second line: price only with 0% cash (as in the reality check above), and with dividends but 0% cash. "
                    "Signals use the price index; each skipped month's $1,000 earns bills until the catch-up.")],
        note="Bills: NBER short-term government rate to 1933, then FRED's 3-month Treasury bill rate. Computed in cash_interest.py.")


if __name__ == "__main__":
    print(text(section()))
