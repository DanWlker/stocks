"""#3 Trend rule: hold the S&P 500 while its close is above its 200-day average, otherwise hold 3-month
Treasury bills. The signal at one day's close sets the next day's position (no look-ahead).
Stocks include dividends (monthly.daily_tr); bills earn the monthly rate spread over its trading days.
Trading costs and taxes are ignored, which flatters the rule.
"""

import numpy as np
import pandas as pd

from monthly import cagr, daily_close, daily_tr, max_drawdown, monthly
from report import section as make_section, table, text

WINDOW = 200
CRASHES = [("1929 crash", "1929-09-16", "1932-06-01"), ("1937", "1937-03-10", "1938-03-31"),
           ("1973-74", "1973-01-11", "1974-10-03"), ("1987", "1987-08-25", "1987-12-04"),
           ("2000-02", "2000-03-24", "2002-10-09"), ("2007-09", "2007-10-09", "2009-03-09"),
           ("2020", "2020-02-19", "2020-03-23"), ("2022", "2022-01-03", "2022-10-12")]


def simulate(price, tr, cash_daily, window=WINDOW):
    """Daily growth of buy-and-hold and of the trend rule, from the first day the average exists."""
    above = price > price.rolling(window).mean()
    in_stocks = above.shift(1, fill_value=True)  # yesterday's signal decides today
    stock_ret = tr.pct_change().fillna(0)
    rule_ret = np.where(in_stocks, stock_ret, cash_daily)
    start = window
    hold = (1 + stock_ret.iloc[start:]).cumprod()
    rule = pd.Series(np.cumprod(1 + rule_ret[start:]), index=hold.index)
    return hold, rule, in_stocks.iloc[start:]


# self-check on a rise then fall, 3-day average, measured from the close of 3:
# the close of 4 (after 5) is below its average 4.33, so the rule is out from the next day on
_p = pd.Series([1, 2, 3, 4, 5, 4, 3, 2, 1.0], index=pd.bdate_range("2001-01-01", periods=9))
_h, _r, _in = simulate(_p, _p, pd.Series(0.0, index=_p.index), window=3)
assert list(_in) == [True, True, True, False, False, False]
assert np.isclose(_h.iloc[-1], 1 / 3) and np.isclose(_r.iloc[-1], 4 / 3)  # hold rides back down to 1; the rule keeps 4

def section():
    price, tr = daily_close("sp500_full"), daily_tr()
    m = monthly()
    # each trading day earns its month's bill return spread evenly over the month's trading days
    days = price.index.to_period("M")
    n_days = pd.Series(1, index=price.index).groupby(days).transform("size")
    cash = pd.Series(((1 + m["cash_ret"].reindex(days).to_numpy()) ** (1 / n_days.to_numpy())) - 1, index=price.index).fillna(0)
    hold, rule, inside = simulate(price, tr, cash)
    switches = int((inside != inside.shift()).sum() - 1)
    months = (hold.index[-1] - hold.index[0]).days / 30.4375
    rows_s, stat = [], {}
    for name, s in [("Buy and hold", hold), ("Trend rule", rule)]:
        yearly = s.groupby(s.index.year).last().pct_change()
        stat[name] = (cagr(s.iloc[-1], months), max_drawdown(s))
        rows_s.append([name, f"{stat[name][0]:+.2%}", f"${10_000 * s.iloc[-1]:,.0f}", f"{stat[name][1]:+.1%}", f"{yearly.min():+.1%}"])

    rows_c = [[[label, f"{a} to {b}"], f"{hold[b] / hold[a] - 1:+.1%}", f"{rule[b] / rule[a] - 1:+.1%}"] for label, a, b in CRASHES]

    rows_d, behind = [], []
    for dec in range(1930, 2030, 10):
        h = hold[str(dec):str(dec + 9)]
        r = rule[h.index]
        n = (h.index[-1] - h.index[0]).days / 30.4375
        gh, gr = cagr(h.iloc[-1] / h.iloc[0], n), cagr(r.iloc[-1] / r.iloc[0], n)
        rows_d.append([f"{dec}s", f"{gh:+.1%}", f"{gr:+.1%}", f"{gr - gh:+.1%}"])
        if gr < gh:
            behind.append(f"{dec}s")

    windows = [50, 100, 150, 200, 250, 300]
    common = price.index[max(windows)]  # every average exists from here
    n = (price.index[-1] - common).days / 30.4375
    h = hold[common:] / hold[common]
    rows_w = [["Buy and hold", f"{cagr(h.iloc[-1], n):+.2%}", f"{max_drawdown(h):+.1%}", "—"]]
    for w in windows:
        _, r, ins = simulate(price, tr, cash, window=w)
        r, ins = r[common:] / r[common], ins[common:]
        rows_w.append([f"{w}-day average", f"{cagr(r.iloc[-1], n):+.2%}", f"{max_drawdown(r):+.1%}", f"{int((ins != ins.shift()).sum() - 1)}"])

    hr, tr_ = stat["Buy and hold"], stat["Trend rule"]
    lede = (f"Hold the S&P 500 while it closes above its {WINDOW}-day average, otherwise 3-month Treasury bills. From "
            f"{hold.index[0].date()} it returned {tr_[0]:+.2%} a year against {hr[0]:+.2%} for buy and hold, with a worst fall of "
            f"{tr_[1]:.1%} instead of {hr[1]:.1%}. It held stocks {inside.mean():.0%} of the time and switched {switches} times "
            f"({switches / (months / 12):.1f} a year), and it trailed buy and hold in {len(behind)} of 10 decades ({', '.join(behind)}).")
    return make_section(
        "trend-rule", "Timing with a moving average", "Can a 200-day average dodge crashes?", lede,
        [table(f"Whole period, {hold.index[0].date()} to {hold.index[-1].date()}", ["", "A year", "$10,000 became", "Worst fall", "Worst calendar year"], rows_s),
         table("Peak to bottom of each crash", ["Crash", "Buy and hold", "Trend rule"], rows_c),
         table("By decade, a year", ["Decade", "Buy and hold", "Trend rule", "Rule minus hold"], rows_d),
         table(f"Other averages, all from {common.date()}", ["", "A year", "Worst fall", "Switches"], rows_w)],
        note="The signal at one close sets the next day's position. Stocks include dividends; bills earn the monthly "
             "3-month rate. Trading costs and taxes are ignored, which flatters the rule: each switch is a sale. Computed in trend_rule.py.")


if __name__ == "__main__":
    print(text(section()))
