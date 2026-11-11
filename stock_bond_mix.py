"""#9 Stocks + bonds: portfolios from 100% S&P 500 (dividends reinvested) to 100% 10-year Treasuries, from 1928.

Monthly data (monthly.py). Rebalanced back to the target mix every month unless stated; costs and taxes ignored.
Drawdowns use month-average stock levels, so they are a little shallower than on daily closes.
"""

import numpy as np
import pandas as pd

from monthly import cagr, max_drawdown, monthly
from report import section as make_section, table, text

MIXES = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]  # share in stocks
CRISES = [("1929-32", "1929-09", "1932-06"), ("1937-38", "1937-03", "1938-03"), ("1973-74", "1973-01", "1974-09"),
          ("2000-02", "2000-03", "2002-09"), ("2007-09", "2007-10", "2009-02"), ("2022", "2022-01", "2022-09")]


def growth(df, w, rebalance=True):
    """Monthly value of $1 in w stocks / 1-w bonds."""
    if rebalance:
        return (1 + w * df["stock_ret"] + (1 - w) * df["bond_ret"]).cumprod()
    return w * (1 + df["stock_ret"]).cumprod() + (1 - w) * (1 + df["bond_ret"]).cumprod()


def recovery_months(values):
    """Months from the peak before the worst drawdown until back at that peak (None if never)."""
    dd = values / values.cummax() - 1
    trough = dd.idxmin()
    peak = values[:trough].idxmax()
    back = values[trough:][values[trough:] >= values[peak]]
    return None if back.empty else (back.index[0] - peak).n


# self-checks: rebalanced 50/50 of +10% and 0% is +5% a month; a fall and full recovery takes its months
_d = pd.DataFrame({"stock_ret": [0.1, 0.1], "bond_ret": [0.0, 0.0]})
assert np.allclose(growth(_d, 0.5), [1.05, 1.05 ** 2])
assert recovery_months(pd.Series([1, 2, 1, 1.5, 2.0], index=pd.period_range("2001-01", periods=5, freq="M"))) == 3

def section():
    df = monthly()
    real = df["cpi"] / df["cpi"].iloc[0]
    n = len(df)
    rows = [(f"{int(w * 100)}/{int(round((1 - w) * 100))}", growth(df, w)) for w in MIXES]
    rows.append(("60/40 never rebalanced", growth(df, 0.6, rebalance=False)))
    rows_m, stat = [], {}
    for label, g in rows:
        rg = g / real
        rec = recovery_months(g)
        ten = rg.shift(-120) / rg - 1
        stat[label] = (cagr(g.iloc[-1], n), max_drawdown(g), rec)
        rows_m.append([label, f"{stat[label][0]:+.2%}", f"{cagr(rg.iloc[-1], n):+.2%}", f"{stat[label][1]:+.1%}",
                       f"{rec / 12:.1f} yrs" if rec is not None else "not yet", f"{(g / g.shift(12) - 1).min():+.1%}",
                       f"{(ten.dropna() < 0).mean():.0%}"])

    rows_c = [[[name, f"{a} to {b}"]] + [f"{g[b] / g[a] - 1:+.1%}" for _, g in rows[:-1]] for name, a, b in CRISES]
    crisis = {name: {label: g[b] / g[a] - 1 for label, g in rows[:-1]} for name, a, b in CRISES}

    rows_y = []
    for y in range(1973, 1982):
        d = df.loc[str(y)]
        rows_y.append([str(y), f"{(1 + d['stock_ret']).prod() - 1:+.1%}", f"{(1 + d['bond_ret']).prod() - 1:+.1%}",
                       f"{d['cpi'].iloc[-1] / df['cpi'][f'{y - 1}-12'] - 1:+.1%}"])

    a, b = stat["100/0"], stat["60/40"]
    lede = (f"Since {df.index[0]}, all stocks returned {a[0]:+.2%} a year with a worst fall of {a[1]:.1%}; 60% stocks and 40% "
            f"10-year Treasuries, rebalanced monthly, returned {b[0]:+.2%} with a worst fall of {b[1]:.1%} and recovered in "
            f"{b[2] / 12:.1f} years instead of {a[2] / 12:.1f}. Bonds cushioned 2000-02 ({crisis['2000-02']['60/40']:+.1%} against "
            f"{crisis['2000-02']['100/0']:+.1%}) and 2007-09, but in 2022 every mix fell between "
            f"{max(crisis['2022'].values()):+.1%} and {min(crisis['2022'].values()):+.1%}.")
    return make_section(
        "stock-bond-mix", "Stocks and bonds", "How much does a bond mix soften crashes?", lede,
        [table(f"Stocks/bonds mixes, {df.index[0]} to {df.index[-1]}, rebalanced monthly",
               ["Stocks/bonds", "A year", "After inflation", "Worst fall", "Recovered in", "Worst 12 months", "10-year periods losing after inflation"],
               rows_m),
         table("In each crisis, peak month to bottom month", ["Crisis"] + [label for label, _ in rows[:-1]], rows_c),
         table("The 1970s by year: bonds rose in dollars but lost to inflation", ["Year", "Stocks", "Bonds", "Inflation"], rows_y)],
        note="Bonds are modelled as a 10-year Treasury bought at par each month and sold a month later at the new yield, not a real "
             "fund. Costs and taxes ignored. Monthly data (monthly.py). Computed in stock_bond_mix.py.")


if __name__ == "__main__":
    print(text(section()))
