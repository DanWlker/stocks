"""Your 4% strategy vs the S&P 500 (dividends reinvested). The 4% is taken as nominal, guaranteed and free to move
in and out of (if it is a locked deposit, the switching rules in part D are not possible).

A. How often stocks beat 4% over 1-20 years, lump sum and monthly DCA, every start month 1928..
B. 4% after inflation, by decade
C. Signs at the start: stocks' win rate over 4% by CAPE, trend, drawdown, bill rate, past 12 months
D. Switching rules: hold 4% while a sign says so, else stocks. Decisions use data known two months before the month
   they apply to (month-average data would otherwise leak part of the month), and CAPE thresholds use only the
   CAPE history up to that point. CAPE ends 2023-09, so the CAPE rules stop there.
"""

import numpy as np
import pandas as pd

from monthly import cagr, max_drawdown, monthly, shiller
from report import section as make_section, table, text

FIXED = 1.04 ** (1 / 12) - 1
HORIZONS = [1, 2, 3, 5, 10, 15, 20]  # years


def switch(stock_ret, in_fixed):
    """Monthly growth of holding 4% in months flagged in_fixed (decided two months earlier), stocks otherwise."""
    flag = in_fixed.shift(2).fillna(in_fixed.iloc[0]).astype(bool)  # the first two months follow the first decision
    return pd.Series((1 + np.where(flag, FIXED, stock_ret)).cumprod(), index=stock_ret.index)


# self-check: flagged in month 1 means the 4% is held in month 3; months 0-1 follow month 0's decision (stocks)
_r = pd.Series([0.1, 0.1, 0.1, 0.1])
assert np.allclose(switch(_r, pd.Series([False, True, False, False])), [1.1, 1.21, 1.331, 1.331 * (1 + FIXED)])

def section():
    df = monthly()
    tr = (1 + df["stock_ret"]).cumprod()
    s = df["stock_ret"].to_numpy()

    rows_a, win = [], {}
    for h in HORIZONS:
        m = 12 * h
        g = (tr.shift(-m) / tr).dropna()
        ann = g ** (1 / h) - 1
        dca = []
        for i in range(len(s) - m + 1):  # $1 a month into each for h years
            v_s = v_f = 0.0
            for r in s[i:i + m]:
                v_s, v_f = (v_s + 1) * (1 + r), (v_f + 1) * (1 + FIXED)
            dca.append(v_s > v_f)
        win[h] = (ann > 0.04).mean()
        rows_a.append([f"{h} yr", f"{len(g)}", f"{win[h]:.0%}", f"{ann.median():+.1%}", f"{ann.quantile(.1):+.1%}",
                       f"{ann.min():+.1%}", f"{np.mean(dca):.0%}"])

    rows_b = []
    for dec in range(1930, 2030, 10):
        d = df[f"{dec}-01":f"{dec + 9}-12"]
        infl = cagr(d["cpi"].iloc[-1] / d["cpi"].iloc[0], len(d) - 1)
        st = cagr((1 + d["stock_ret"]).prod(), len(d))
        rows_b.append([f"{dec}s", f"{infl:+.1%}", f"{1.04 / (1 + infl) - 1:+.1%}", f"{(1 + st) / (1 + infl) - 1:+.1%}"])
    real10 = (1.04 ** 10) / (df["cpi"].shift(-120) / df["cpi"])
    lost10 = (real10.dropna() < 1).mean()

    price = df["price"]
    signs = {
        "CAPE": pd.cut(df["cape"], [0, 12, 17, 22, 27, 100], labels=["under 12", "12-17", "17-22", "22-27", "over 27"]),
        "earnings yield (1/CAPE) vs 4%": np.where(df["cape"].isna(), None, np.where(1 / df["cape"] < 0.04, "below 4%", "4% or more")),
        "price vs its 10-month average": np.where(price < price.rolling(10).mean(), "below", "above"),
        "below the record high": np.where(price / price.cummax() - 1 <= -0.2, "20% or more", "less than 20%"),
        "3-month bill rate": np.where(df["cash_ret"] * 12 >= 0.04, "4% or more", "under 4%"),
        "past 12 months, stocks": pd.cut(tr / tr.shift(12) - 1, [-1, -0.1, 0, 0.1, 0.2, 10],
                                         labels=["down 10%+", "down 0-10%", "up 0-10%", "up 10-20%", "up 20%+"]),
    }
    wins = {h: (tr.shift(-12 * h) / tr) > 1.04 ** h for h in (1, 5, 10)}
    have = {h: tr.shift(-12 * h).notna() for h in (1, 5, 10)}
    rows_c, cape10 = [], {}
    for name, b in signs.items():
        b = pd.Series(b, index=df.index)
        for cat in (b.cat.categories if hasattr(b, "cat") else sorted(b.dropna().unique())):
            sel = b == cat
            cells = [[f"{wins[h][sel & have[h]].mean():.0%}", f"n={int((sel & have[h]).sum())}"] for h in (1, 5, 10)]
            if name == "CAPE":
                cape10[cat] = (wins[10][sel & have[10]].mean(), int((sel & have[10]).sum()))
            rows_c.append([[str(cat), name]] + cells)

    sh = shiller()["cape"]
    pct80 = sh.expanding(min_periods=120).quantile(0.8).reindex(df.index)
    trend_out = price < price.rolling(10).mean()
    cape_end = df["cape"].last_valid_index()
    # name -> (in-4% flags, uses CAPE)
    rules = {
        "all stocks": (pd.Series(False, index=df.index), False),
        "all 4%": (pd.Series(True, index=df.index), False),
        "4% when CAPE in its top 20% so far": (df["cape"] > pct80, True),
        "4% when CAPE over 27": (df["cape"] > 27, True),
        "4% when price below 10-month average": (trend_out, False),
        "4% when CAPE top 20% and below average": ((df["cape"] > pct80) & trend_out, True),
    }
    tables_d = []
    for label, until in [(f"{df.index[0]} to {cape_end} (end of CAPE data)", cape_end), (f"{df.index[0]} to {df.index[-1]}, rules without CAPE", df.index[-1])]:
        d = df[:until]
        n = len(d)
        rows = []
        for name, (flag, uses_cape) in rules.items():
            if until != cape_end and uses_cape:
                continue
            f = flag[:until]
            g = switch(d["stock_ret"], f)
            real = g / (d["cpi"] / d["cpi"].iloc[0])
            rows.append([name, f"{cagr(g.iloc[-1], n):+.2%}", f"{cagr(real.iloc[-1], n):+.2%}", f"{max_drawdown(g):+.1%}",
                         f"{f.shift(2).fillna(f.iloc[0]).astype(bool).mean():.0%}"])
        tables_d.append(table(f"D. Switching rules, {label}", ["Rule", "A year", "After inflation", "Worst fall", "Months in 4%"], rows))

    # a rule that only wins in one stretch of history is luck, not a sign: split the CAPE era in three
    periods = [("1928-01", "1959-12"), ("1960-01", "1989-12"), ("1990-01", str(cape_end))]
    rows_p = []
    for name, (flag, _) in rules.items():
        cells = []
        for a, b in periods:
            g = switch(df["stock_ret"][a:b], flag[a:b])
            cells.append(f"{cagr(g.iloc[-1], len(g)):+.2%}")
        rows_p.append([name] + cells)

    lede = (f"Treating your strategy as a guaranteed 4% a year. Since {df.index[0]}, stocks with dividends beat it in "
            f"{win[1]:.0%} of 1-year periods, {win[5]:.0%} of 5-year periods and {win[20]:.0%} of 20-year periods. "
            f"A fixed 4% lost to inflation over {lost10:.0%} of 10-year periods. Of the signs tested, only valuation (CAPE) "
            f"mattered, and only for 5 to 10 years ahead: with CAPE over 27, stocks beat 4% over the next 10 years in "
            f"{cape10['over 27'][0]:.0%} of {cape10['over 27'][1]} start months, against {cape10['under 12'][0]:.0%} when CAPE was under 12.")
    return make_section(
        "four-percent", "Your 4% strategy vs stocks", "When is a fixed 4% a year worth it?", lede,
        [table("A. Stocks vs 4% a year, every start month", ["Held", "Start months", "Lump: stocks won", "Median stocks a year",
                                                             "10th percentile", "Worst", "Monthly DCA: stocks won"], rows_a),
         table("B. After inflation, by decade", ["Decade", "Inflation", "4% after inflation", "Stocks after inflation"], rows_b,
               note=f"Over every 10-year period, a fixed 4% lost to inflation in {lost10:.0%} of start months."),
         table("C. Signs at the start month: share of starts where stocks beat 4%", ["Sign", "Over 1 year", "Over 5 years", "Over 10 years"], rows_c,
               note="n is the number of start months with that sign. CAPE data ends 2023-09, so recent months have no CAPE reading."),
         *tables_d,
         table("D. The same rules by sub-period, a year (each restarted at the period start)",
               ["Rule"] + [f"{a[:4]}-{b[:4]}" for a, b in periods], rows_p,
               note="The combined rule was chosen after seeing the data; a sub-period where it does not win would mark it as luck.")],
        note="Switching rules decide two months ahead using only data known then, assume the 4% can be entered and left freely "
             "at no cost, and ignore taxes. Stocks are the S&P 500 with dividends reinvested (monthly.py). Computed in four_percent.py.")


if __name__ == "__main__":
    print(text(section()))
