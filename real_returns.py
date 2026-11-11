"""#7 Inflation-adjusted returns: how long S&P 500 crashes took to recover, measured three ways, and returns by decade.

Monthly data (monthly.py): price only, total return (dividends reinvested), and real total return
(total return / consumer prices). Recovery = first month back at the previous record. Month-average levels
smooth daily swings, so depths here are a little shallower than on daily closes.
"""

import pandas as pd

from crashes import from_ath
from monthly import cagr, monthly
from report import section as make_section, table, text


def levels(df):
    tr = (1 + df["stock_ret"]).cumprod()
    return {"price": df["price"], "with dividends": tr, "with dividends, after inflation": tr / df["cpi"]}


def section():
    df = monthly()
    lv = {k: v.set_axis(v.index.to_timestamp()) for k, v in levels(df).items()}
    found = {k: from_ath(v) for k, v in lv.items()}
    # line the three up on the crashes the real measure finds (it sees the most)
    rows_r, years = [], {}
    for p, t, r in found["with dividends, after inflation"]:
        depth = lv["with dividends, after inflation"][t] / lv["with dividends, after inflation"][p] - 1
        row = []
        for k in lv:
            match = [c for c in found[k] if c[0] <= t and (c[2] is None or c[2] > p)]  # same episode in this measure
            if not match:
                row.append("no 20% fall")
            else:
                pk, _, rec = match[0]
                if abs((pk - p).days) > 365:  # this measure was still below an older record: same long fall
                    row.append(f"in {pk.year} fall")
                else:
                    row.append("not yet" if rec is None else f"{(rec - pk).days / 365.25:.1f} yrs")
        years[p.year] = row
        rows_r.append([p.date().isoformat()[:7], f"{depth:+.0%}", *row])

    rows_d, bills_neg = [], 0
    for dec in range(1930, 2030, 10):
        d = df[f"{dec}-01":f"{dec + 9}-12"]
        n = len(d)
        st = cagr((1 + d["stock_ret"]).prod(), n)
        infl = cagr(d["cpi"].iloc[-1] / d["cpi"].iloc[0], n - 1)
        bills = cagr((1 + d["cash_ret"]).prod(), n)
        bills_real = (1 + bills) / (1 + infl) - 1
        bills_neg += bills_real < 0
        rows_d.append([f"{dec}s", f"{st:+.1%}", f"{infl:+.1%}", f"{(1 + st) / (1 + infl) - 1:+.1%}", f"{bills_real:+.1%}"])
    n = len(df)
    st, infl = cagr((1 + df["stock_ret"]).prod(), n), cagr(df["cpi"].iloc[-1] / df["cpi"].iloc[0], n - 1)
    rows_d.append(["All", f"{st:+.1%}", f"{infl:+.1%}", f"{(1 + st) / (1 + infl) - 1:+.1%}", ""])

    lede = (f"Counting inflation changes how long recoveries took. With dividends, the 1973 crash was back at its record after "
            f"{years[1973][1]}, but {years[1973][2]} after inflation; 2000 took {years[2000][1]} against {years[2000][2]}. "
            f"1929 went the other way ({years[1929][1]} against {years[1929][2]}) because prices fell in the 1930s. Since 1928 stocks "
            f"returned {(1 + st) / (1 + infl) - 1:+.1%} a year after inflation, while bills lost to inflation in {bills_neg} of 10 decades.")
    return make_section(
        "inflation", "After inflation", "How long recoveries really took", lede,
        [table("Falls of 20%+ from a record, S&P 500: years from the peak until back at the record",
               ["Peak (after inflation)", "Fall after inflation", "Price only", "With dividends", "With dividends, after inflation"], rows_r,
               note="'In 1929 fall': that measure was still below its 1929 record, so this was part of the same long fall."),
         table("Returns by decade, a year", ["Decade", "Stocks with dividends", "Inflation", "Stocks after inflation", "Bills after inflation"], rows_d)],
        note="Monthly data (monthly.py); month-average levels smooth daily swings, so falls look slightly shallower than on daily "
             "closes. Inflation is US consumer prices (FRED). Computed in real_returns.py.")


if __name__ == "__main__":
    print(text(section()))
