"""Monthly dip rule: each month buy BUY on the first close `step` below the month's first close,
or on the month's last trading day if that never happens. Compared with the same money as
monthly DCA on the first day, monthly DCA on the last day, and a lump sum on day one.
"""

from pathlib import Path

import pandas as pd

from dca import BUY, PEAK_YEAR, STEP, result

STEPS = [STEP, 0.04]


def month_dip(close, step=STEP):
    buys, dips = [], []
    for _, m in close.groupby(close.index.to_period("M")):
        hit = m[m <= m.iloc[0] * (1 - step)]
        buys.append(hit.index[0] if len(hit) else m.index[-1])
        if len(hit):
            dips.append(hit.index[0])
    return buys, dips


# self-check: Jan dips to 89 on day 3 (buy there, not the deeper 85); Feb never dips 10%, buy on its last day
_t = pd.Series([100, 95, 89, 85, 100, 95, 92.0],
               index=pd.to_datetime(["2001-01-02", "2001-01-03", "2001-01-04", "2001-01-31",
                                     "2001-02-01", "2001-02-15", "2001-02-28"]))
assert month_dip(_t) == ([pd.Timestamp("2001-01-04"), pd.Timestamp("2001-02-28")], [pd.Timestamp("2001-01-04")])
# at 4% both months dip on their second day (-5%)
assert month_dip(_t, 0.04)[1] == [pd.Timestamp("2001-01-03"), pd.Timestamp("2001-02-15")]

if __name__ == "__main__":
    data = Path(__file__).parent / "data"
    for name in ["depression_sp500", "dotcom_sp500", "dotcom_nasdaq"]:
        full = pd.read_csv(data / f"{name}.csv", parse_dates=["Date"], index_col="Date")["Close"]
        for window, close in [("full range", full), ("from peak", full[full[PEAK_YEAR[name]].idxmax():])]:
            months = close.groupby(close.index.to_period("M"))
            print(f"\n== {name} [{window}]  {close.index[0].date()} -> {close.index[-1].date()}  {months.ngroups} months")
            runs = []
            for step in STEPS:
                buys, dips = month_dip(close, step)
                print(f"    {step:.0%} dip: dip buys in {len(dips)}, month-end fallback in {len(buys) - len(dips)}")
                runs.append((f"dip {step:.0%}", buys, BUY))
            runs += [
                ("1st day", list(months.head(1).index), BUY),
                ("last day", list(months.tail(1).index), BUY),
                ("lump sum", [close.index[0]], BUY * months.ngroups),
            ]
            for label, ds, amt in runs:
                inv, val, avg, irr = result(close, ds, amt)
                print(f"    {label:<9} invested ${inv:>9,.0f}  final ${val:>10,.0f}  "
                      f"return {val / inv - 1:+8.1%}  XIRR {irr:+7.2%}/yr  avg cost {avg:9.2f}")
