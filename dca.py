"""Buy-on-10%-drop rules vs lump sum and monthly DCA with the same total money.

Each rule buys BUY dollars at the day's close when triggered:
  ladder - buy on day one, then each time the close is 10% below the previous buy price
  peak   - buy once per 10% tier (-10%, -20%, ...) below the running high; tiers reset at a new high
  daily  - buy on any day the close fell 10% or more from the previous close
Comparisons invest the rule's total: all on day one (lump), or split evenly over the
first trading day of every month (monthly). Everything is valued at the last close.
Each dataset runs over its full range and again from the bubble peak (highest close in PEAK_YEAR).
XIRR is the annualised return given when each dollar went in (actual/365).
"""

from pathlib import Path

import pandas as pd

BUY = 1000.0
STEP = 0.10


def ladder(close):
    buys, last = [close.index[0]], close.iloc[0]
    for date, price in close.iloc[1:].items():
        if price <= last * (1 - STEP):
            buys.append(date)
            last = price
    return buys


def peak(close):
    buys, high, tier = [], close.iloc[0], 0
    for date, price in close.items():
        if price > high:
            high, tier = price, 0
        # round() guards against 0.3/0.1 = 2.999... float error
        reached = int(round((1 - price / high) / STEP, 9))
        if reached > tier:
            buys.append(date)  # one buy per day even if several tiers were crossed at once
            tier = reached
    return buys


def daily(close):
    ret = close.pct_change()
    return list(ret[ret <= -STEP].index)


def xirr(dates, amount, end, value):
    # NPV scaled to the end date is decreasing in the rate (all outflows precede the one inflow), so bisect
    def npv(r):
        return value - sum(amount * (1 + r) ** ((end - d).days / 365) for d in dates)

    lo, hi = -0.99, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if npv(mid) > 0 else (lo, mid)
    return mid


def result(close, dates, amount):
    shares = sum(amount / close[d] for d in dates)
    invested = amount * len(dates)
    value = shares * close.iloc[-1]
    return invested, value, invested / shares, xirr(dates, amount, close.index[-1], value)


# self-check on a toy series: 100 -> 89 -> 95 -> 80 -> 79 -> 110 -> 98
_t = pd.Series([100, 89, 95, 80, 79, 110, 98.0], index=pd.date_range("2000-01-01", periods=7))
assert [d.day for d in ladder(_t)] == [1, 2, 4]  # 89 <= 90, 80 <= 80.1
assert [d.day for d in peak(_t)] == [2, 4, 7]  # -11% tier1, -20% tier2, new high 110 then -10.9%
assert [d.day for d in daily(_t)] == [2, 4, 7]  # -11%, -15.8%, -10.9%
_d = pd.Timestamp("2001-01-01")
assert abs(xirr([_d], 1000, _d + pd.Timedelta(days=365), 1100) - 0.10) < 1e-9
assert abs(xirr([_d], 1000, _d + pd.Timedelta(days=730), 810) - (-0.10)) < 1e-9

PEAK_YEAR = {"depression_sp500": "1929", "dotcom_sp500": "2000", "dotcom_nasdaq": "2000"}

if __name__ == "__main__":
    data = Path(__file__).parent / "data"
    for name in ["depression_sp500", "dotcom_sp500", "dotcom_nasdaq"]:
        full = pd.read_csv(data / f"{name}.csv", parse_dates=["Date"], index_col="Date")["Close"]
        for window, close in [("full range", full), ("from peak", full[full[PEAK_YEAR[name]].idxmax():])]:
            month_starts = list(close.groupby(close.index.to_period("M")).head(1).index)
            print(f"\n== {name} [{window}]  {close.index[0].date()} -> {close.index[-1].date()}  "
                  f"(start {close.iloc[0]:.2f}, end {close.iloc[-1]:.2f})")
            for rule in (ladder, peak, daily):
                dates = rule(close)
                if not dates:
                    print(f"  {rule.__name__}: never triggered")
                    continue
                total = BUY * len(dates)
                print(f"  {rule.__name__}: {len(dates)} buys, first {dates[0].date()}, last {dates[-1].date()}")
                print(f"    buy dates: {', '.join(str(d.date()) for d in dates)}")
                for label, ds, amt in [
                    (rule.__name__, dates, BUY),
                    ("lump sum", [close.index[0]], total),
                    ("monthly", month_starts, total / len(month_starts)),
                ]:
                    inv, val, avg, irr = result(close, ds, amt)
                    print(f"    {label:<9} invested ${inv:>9,.0f}  final ${val:>10,.0f}  "
                          f"return {val / inv - 1:+8.1%}  XIRR {irr:+7.2%}/yr  avg cost {avg:9.2f}")
