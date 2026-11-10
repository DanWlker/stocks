"""You have a lump sum L and keep adding $BUY a month. Invest L at once, or add an extra $X a month until
L is used up, then carry on with $BUY a month?

Both plans add the same $BUY every month, so they differ only in how L goes in: all on day one, or spread
over N = L / X months (the last month takes what is left). Cash waiting to be invested earns 0%.
On the L part alone, spreading beats investing at once by  P0 * mean(1 / P_k) - 1  where P0 is day one's
price and P_k the prices of the N spread buys: it wins when the average price paid is below day one's.
That depends only on N, not on L or X separately, nor on the end date.

  1. every start month in history: spread over N months vs at once, on the L part
  2. starting at a crash peak
  3. whole portfolio after 10 years for a grid of lump sizes L and extra amounts X
"""

from pathlib import Path

import numpy as np
import pandas as pd

from dca import BUY

HORIZON = 120  # months, for the whole-portfolio grid
SPREADS = [2, 3, 6, 12, 18, 24, 36, 48, 60]
LUMPS = [12_000, 36_000, 60_000, 120_000, 240_000]
EXTRAS = [500, 1_000, 2_000, 5_000, 10_000, 20_000]
data = Path(__file__).parent / "data"


def load(name):
    return pd.read_csv(data / f"{name}.csv", parse_dates=["Date"], index_col="Date")["Close"]


def month_start_prices(close):
    return close.groupby(close.index.to_period("M")).head(1)


def spread_vs_lump(prices):
    """prices[0] is day one, prices[k] the k-th spread buy. Spread's L-part value / lump's - 1."""
    p = np.asarray(prices, dtype=float)
    return p[0] * np.mean(1 / p) - 1


def plan_totals(m, s, lump, extra, horizon=HORIZON):
    """Whole-portfolio values (at once, spread) at month-start index s + horizon, buying at m[s..s+horizon-1].
    Spread buys min(extra, what is left) each month; any lump not yet invested by the end counts as cash."""
    buy, end = m[s:s + horizon], m[s + horizon]
    base = BUY * np.sum(end / buy)
    at_once = base + lump * end / buy[0]
    k = np.arange(horizon)
    chunks = np.clip(lump - extra * k, 0, extra)
    spread = base + np.sum(chunks * end / buy) + (lump - chunks.sum())
    return at_once, spread


# self-checks
assert spread_vs_lump([100, 100, 100]) == 0
assert np.isclose(spread_vs_lump([100, 50]), 0.5)  # half bought at 100, half at 50: 1.5x the shares
_m = np.array([100.0, 50, 80, 120, 90])
_a, _b = plan_totals(_m, 0, 3_000, 1_500, horizon=4)
# the L part: at once 3000 * 90/100 = 2700; spread 1500*90/100 + 1500*90/50 = 4050; same $BUY part in both
assert np.isclose(_b - _a, 4050 - 2700)
_a, _b = plan_totals(_m, 0, 3_000, 500, horizon=4)  # 6 months needed, only 4 fit: 1000 left as cash
assert np.isclose(_b - _a, 500 * 90 * (1 / 100 + 1 / 50 + 1 / 80 + 1 / 120) + 1_000 - 2700)


# datasets the analysis runs on: key -> (price file, label)
DATASETS = {"price": ("sp500_full", "S&P 500, price only"), "tr": ("sp500_tr", "S&P 500 with dividends")}
PEAKS = [("S&P 1929", "sp500_full", "1929-09-16"), ("S&P 1987", "sp500_full", "1987-08-25"),
         ("S&P 2000", "sp500_full", "2000-03-24"), ("S&P 2007", "sp500_full", "2007-10-09"),
         ("S&P 2020", "sp500_full", "2020-02-19"), ("NASDAQ 2000", "nasdaq_full", "2000-03-10")]
PEAK_LEADS = [0, 3, 6, 9, 12, 18, 24, 36, 48, 60, 84, 120]  # months before the peak the lump arrives


def by_spread(m):
    """For each N in SPREADS, spread-vs-at-once on the lump part over every start month where N fits."""
    rows = []
    for n in SPREADS:
        r = pd.Series([spread_vs_lump(m.iloc[s:s + n]) for s in range(len(m) - n + 1)], index=m.index[:len(m) - n + 1])
        rows.append({"n": n, "starts": len(r), "at_once_wins": (r < 0).mean(), "median": r.median(),
                     "p10": r.quantile(.1), "p90": r.quantile(.9), "best": r.max(), "best_from": r.idxmax(),
                     "worst": r.min(), "worst_from": r.idxmin()})
    return rows


def peak_rows(lead_months=0):
    """Spread-vs-at-once on the lump part when the lump arrives lead_months before a crash peak (first trading
    day on or after that date). start is None when the data does not reach back that far."""
    out = []
    for label, file, day in PEAKS:
        close = load(file)
        target = pd.Timestamp(day) - pd.DateOffset(months=lead_months)
        if target < close.index[0]:
            out.append({"label": label, "day": day, "start": None, "vs": None})
            continue
        d0 = close.index[close.index >= target][0]
        after = month_start_prices(close[d0 + pd.Timedelta(days=1):])
        out.append({"label": label, "day": day, "start": d0,
                    "vs": [spread_vs_lump([close[d0], *after.iloc[:n - 1]]) for n in SPREADS]})
    return out


def grid(m):
    """Whole portfolio after HORIZON months, every start month: spread vs at once for each lump and extra."""
    arr = m.to_numpy()
    starts = range(len(arr) - HORIZON)
    cells = []
    for lump in LUMPS:
        row = []
        for x in EXTRAS:
            t = np.array([plan_totals(arr, s, lump, x) for s in starts])
            d = t[:, 1] / t[:, 0] - 1
            row.append({"lump": lump, "extra": x, "months": int(np.ceil(lump / x)),
                        "spread_wins": (d > 0).mean(), "median": np.median(d)})
        cells.append(row)
    return {"starts": len(starts), "first": m.index[0], "last": m.index[len(arr) - HORIZON - 1], "cells": cells}


def pct(x):
    return f"{x:+.1%}"


if __name__ == "__main__":
    print("== 1. every start month: spread the lump over N months vs invest it at once (on the lump part)")
    for name, label in DATASETS.values():
        m = month_start_prices(load(name))
        print(f"\n  {label}, start months {m.index[0].date()} .. (as far as each N fits) ")
        print(f"  {'N':>3} {'starts':>6} {'at once wins':>13} {'median':>8} {'10th pct':>9} {'90th pct':>9}   best for spreading / worst for spreading")
        for r in by_spread(m):
            print(f"  {r['n']:>3} {r['starts']:>6} {r['at_once_wins']:>13.0%} {pct(r['median']):>8} {pct(r['p10']):>9} {pct(r['p90']):>9}"
                  f"   {pct(r['best'])} from {r['best_from'].date()} / {pct(r['worst'])} from {r['worst_from'].date()}")

    print("\n== 2. starting on a crash peak day (then buying on each following month's first trading day)")
    print(f"  {'':<12}" + "".join(f"{'N=' + str(n):>8}" for n in SPREADS))
    for r in peak_rows():
        print(f"  {r['label']:<12}" + "".join(f"{pct(v):>8}" for v in r["vs"]))
    print("\n  12-month spread vs at once, by how many months before the peak the lump arrived")
    print(f"  {'':<12}" + "".join(f"{str(k) + 'mo':>8}" for k in PEAK_LEADS))
    by_lead = [peak_rows(k) for k in PEAK_LEADS]
    i12 = SPREADS.index(12)
    for j, (label, _, _) in enumerate(PEAKS):
        print(f"  {label:<12}" + "".join(f"{'—':>8}" if rows[j]["vs"] is None else f"{pct(rows[j]['vs'][i12]):>8}" for rows in by_lead))

    print(f"\n== 3. whole portfolio after {HORIZON // 12} years: lump L plus ${BUY:,.0f}/month; spread = extra $X/month until L is used up")
    for name, label in DATASETS.values():
        g = grid(month_start_prices(load(name)))
        print(f"\n  {label}: {g['starts']} start months {g['first'].date()} .. {g['last'].date()}")
        print("  cell = share of starts where spreading wins | median spread vs at once (whole portfolio) | months to use up L")
        print(f"  {'lump':>9}" + "".join(f"{'+$' + format(x, ','):>26}" for x in EXTRAS))
        for lump, row in zip(LUMPS, g["cells"]):
            cells = [f"{c['spread_wins']:>4.0%} | {pct(c['median']):>6} | {c['months']:>3}mo" for c in row]
            print(f"  ${lump:>8,}" + "".join(f"{c:>26}" for c in cells))
