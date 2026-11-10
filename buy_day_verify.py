"""Cross-checks of buy_day.py's finding that the 25th-28th are the cheapest dates to buy, five ways:
  1. other markets, including the S&P 500 total-return index (dividends reinvested)
  2. bootstrap: resample S&P months with replacement, see where the cheapest date lands
  3. split history: 1928-1976 and 1977-2026 on their own
  4. trading-day position (n-th trading day from the start / end of the month) instead of calendar date
  5. a real DCA simulation: $BUY each month on date D (first trading day on or after D), final value
  6. same cash date: with money ready on the 26th, buy then or wait for the 1st; ready on the 1st, buy or wait for the 26th
Same measure as buy_day.py: close vs the centred 21-trading-day average, 10% trimmed mean across months.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from buy_day import MIN_DAYS, TRIM, by_calendar_date, trimmed_mean, vs_centred_window
from dca import BUY, result

data = Path(__file__).parent / "data"
LATE, EARLY = list(range(24, 29)), list(range(3, 9))  # the window buy_day.py found cheapest / dearest
BOOT = 2000


def load(name):
    return pd.read_csv(data / f"{name}.csv", parse_dates=["Date"], index_col="Date")["Close"]


def trim_by_date(df):
    return df.groupby("date")["rel"].apply(trimmed_mean) * 100


def describe(t):
    r = t.sort_values()
    rank26 = int((t < t[26]).sum()) + 1
    return (f"cheapest {', '.join(f'{d}' for d in r.index[:3])} | 26th {t[26]:+.2f}% (rank {rank26}/31) | "
            f"24th-28th {t[LATE].mean():+.2f}% vs 3rd-8th {t[EARLY].mean():+.2f}%")


def dca_by_date(close):
    """Final value of $BUY a month bought on date D, for D = 1..31, as % vs the average over all D."""
    months = [m for _, m in close.groupby(close.index.to_period("M")) if len(m) >= MIN_DAYS]
    vals = {}
    for d in range(1, 32):
        dates = [m.index[min(np.searchsorted(m.index.day, d), len(m) - 1)] for m in months]
        vals[d] = result(close, dates, BUY)[1]
    s = pd.Series(vals)
    return (s / s.mean() - 1) * 100


# self-check: dca_by_date picks the first trading day on or after D, else the month's last day
_c = pd.Series(1.0, index=pd.bdate_range("2001-01-01", "2001-02-28"))
_c["2001-02-26":] = 2.0  # buying on/after Feb 26 costs double, so those dates end with fewer shares
_v = dca_by_date(_c)
# Feb 24-25 2001 is a weekend, so dates 24-26 all land on Monday the 26th; 23 lands on Friday the 23rd
assert _v[23] > _v[24] and np.isclose(_v[24], _v[26]) and np.isclose(_v[26], _v[31])

if __name__ == "__main__":
    sp = load("sp500_full")
    sp_df = by_calendar_date(vs_centred_window(sp))

    print("== 1. other markets")
    for name in ["sp500_full", "sp500_tr", "nasdaq_full", "dow", "russell2000", "ftse100", "dax", "nikkei225"]:
        close = load(name)
        df = by_calendar_date(vs_centred_window(close))
        print(f"  {name:<12} {df['month'].min()}..{df['month'].max()} ({df['month'].nunique():>4} months): {describe(trim_by_date(df))}")

    print(f"\n== 2. bootstrap, {BOOT} resamples of the S&P's {sp_df['month'].nunique()} months")
    wide = sp_df.pivot(index="month", columns="date", values="rel").to_numpy()
    rng = np.random.default_rng(0)
    k = int(len(wide) * TRIM)
    best = []
    for _ in range(BOOT):
        sample = np.sort(wide[rng.integers(0, len(wide), len(wide))], axis=0)
        best.append(int(np.argmin(sample[k:len(sample) - k].mean(axis=0))) + 1)
    best = pd.Series(best)
    print(f"  cheapest date was in the 24th-28th in {best.isin(LATE).mean():.1%} of resamples, "
          f"in the 20th-31st in {best.between(20, 31).mean():.1%}, exactly the 26th in {(best == 26).mean():.1%}")
    print("  most common cheapest dates: " + ", ".join(f"{d} ({n / BOOT:.1%})" for d, n in best.value_counts().head(6).items()))

    print("\n== 3. split history (S&P)")
    for lo, hi in [(1928, 1976), (1977, 2026)]:
        half = sp_df[sp_df["month"].map(lambda p: lo <= p.year <= hi)]
        print(f"  {lo}-{hi} ({half['month'].nunique()} months): {describe(trim_by_date(half))}")

    print("\n== 4. trading-day position (S&P): -1 = last trading day of the month, 1 = first")
    rel = vs_centred_window(sp)
    rows, lands = [], []
    for _, m in rel.groupby(rel.index.to_period("M")):
        if len(m) < MIN_DAYS or m.isna().any():
            continue
        n = len(m)
        rows += [(i + 1, i - n, v) for i, v in enumerate(m.to_numpy())]
        lands.append(int(np.searchsorted(m.index.day, 26)) - n)  # where a buy set for the 26th lands, from the end
    pos = pd.DataFrame(rows, columns=["from_start", "from_end", "rel"])
    fe = pos.groupby("from_end")["rel"].apply(trimmed_mean).loc[-10:] * 100
    fs = pos.groupby("from_start")["rel"].apply(trimmed_mean).loc[:8] * 100
    print("  from the end:   " + "  ".join(f"{p}: {v:+.2f}" for p, v in fe.items()))
    print("  from the start: " + "  ".join(f"{p}: {v:+.2f}" for p, v in fs.items()))
    lands = pd.Series(lands).value_counts(normalize=True).sort_index()
    print(f"  cheapest position: {fe.idxmin()} ({fe.min():+.2f}%); a buy set for the 26th lands at "
          + ", ".join(f"{p} in {share:.0%}" for p, share in lands.items() if share >= 0.05) + " of months")

    print("\n== 5. DCA simulation: final value of $1,000/month bought on date D, vs the average over all dates")
    for name in ["sp500_full", "sp500_tr"]:
        v = dca_by_date(load(name))
        r = v.sort_values(ascending=False)
        print(f"  {name:<11} best {', '.join(f'{d} ({x:+.2f}%)' for d, x in r.head(3).items())} | "
              f"26th {v[26]:+.2f}% (rank {int((v > v[26]).sum()) + 1}/31) | 1st {v[1]:+.2f}% | worst {r.index[-1]} ({r.iloc[-1]:+.2f}%)")

    print("\n== 6. same cash date: is waiting worth it? (+ = waiting costs more, 10% trimmed mean / median, share of months)")
    for name in ["sp500_full", "sp500_tr"]:
        close = load(name)
        months = [m for _, m in close.groupby(close.index.to_period("M")) if len(m) >= MIN_DAYS]
        on = {d: np.array([m.iloc[min(np.searchsorted(m.index.day, d), len(m) - 1)] for m in months]) for d in (1, 26)}
        wait_from_26 = (on[1][1:] / on[26][:-1] - 1) * 100  # ready on the 26th, wait for next month's 1st
        wait_from_1 = (on[26] / on[1] - 1) * 100  # ready on the 1st, wait for the 26th
        for label, x in [("ready on the 26th, wait to the 1st", wait_from_26), ("ready on the 1st, wait to the 26th", wait_from_1)]:
            print(f"  {name:<11} {label}: {trimmed_mean(x):+.2f}% / {np.median(x):+.2f}%, waiting cost more in {(x > 0).mean():.0%} of months")
