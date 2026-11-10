"""How long does a crash (a fall of 20% or more) last? S&P 500 daily closes 1928-2026, price only.

Two common definitions:
  from all-time high - starts at an all-time high, bottoms at the lowest close before that high is
                       regained, ends when the close is back at the old high. Only a new all-time
                       high can start the next one (the rule the bear-pause strategy uses).
  bear market        - the classic 20% rule: starts at any peak the market then falls 20% from, bottoms
                       at the low it then rises 20% from. 1937 and 1946 count separately even though
                       the 1929 high was not regained until 1954.
Durations are calendar months (days / 30.44). Recovery = first close back at the peak's level.
"""

from pathlib import Path

import pandas as pd

from buy_day import trimmed_mean

DROP = 0.20
MONTH = 365.25 / 12


def from_ath(close):
    """List of (peak, trough, recovery date or None)."""
    out = []
    for _, s in close.groupby((close >= close.cummax()).cumsum()):  # each group: one high until the next close at/above it
        if s.min() <= s.iloc[0] * (1 - DROP):
            after = close.index.get_loc(s.index[-1]) + 1
            out.append((s.index[0], s.idxmin(), close.index[after] if after < len(close) else None))
    return out


def bear_markets(close):
    """List of (peak, trough, recovery date or None, ended) where ended is False for a bear still running."""
    out, peak, trough, in_bear = [], close.index[0], None, False
    for date, price in close.items():
        if not in_bear:
            if price > close[peak]:
                peak = date
            elif price <= close[peak] * (1 - DROP):
                in_bear, trough = True, date
        elif price < close[trough]:
            trough = date
        elif price >= close[trough] * (1 + DROP):
            out.append((peak, trough, True))
            in_bear, peak = False, date
    if in_bear:
        out.append((peak, trough, False))
    result = []
    for peak, trough, ended in out:
        back = close[trough:][close[trough:] >= close[peak]]
        result.append((peak, trough, back.index[0] if len(back) else None, ended))
    return result


# self-checks: a 1937-style second fall inside one unrecovered high
_d = pd.date_range("2001-01-01", periods=5)
_t = pd.Series([100, 50, 70, 40, 100.0], index=_d)
assert from_ath(_t) == [(_d[0], _d[3], _d[4])]
assert bear_markets(_t) == [(_d[0], _d[1], _d[4], True), (_d[2], _d[3], _d[4], True)]
_t = pd.Series([100, 110, 80, 90, 120.0], index=_d)
assert from_ath(_t) == [(_d[1], _d[2], _d[4])] and bear_markets(_t) == [(_d[1], _d[2], _d[4], True)]


def months(a, b):
    return (b - a).days / MONTH if b is not None else float("nan")


def report(title, rows):
    df = pd.DataFrame(rows, columns=["peak", "trough", "recovered", "ended"])
    df["depth_%"] = [(close[t] / close[p] - 1) * 100 for p, t in zip(df.peak, df.trough)]
    df["peak_to_bottom_mo"] = [months(p, t) for p, t in zip(df.peak, df.trough)]
    df["bottom_to_recovery_mo"] = [months(t, r) for t, r in zip(df.trough, df.recovered)]
    df["peak_to_recovery_mo"] = [months(p, r) for p, r in zip(df.peak, df.recovered)]
    print(f"\n== {title}: {len(df)} crashes")
    show = df.copy()
    for c in ["peak", "trough", "recovered"]:
        show[c] = show[c].map(lambda d: d.date() if d is not None else "not yet")
    print(show.drop(columns="ended").to_string(index=False, float_format="{:.1f}".format))
    if not df.ended.all():
        print("  last one is still running: its bottom may not be final and it is left out of the averages")
    done = df[df.ended]
    print(f"  {'':<22}{'n':>4}{'mean':>8}{'median':>8}{'trim10%':>9}{'min':>7}{'max':>7}")
    for col in ["depth_%", "peak_to_bottom_mo", "bottom_to_recovery_mo", "peak_to_recovery_mo"]:
        v = done[col].dropna()
        print(f"  {col:<22}{len(v):>4}{v.mean():>8.1f}{v.median():>8.1f}{trimmed_mean(v):>9.1f}{v.min():>7.1f}{v.max():>7.1f}")


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    close = pd.read_csv(Path(__file__).parent / "data" / "sp500_full.csv", parse_dates=["Date"], index_col="Date")["Close"]
    print(f"S&P 500 {close.index[0].date()} -> {close.index[-1].date()}")
    report("from all-time high", [(p, t, r, True) for p, t, r in from_ath(close)])
    report("bear market (classic 20% rule)", bear_markets(close))
