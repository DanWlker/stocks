"""Bear-pause rule: monthly DCA of BUY on each month's first trading day, except that when a close
is 20% below the all-time high, buying pauses until the first trading day 31 months after that high.
That day buys every skipped month at once, then monthly DCA continues. Only a new all-time high
can start the next pause. Compared with plain monthly DCA and a lump sum with the same money.
"""

from pathlib import Path

import pandas as pd

from dca import BUY, PEAK_YEAR, result

DROP = 0.20
PAUSE_MONTHS = 31


def bear_pause(close, pause_months=PAUSE_MONTHS):
    """Returns (buys, pauses). A date appears in buys once per BUY unit bought that day, so a
    catch-up buy of 30 months is the date 30 times. Each pause is (peak date, pause start,
    resume date or None if still paused at the end, skipped months)."""
    month_starts = set(close.groupby(close.index.to_period("M")).head(1).index)
    buys, pauses, pause = [], [], None
    high, peak, used_peak = close.iloc[0], close.index[0], None
    for date, price in close.items():
        if price > high:
            high, peak = price, date
        if pause and date >= pause["until"]:
            buys += [date] * pause["skipped"]
            pauses.append((pause["peak"], pause["start"], date, pause["skipped"]))
            pause = None
        if not pause and peak != used_peak and price <= high * (1 - DROP):
            used_peak = peak
            pause = {"peak": peak, "start": date, "until": peak + pd.DateOffset(months=pause_months), "skipped": 0}
        if date in month_starts:
            if pause:
                pause["skipped"] += 1
            else:
                buys.append(date)
    if pause:
        pauses.append((pause["peak"], pause["start"], None, pause["skipped"]))
    return buys, pauses


# self-check on one close per month: peak 110 in month 1, -22.7% in month 2 pauses buying,
# month 32 (31 months after the peak) buys the 30 skipped months plus its own; 80 never makes a new high
_m = pd.date_range("2001-01-01", periods=40, freq="MS")
_b, _p = bear_pause(pd.Series([100, 110, 85] + [80.0] * 37, index=_m))
assert _p == [(_m[1], _m[2], _m[32], 30)], _p
assert _b.count(_m[32]) == 31 and len(_b) == 40

if __name__ == "__main__":
    data = Path(__file__).parent / "data"
    for name in ["depression_sp500", "dotcom_sp500", "dotcom_nasdaq"]:
        full = pd.read_csv(data / f"{name}.csv", parse_dates=["Date"], index_col="Date")["Close"]
        for window, close in [("full range", full), ("from peak", full[full[PEAK_YEAR[name]].idxmax():])]:
            months = close.groupby(close.index.to_period("M"))
            buys, pauses = bear_pause(close)
            print(f"\n== {name} [{window}]  {close.index[0].date()} -> {close.index[-1].date()}  {months.ngroups} months")
            for peak, start, resume, skipped in pauses:
                end = f"resumed {resume.date()} at {close[resume]:.2f}, catch-up ${skipped * BUY:,.0f}" if resume \
                    else f"STILL PAUSED at the end, ${skipped * BUY:,.0f} never invested"
                print(f"    peak {peak.date()} ({close[peak]:.2f}) -> paused {start.date()} ({close[start]:.2f}) -> {end}")
            for label, ds, amt in [
                ("bear rule", buys, BUY),
                ("monthly", list(months.head(1).index), BUY),
                ("lump sum", [close.index[0]], BUY * months.ngroups),
            ]:
                inv, val, avg, irr = result(close, ds, amt)
                print(f"    {label:<9} invested ${inv:>9,.0f}  final ${val:>10,.0f}  "
                      f"return {val / inv - 1:+8.1%}  XIRR {irr:+7.2%}/yr  avg cost {avg:9.2f}")
