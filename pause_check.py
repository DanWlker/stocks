"""Bear-pause rule on the full S&P 500 history (from 1928), against plain monthly DCA with the same
$BUY on each month's first trading day, for several pause lengths and start dates.

Per pause: DCA would have bought the skipped months at their first-day closes, costing on average
the harmonic mean of those closes per share; the pause rule bought them all at the catch-up close.
A negative "catch-up vs DCA" means the pause got the skipped months cheaper.
Pauses can overlap: a new all-time high can come while still paused for the previous crash, and the
next pause can then start right after that catch-up.
"""

from pathlib import Path

import pandas as pd

from bear_pause import PAUSE_MONTHS, bear_pause
from crashes import from_ath
from dca import BUY, result

MONTH = 365.25 / 12

LENGTHS = [3, 6, 9, 12, 15, 18, 24, 31, 36, 48]
# whole history; after the 1929 high was regained (1954-09-22); recent decades
STARTS = ["1927-12-30", "1954-10-01", "1990-01-01"]


def month_starts(close):
    return list(close.groupby(close.index.to_period("M")).head(1).index)


def pause_rows(close, pause_months):
    starts = month_starts(close)
    rows = []
    for peak, start, resume, skipped in bear_pause(close, pause_months)[1]:
        skipped_starts = [d for d in starts if start <= d and (resume is None or d <= resume)][:skipped]
        # no skipped months when the 20% drop came after the pause had already run out: the pause did nothing
        dca_cost = len(skipped_starts) / sum(1 / close[d] for d in skipped_starts) if skipped_starts else None
        rows.append({"peak": peak, "peak_price": close[peak], "start": start, "resume": resume,
                     "resume_price": None if resume is None else close[resume], "skipped": skipped,
                     "dca_cost": dca_cost,
                     "vs": None if resume is None or dca_cost is None else close[resume] / dca_cost - 1,
                     "peak_to_drop_mo": (start - peak).days / MONTH,
                     "drop_to_resume_mo": None if resume is None else (resume - start).days / MONTH})
    for prev, cur in zip(rows, rows[1:]):
        # the next high came before this catch-up: the catch-up landed inside the next crash's run-up or fall
        prev["next_peak_before_resume"] = prev["resume"] is not None and cur["peak"] < prev["resume"]
        prev["next_peak"] = cur["peak"]
        # this crash's pre-drop months were skipped by the previous pause instead of bought
        cur["peak_in_prev_pause"] = prev["next_peak_before_resume"]
        cur["days_after_prev_resume"] = (cur["start"] - prev["resume"]).days if cur["peak_in_prev_pause"] else None
        if cur["peak_in_prev_pause"]:
            cur["pre_drop_months"] = sum(cur["peak"] <= d < cur["start"] for d in starts)
            cur["pre_drop_in_prev"] = sum(cur["peak"] <= d < prev["resume"] for d in starts)
    for r in rows:
        r.setdefault("next_peak_before_resume", False)
        r.setdefault("peak_in_prev_pause", False)
        r.setdefault("days_after_prev_resume", None)
        r.setdefault("pre_drop_months", None)
        r.setdefault("pre_drop_in_prev", None)
    return rows


def absorbed(close, rows):
    """Crashes (20%+ falls from an all-time high) that got no pause of their own, each with the pause it fell inside."""
    out = []
    for peak, _, _ in from_ath(close):
        # the pause rule and from_ath can pick neighbouring days when two closes tie at the high
        if any(abs((r["peak"] - peak).days) <= 7 for r in rows):
            continue
        inside = [r for r in rows if r["start"] <= peak and (r["resume"] is None or peak < r["resume"])]
        out.append({"peak": peak, "inside_peak": inside[0]["peak"] if inside else None})
    return out


def vs_dca(close, pause_months):
    """Final value of the pause rule (plus cash still waiting, at 0%) relative to plain monthly DCA."""
    starts = month_starts(close)
    _, dca_val, _, _ = result(close, starts, BUY)
    buys = bear_pause(close, pause_months)[0]
    return (result(close, buys, BUY)[1] + (len(starts) - len(buys)) * BUY) / dca_val - 1


def load():
    return pd.read_csv(Path(__file__).parent / "data" / "sp500_full.csv", parse_dates=["Date"], index_col="Date")["Close"]


if __name__ == "__main__":
    close = load()
    print(f"S&P 500 {close.index[0].date()} -> {close.index[-1].date()}")
    print(f"\n== {PAUSE_MONTHS}-month pause, every pause:")
    for r in pause_rows(close, PAUSE_MONTHS):
        end = f"resume {r['resume'].date()} at {r['resume_price']:>8.2f}" if r["resume"] is not None else "still paused at the end    "
        vs = f"{r['vs']:+7.1%}" if r["vs"] is not None else "    n/a"
        print(f"  peak {r['peak'].date()} {r['peak_price']:>8.2f} | paused {r['start'].date()} | {end} | "
              f"{r['skipped']:>2} months skipped, DCA would have paid {r['dca_cost']:>8.2f} | catch-up vs DCA {vs}"
              + (" | next high came before this catch-up" if r["next_peak_before_resume"] else "")
              + (f" | peak inside previous pause, paused {r['days_after_prev_resume']} days after its catch-up" if r["peak_in_prev_pause"] else ""))
    print("\n== overlapping pauses by pause length")
    for m in LENGTHS:
        rows = pause_rows(close, m)
        pairs = [f"{a['peak'].year}->{b['peak'].year}" for a, b in zip(rows, rows[1:]) if a["next_peak_before_resume"]]
        idle = sum(r["skipped"] == 0 for r in rows)
        gone = ", ".join(f"{a['peak'].date()} inside the {a['inside_peak'].year} pause" for a in absorbed(close, rows))
        print(f"  {m:>2} months: {len(rows)} pauses ({idle} skipped nothing), overlaps: {', '.join(pairs) or 'none'}"
              + (f", no pause of its own: {gone}" if gone else ""))
    print("\n== final value vs plain monthly DCA, by pause length and start date")
    print(f"  {'months':>6}" + "".join(f"{'from ' + s[:4]:>12}" for s in STARTS))
    for m in LENGTHS:
        print(f"  {m:>6}" + "".join(f"{vs_dca(close[s:], m):>+12.1%}" for s in STARTS))
