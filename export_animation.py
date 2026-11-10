"""Compute per-day DCA simulation data and render the page into site/ from animation_template.html.

Each crash is replayed from several start dates (0..10 years before its peak, all ending on the same
date). Each replay's daily data goes to site/anim/<id>.json, loaded by the page on demand; the page
itself (site/index.html) carries only summaries. All numbers shown come from here; the page only draws them.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from dca import BUY, PEAK_YEAR, STEP, daily, ladder, peak, result
from bear_pause import DROP, PAUSE_MONTHS, bear_pause
from month_dip import month_dip
from pause_check import LENGTHS, STARTS, absorbed, pause_rows, vs_dca
from date_rank import WINDOWS, rank_dates
import lump_vs_spread as lvs
from pause_check import load as load_full

root = Path(__file__).parent

# key -> (label, threshold, rule returning (all buys, buys fired by a price trigger))
RULES = {
    "dca": ("Normal DCA, 1st trading day of the month", None,
            lambda c: (list(c.groupby(c.index.to_period("M")).head(1).index), [])),
    "ladder": ("Every 10% below my last buy", STEP, lambda c: (b := ladder(c), b[1:])),
    "peak": ("Every 10% step below the high", STEP, lambda c: (b := peak(c), b)),
    "daily": ("Any single day down 10%+", STEP, lambda c: (b := daily(c), b)),
    "month10": ("Monthly, early on a 10% dip", 0.10, lambda c: month_dip(c, 0.10)),
    "month4": ("Monthly, early on a 4% dip", 0.04, lambda c: month_dip(c, 0.04)),
    # its trigger fires the pause, not a buy
    "bear31": ("Monthly, paused 31 months after a 20% fall", DROP,
               lambda c: (lambda b, p: (b, [start for _, start, _, _ in p]))(*bear_pause(c))),
}
# crash -> (price file, last day of every replay, era, index); the peak is the highest close in PEAK_YEAR
CRASHES = {
    "depression_sp500": ("sp500_full", "1939-12-29", "Great Depression", "S&P (^GSPC)"),
    "dotcom_sp500": ("sp500_full", "2007-12-31", "Dot-com", "S&P 500 (^GSPC)"),
    "dotcom_nasdaq": ("nasdaq_full", "2007-12-31", "Dot-com", "NASDAQ (^IXIC)"),
}
OFFSETS = range(0, 11)  # start this many years before the peak


def trigger(key, step, close, buys):
    """Close at or below which the rule buys on the next day, as known after each day's close."""
    if key == "dca":
        return pd.Series(np.nan, index=close.index)  # no trigger: it buys every month
    if key == "ladder":
        return close.where(close.index.isin(buys)).ffill() * (1 - STEP)
    if key.startswith("month"):
        months = close.groupby(close.index.to_period("M"))
        bought = pd.Series(close.index.isin(buys).astype(int), index=close.index)
        done = bought.groupby(close.index.to_period("M")).cummax() == 1
        last_day = months.cumcount(ascending=False) == 0
        return (months.transform("first") * (1 - step)).where(~done & ~last_day)
    if key == "bear31":
        high = close.cummax()
        dates = close.index.to_series()
        peak_date = dates.where(close > high.shift(1)).ffill().fillna(close.index[0])
        nxt = dates.shift(-1)
        ok = pd.Series(True, index=close.index)
        for peak_d, start, resume, _ in bear_pause(close)[1]:
            ok &= ~((nxt > start) & ((nxt < resume) if resume is not None else True))  # paused when tomorrow's check runs
            ok &= ~((peak_date == peak_d) & (dates >= start))  # this high already caused a pause
        return (high * (1 - DROP)).where(ok)
    if key == "peak":
        high = close.cummax()
        new_high = close > high.shift(1)
        reached = np.floor(((1 - close / high) / STEP).round(9))
        tier = reached.groupby(new_high.cumsum()).cummax()
        return high * (1 - STEP * (tier + 1))
    return close * (1 - STEP)


def holdings(close, dates, amount):
    # a date listed k times buys k * amount that day
    bought = pd.Series(dates).value_counts().reindex(close.index, fill_value=0) * amount
    shares = (bought / close).cumsum()
    return bought.cumsum(), shares * close, shares


def r2(s, digits=2):
    return [None if math.isnan(v) else round(v, digits) for v in s]


def simulate(close):
    """Every rule's daily series, buys and summary for one replay window."""
    month_starts = list(close.groupby(close.index.to_period("M")).head(1).index)
    high = close.cummax()
    rules = {}
    month_ref = close.groupby(close.index.to_period("M")).transform("first")
    for key, (rule_label, step, rule) in RULES.items():
        buys, triggered = rule(close)
        if not buys:
            continue
        trig = trigger(key, step, close, buys)
        # cross-check: the exported trigger line must reproduce the rule's triggered buy dates exactly
        fired = list(close.index[close <= trig.shift(1)])
        assert fired == triggered, (close.index[0], key)

        total = BUY * len(buys)
        series, summary = {}, []
        for label, ds, amt in [
            ("rule", buys, BUY),
            ("lump", [close.index[0]], total),
            ("monthly", month_starts, total / len(month_starts)),
        ]:
            inv, val, shares = holdings(close, ds, amt)
            f_inv, f_val, f_avg, f_irr = result(close, ds, amt)
            assert math.isclose(val.iloc[-1], f_val) and math.isclose(inv.iloc[-1], f_inv)
            series[label] = {"inv": r2(inv), "val": r2(val)}
            if label in ("rule", "lump"):
                # return on the money put in so far: comparable across rules that invest different totals
                series[label]["ret"] = r2(val / inv.where(inv > 0) - 1, 4)
            summary.append({"key": label, "invested": round(f_inv, 2),
                            "value": round(f_val, 2), "ret": f_val / f_inv - 1, "xirr": f_irr,
                            "avg": round(f_avg, 2)})
            if label == "rule":
                rule_inv, rule_shares = inv, shares

        monthly_value = next(x["value"] for x in summary if x["key"] == "monthly")
        summary[0]["edge"] = summary[0]["value"] / monthly_value - 1  # vs the same total spread monthly
        prev_trig = trig.shift(1)
        day_chg = close.pct_change()
        events = []
        for n, d in enumerate(dict.fromkeys(buys), 1):
            i = close.index.get_loc(d)
            events.append({
                "i": i, "n": n, "units": buys.count(d), "price": round(close[d], 2), "dip": d in triggered,
                "ref": round(month_ref[d], 2), "refchg": close[d] / month_ref[d] - 1,
                "trig": None if math.isnan(prev_trig[d]) else round(prev_trig[d], 2),
                "chg": None if math.isnan(day_chg[d]) else day_chg[d],
                "high": round(high[d], 2), "dd": close[d] / high[d] - 1,
                "avg": round(rule_inv[d] / rule_shares[d], 2),
            })
        pauses = [{"peak_i": close.index.get_loc(pk), "peak_price": round(close[pk], 2),
                   "start_i": close.index.get_loc(st), "skipped": sk,
                   "resume_i": None if rs is None else close.index.get_loc(rs)}
                  for pk, st, rs, sk in (bear_pause(close)[1] if key == "bear31" else [])]
        rules[key] = {"label": rule_label, "step": step, "trigger": r2(trig), "events": events, "pauses": pauses,
                      "series": series, "summary": summary}
    return rules


def starts(close, peak_day):
    """(years before the peak, first trading day) for each offset the data reaches, plus the first day
    of data when it falls short of the next whole year (the Depression data starts 1.7 years before 1929)."""
    out = []
    for n in OFFSETS:
        target = peak_day - pd.DateOffset(years=n)
        if target < close.index[0]:
            out.append((round((peak_day - close.index[0]).days / 365.25, 1), close.index[0]))
            break
        out.append((n, close.index[close.index >= target][0]))
    return out


def day(d):
    return None if d is None else str(d.date())


site = root / "site"
(site / "anim").mkdir(parents=True, exist_ok=True)
scenarios, start_compare = [], {}
for name, (file, end, era, index_label) in CRASHES.items():
    prices = pd.read_csv(root / "data" / f"{file}.csv", parse_dates=["Date"], index_col="Date")["Close"][:end]
    peak_day = prices[PEAK_YEAR[name]].idxmax()
    cmp = {"era": era, "index": index_label, "peak": day(peak_day), "end": end, "years": [], "starts": [],
           "rules": {k: {"label": v[0], **{m: [] for m in ("edge", "ret", "xirr", "value", "invested", "buy_days")}}
                     for k, v in RULES.items()}}
    for years, start in starts(prices, peak_day):
        close = prices[start:]
        rules = simulate(close)
        sid = f"{name}-{years}y"
        meta = {"id": sid, "crash": name, "era": era, "index": index_label, "years": years,
                "start": day(start), "peak": day(peak_day), "end": day(close.index[-1])}
        anim = {**meta, "dates": [d.strftime("%Y-%m-%d") for d in close.index], "close": r2(close), "rules": rules}
        (site / "anim" / f"{sid}.json").write_text(json.dumps(anim, separators=(",", ":")))
        scenarios.append(meta)
        cmp["years"].append(years)
        cmp["starts"].append(day(start))
        for k, c in cmp["rules"].items():
            x = rules[k]["summary"][0] if k in rules else None
            for m in ("edge", "ret", "xirr", "value", "invested"):
                c[m].append(None if x is None else x[m])
            c["buy_days"].append(None if x is None else len(rules[k]["events"]))
        print(f"{sid}: {day(start)} -> {day(close.index[-1])}, {len(close)} days, rules {list(rules)}")
    start_compare[name] = cmp

def num(v, digits=2):
    return None if v is None else round(v, digits)


def pause_json(rows):
    return [{"peak": day(r["peak"]), "peak_price": num(r["peak_price"]), "start": day(r["start"]), "resume": day(r["resume"]),
             "resume_price": num(r["resume_price"]), "skipped": r["skipped"], "dca_cost": num(r["dca_cost"]), "vs": r["vs"],
             "peak_to_drop_mo": num(r["peak_to_drop_mo"], 1), "drop_to_resume_mo": num(r["drop_to_resume_mo"], 1),
             "next_peak": day(r.get("next_peak")) if r["next_peak_before_resume"] else None,
             "peak_in_prev_pause": r["peak_in_prev_pause"], "days_after_prev_resume": r["days_after_prev_resume"],
             "pre_drop_months": r["pre_drop_months"], "pre_drop_in_prev": r["pre_drop_in_prev"]}
            for r in rows]


full = load_full()
by_length = {}
for m in LENGTHS:
    rows = pause_rows(full, m)
    by_length[m] = {"pauses": pause_json(rows),
                    "absorbed": [{"peak": day(a["peak"]), "inside_peak": day(a["inside_peak"])} for a in absorbed(full, rows)]}
pause_check = {
    "months": PAUSE_MONTHS, "first": str(full.index[0].date()), "last": str(full.index[-1].date()),
    "byLength": by_length, "starts": STARTS,
    "lengths": [{"months": m, "vs": [vs_dca(full[s:], m) for s in STARTS]} for m in LENGTHS],
}
print(f"pause check: {len(LENGTHS)} lengths x {len(STARTS)} starts")

date_rank = {}
for years in WINDOWS:
    ranked, rank_meta = rank_dates(full, years)
    date_rank[years] = {**rank_meta, "years": years,
                        "rows": [{"date": int(d), "rank": int(r["rank"]), "value": round(r["value"], 2),
                                  "invested": round(r["invested"], 2), "avg_cost": round(r["avg_cost"], 2),
                                  "vs_best": r["vs_best_%"], "vs_month": r["vs_month_%"],
                                  "vs_month_rank": int(r["vs_month_rank"])} for d, r in ranked.iterrows()]}
print(f"date rank: 31 dates over the last {', '.join(map(str, WINDOWS))} years")

lump_spread = {"horizon": lvs.HORIZON, "spreads": lvs.SPREADS, "lumps": lvs.LUMPS, "extras": lvs.EXTRAS,
               "peakLeads": lvs.PEAK_LEADS,
               "peaks": {k: [{**r, "start": day(r["start"])} for r in lvs.peak_rows(k)] for k in lvs.PEAK_LEADS},
               "datasets": {}}
for key, (file, label) in lvs.DATASETS.items():
    m = lvs.month_start_prices(lvs.load(file))
    g = lvs.grid(m)
    lump_spread["datasets"][key] = {
        "label": label, "first": day(m.index[0]),
        "rows": [{**r, "best_from": day(r["best_from"]), "worst_from": day(r["worst_from"])} for r in lvs.by_spread(m)],
        "grid": {**g, "first": day(g["first"]), "last": day(g["last"])}}
print(f"lump vs spread: {len(lvs.SPREADS)} spreads, {len(lvs.LUMPS)}x{len(lvs.EXTRAS)} grid, {len(lvs.PEAKS)} peaks")

payload = json.dumps({"buy": BUY, "scenarios": scenarios, "startCompare": start_compare,
                      "pauseCheck": pause_check, "dateRank": date_rank, "lumpSpread": lump_spread}, separators=(",", ":"))
html = (root / "animation_template.html").read_text().replace("/*DATA*/null", payload)
(site / "index.html").write_text(html)
sizes = [f.stat().st_size for f in (site / "anim").glob("*.json")]
print(f"wrote site/index.html ({len(html) / 1e6:.1f} MB) and {len(sizes)} replay files "
      f"({sum(sizes) / 1e6:.1f} MB, largest {max(sizes) / 1e6:.1f} MB)")
