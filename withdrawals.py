"""#12 Withdrawals: start with $1,000,000, take RATE x that in year one, raise it with inflation every month, for 30 years.
Every start month 1928-01 .. 1996-11. Portfolios rebalanced monthly; "4% fixed" is the user's strategy that returns
4% a year (taken as nominal and guaranteed). Withdrawals come out at the start of each month.
"""

import numpy as np
import pandas as pd

from monthly import monthly
from report import section as make_section, table, text

START_VALUE = 1_000_000
HORIZON = 360
RATES = [0.03, 0.035, 0.04, 0.045, 0.05, 0.06]
FIXED = 1.04 ** (1 / 12) - 1


def run(rets, cpi, rate, horizon=HORIZON):
    """(months lasted (horizon if it never ran out), final real value in start-month dollars)."""
    v, base = START_VALUE, rate * START_VALUE / 12
    for t in range(horizon):
        v -= base * cpi[t] / cpi[0]
        if v < -1e-6:  # this withdrawal could not be paid in full
            return t, 0.0
        v *= 1 + rets[t]
    return horizon, v * cpi[0] / cpi[horizon - 1]


# self-check: no growth and no inflation, 4% a year lasts exactly 25 years
assert run(np.zeros(360), np.ones(360), 0.04)[0] == 300

def section():
    df = monthly()
    s, b, cpi = df["stock_ret"].to_numpy(), df["bond_ret"].to_numpy(), df["cpi"].to_numpy()
    portfolios = {"100% stocks": s, "60% stocks / 40% bonds": 0.6 * s + 0.4 * b,
                  "100% your 4% fixed": np.full(len(s), FIXED), "60% stocks / 40% your 4%": 0.6 * s + 0.4 * FIXED}
    starts = range(len(df) - HORIZON + 1)
    rows_s, results, success = [], {}, {}
    for name, rets in portfolios.items():
        row = []
        for r in RATES:
            out = [run(rets[i:i + HORIZON], cpi[i:i + HORIZON], r) for i in starts]
            results[name, r] = out
            success[name, r] = np.mean([m == HORIZON for m, _ in out])
            row.append(f"{success[name, r]:.0%}")
        rows_s.append([name] + row)

    rows_w = []
    for name in portfolios:
        out = results[name, 0.04]
        lasted = pd.Series([m for m, _ in out], index=df.index[:len(out)])
        rows_w.append([name, f"{lasted.idxmin()}", f"{lasted.min() / 12:.1f} yrs", f"${np.median([v for _, v in out]):,.0f}"])

    out = results["100% your 4% fixed", 0.04]
    ok = pd.Series([m == HORIZON for m, _ in out], index=df.index[:len(out)])
    rows_d = [[f"{dec}s", f"{ok[f'{max(dec, 1928)}-01':f'{dec + 9}-12'].mean():.0%}"] for dec in range(1920, 2000, 10)]

    lede = (f"Start with $1,000,000, withdraw 4% of it in year one and raise that with inflation, for 30 years. Across "
            f"{len(starts)} start months since {df.index[0]}, the money lasted in {success['100% stocks', 0.04]:.0%} of them with all "
            f"stocks, {success['60% stocks / 40% bonds', 0.04]:.0%} with 60/40 stocks and bonds, and "
            f"{success['60% stocks / 40% your 4%', 0.04]:.0%} with 60% stocks and 40% in your 4% strategy. All in your 4% it "
            f"lasted only {success['100% your 4% fixed', 0.04]:.0%} of the time: withdrawals that rise with inflation outgrow a fixed 4%.")
    return make_section(
        "withdrawals", "Living off the money", "Does the 4% withdrawal rule hold up?", lede,
        [table("Share of start months where the money lasted 30 years, by first-year withdrawal rate",
               ["Portfolio"] + [f"{r:.1%}" for r in RATES], rows_s),
         table("At a 4% withdrawal: the worst start month, and the median left after 30 years",
               ["Portfolio", "Worst start", "Money lasted", "Median left (start-month dollars)"], rows_w),
         table("All in your 4% at a 4% withdrawal: share of starts that lasted, by start decade", ["Started in", "Lasted 30 years"], rows_d,
               note="The 1920s row covers only 1928-29 starts.")],
        note="Withdrawals come out at the start of each month; portfolios are rebalanced monthly; your 4% is taken as nominal and "
             "guaranteed. Bonds are the modelled 10-year Treasury. Taxes and fees ignored. Computed in withdrawals.py.")


if __name__ == "__main__":
    print(text(section()))
