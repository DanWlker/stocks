"""#11 Fees: $BUY a month into the S&P 500 with dividends, for 10-40 years, every start month from 1928,
with an annual fee taken monthly from the balance. Median final value and how much less than the cheapest fund.
"""

import numpy as np

from dca import BUY
from monthly import monthly
from report import section as make_section, table, text

FEES = [0.0003, 0.002, 0.005, 0.01, 0.02]  # a year
YEARS = [10, 20, 30, 40]


def final_values(rets, months, fee):
    """Final value of $BUY a month for `months` months, fee charged monthly, for every start month."""
    keep = (1 - fee) ** (1 / 12)
    out = []
    for s in range(len(rets) - months + 1):
        v = 0.0
        for r in rets[s:s + months]:
            v = (v + BUY) * (1 + r) * keep
        out.append(v)
    return np.array(out)


# self-check: no growth, no fee -> just the money put in; a fee is a steady proportional drag
assert np.allclose(final_values(np.zeros(12), 12, 0), BUY * 12)
assert np.isclose(final_values(np.zeros(1), 1, 0.12)[0], BUY * (1 - 0.12) ** (1 / 12))

def section():
    df = monthly()
    rets = df["stock_ret"].to_numpy()
    rows, lost = [], {}
    for y in YEARS:
        base = final_values(rets, 12 * y, FEES[0])
        cells = []
        for f in FEES:
            v = final_values(rets, 12 * y, f)
            rel = np.median(v / base - 1)
            lost[y, f] = (rel, np.median(base) - np.median(v))
            cells.append([f"${np.median(v):,.0f}", "cheapest fund" if f == FEES[0] else f"{rel:+.1%}"])
        rows.append([[f"{y} years", f"${12 * y * BUY:,.0f} put in"]] + cells)

    one = lost[30, 0.01]
    lede = (f"A fee is taken from the whole balance every year, so its cost compounds. Investing ${BUY:,.0f} a month for 30 years, "
            f"a 1% fee left a median {-one[0]:.1%} less than a 0.03% index fund, about ${one[1]:,.0f}; a 2% fee left "
            f"{-lost[30, 0.02][0]:.1%} less, and over 40 years {-lost[40, 0.02][0]:.1%} less.")
    return make_section(
        "fees", "Fees", "What a fee costs over a lifetime of DCA", lede,
        [table(f"${BUY:,.0f} a month into the S&P 500 with dividends: median final value, and how much less than the cheapest fund",
               ["Investing for"] + [f"{f:.2%} a year" for f in FEES], rows,
               note=f"Every start month from {df.index[0]} that the period fits in.")],
        note="The fee is charged monthly on the balance. The dollar difference is between the two medians. Computed in fees.py.")


if __name__ == "__main__":
    print(text(section()))
