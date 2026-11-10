"""Shared shape for an analysis section: each analysis script prints it, export_animation.py puts it on the page.

section: {"id", "eyebrow", "title", "lede", "tables": [table], "note"}
table:   {"title", "head": [str], "rows": [[cell]], "highlight": [row index], "note"}
cell:    str, or [main, sub] where sub is a second, smaller line
"""

import textwrap


def section(id, eyebrow, title, lede, tables, note=""):
    return {"id": id, "eyebrow": eyebrow, "title": title, "lede": lede, "tables": tables, "note": note}


def table(title, head, rows, highlight=(), note=""):
    assert all(len(r) == len(head) for r in rows), title
    return {"title": title, "head": head, "rows": rows, "highlight": list(highlight), "note": note}


def text(sec):
    flat = lambda c: f"{c[0]} ({c[1]})" if isinstance(c, list) else c
    out = [f"== {sec['title']}", textwrap.fill(sec["lede"], 120), ""]
    for t in sec["tables"]:
        rows = [[flat(c) for c in r] for r in t["rows"]]
        widths = [max(len(str(x)) for x in col) for col in zip(t["head"], *rows)]
        line = lambda cells: "  " + "  ".join(str(c).ljust(w) if i == 0 else str(c).rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))
        out += [f"-- {t['title']}", line(t["head"])] + [line(r) for r in rows]
        if t["note"]:
            out.append(textwrap.fill(t["note"], 120, initial_indent="  ", subsequent_indent="  "))
        out.append("")
    if sec["note"]:
        out.append(textwrap.fill(sec["note"], 120))
    return "\n".join(out)
