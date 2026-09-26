"""Independent two-way fixed-effects fit (dense normal equations, numpy only) for cross-checking web/model.js.

Usage: check_fe.py <tokId> <first|mean>  -> JSON list of typical tokens per language index.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    tok_id, agg = sys.argv[1], sys.argv[2]
    d = json.loads((ROOT / "build" / "data.json").read_text())
    p = d["pairs"]
    task = np.array(p["task"])
    lang = np.array(p["lang"])
    if tok_id == "chars":
        src = {"first": p["chars_first"], "sum": p["chars_sum"]}
    else:
        src = p["tok"][tok_id]
    vals = np.array(src["first"], float) if agg == "first" else np.array(src["sum"], float) / np.array(p["nvar"])
    y = np.log(vals)
    T, L = len(d["tasks"]), len(d["langs"])

    col_l = T + lang
    K = T + L
    xtx = np.zeros((K, K))
    np.add.at(xtx, (task, task), 1)
    np.add.at(xtx, (col_l, col_l), 1)
    np.add.at(xtx, (task, col_l), 1)
    np.add.at(xtx, (col_l, task), 1)
    xty = np.zeros(K)
    np.add.at(xty, task, y)
    np.add.at(xty, col_l, y)
    beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    a, b = beta[:T], beta[T:]
    print(json.dumps(np.exp(b + a.mean()).tolist()))


if __name__ == "__main__":
    main()
