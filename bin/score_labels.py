#!/usr/bin/env python3
"""Score any set of label tables against the 58 gold studies, like for like.

Published agreement figures are not comparable unless the comparison is: the
same 58 studies, the same binarisation, the same twelve columns. This prints
that table for our reader and for every public CSV you point it at.

    python3 bin/score_labels.py --data DIR [--tables pub/**/*.csv ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import reports as R, rules  # noqa: E402
from knee.constants import TARGETS  # noqa: E402


def load_table(path: Path) -> pd.DataFrame | None:
    frame = pd.read_csv(path)
    if "StudyInstanceUID" not in frame.columns:
        return None
    if not all(t in frame.columns for t in TARGETS):
        return None
    return frame.set_index("StudyInstanceUID")[list(TARGETS)]


def bootstrap_gap(a: np.ndarray, b: np.ndarray, truth: np.ndarray, draws: int = 4000):
    """95% CI on the macro-agreement gap b - a, resampling studies.

    58 studies is a small stick to measure with; a gap without an interval
    invites reading noise as progress.
    """
    rng = np.random.default_rng(0)
    n = len(truth)
    gaps = np.empty(draws)
    for d in range(draws):
        i = rng.integers(0, n, n)
        gaps[d] = (b[i] == truth[i]).mean(axis=0).mean() - (a[i] == truth[i]).mean(axis=0).mean()
    return float(gaps.mean()), float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tables", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    train = pd.read_csv(args.data / "train.csv").set_index("StudyInstanceUID")
    gold = train[train[list(TARGETS)].notna().all(axis=1)][list(TARGETS)]
    truth = gold.to_numpy() > 0.5

    states, _ = rules.read_corpus(train.loc[gold.index, "Report"])
    tables = {"ours/rules": (states == R.LabelState.POSITIVE.value).astype(int)}
    for path in args.tables:
        table = load_table(path)
        if table is None:
            continue
        missing = gold.index.difference(table.index)
        if len(missing):
            print(f"skipping {path.name}: {len(missing)} gold studies absent")
            continue
        tables[f"{path.parent.name}/{path.name}"] = (table.loc[gold.index] > 0.5).astype(int)

    rows = []
    for target in TARGETS:
        row = {"target": target, "gold_pos": int((gold[target] > 0.5).sum())}
        for name, table in tables.items():
            agree = (table[target].to_numpy() > 0.5) == (gold[target].to_numpy() > 0.5)
            row[name] = round(float(agree.mean()), 3)
        rows.append(row)
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))

    print("\nMACRO agreement:")
    macro = {name: float(frame[name].mean()) for name in tables}
    all_negative = float(np.mean([(gold[t] <= 0.5).mean() for t in TARGETS]))
    for name, value in sorted(macro.items(), key=lambda kv: -kv[1]):
        print(f"  {name:48} {value:.4f}")
    print(f"  {'(all-negative baseline)':48} {all_negative:.4f}")

    ours = tables["ours/rules"].to_numpy() > 0.5
    best = max((k for k in tables if k != "ours/rules"), key=lambda k: macro[k], default=None)
    if best:
        mean, low, high = bootstrap_gap(ours, tables[best].to_numpy() > 0.5, truth)
        print(f"\n{best} minus ours: {mean:+.3f}  95% CI [{low:+.3f}, {high:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
