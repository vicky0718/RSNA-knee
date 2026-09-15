#!/usr/bin/env python3
"""Score label tables against the 58 gold studies.

**Macro AUC is the headline**, and binarising first is a trap: the public label
tables hold calibrated probabilities, not 0/1, so thresholding them at 0.5
discards exactly the information the competition metric rewards. Doing that
made the best public table look like 0.819 when it is 0.893 — which is the
figure its author published. Agreement is printed underneath as a secondary
view, useful for reading a rule-based extractor but not for ranking supervision.

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
from knee.metrics import macro_auc, target_auc  # noqa: E402


def load_table(path: Path) -> pd.DataFrame | None:
    frame = pd.read_csv(path)
    if "StudyInstanceUID" not in frame.columns:
        return None
    if not all(t in frame.columns for t in TARGETS):
        return None
    return frame.set_index("StudyInstanceUID")[list(TARGETS)]


def bootstrap_gap(a: np.ndarray, b: np.ndarray, truth: np.ndarray, draws: int = 2000):
    """95% CI on the macro-AUC gap b - a, resampling studies.

    58 studies is a small stick to measure with; a gap without an interval
    invites reading noise as progress.
    """
    rng = np.random.default_rng(0)
    n = len(truth)
    gaps = []
    for _ in range(draws):
        i = rng.integers(0, n, n)
        gap = macro_auc(truth[i], b[i]) - macro_auc(truth[i], a[i])
        if np.isfinite(gap):
            gaps.append(gap)
    gaps = np.asarray(gaps)
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
    all_states, _ = rules.read_corpus(train["Report"])
    calibration = R.calibrate(all_states, gold)
    ours = pd.DataFrame(
        calibration.apply(all_states).astype(float), index=train.index, columns=list(TARGETS)
    ).loc[gold.index]
    tables = {"ours/rules (calibrated)": ours}

    for path in args.tables:
        table = load_table(path)
        if table is None:
            continue
        missing = gold.index.difference(table.index)
        if len(missing):
            print(f"skipping {path.name}: {len(missing)} gold studies absent")
            continue
        tables[path.name] = table.loc[gold.index].astype(float)

    rows = []
    for target in TARGETS:
        column = list(TARGETS).index(target)
        row = {"target": target, "gold_pos": int((gold[target] > 0.5).sum())}
        for name, table in tables.items():
            row[name] = round(target_auc(truth[:, column], table[target].to_numpy()), 3)
        rows.append(row)
    frame = pd.DataFrame(rows)
    print("per-target AUC against the 58 gold studies:")
    print(frame.to_string(index=False))

    macro = {name: macro_auc(truth, table.to_numpy()) for name, table in tables.items()}
    print("\nMACRO AUC (the headline):")
    for name, value in sorted(macro.items(), key=lambda kv: -kv[1]):
        print(f"  {name:44} {value:.4f}")

    print("\nbinary agreement at 0.5 (secondary; misleading for soft tables):")
    for name, table in sorted(tables.items()):
        agree = np.mean([
            ((table[t].to_numpy() > 0.5) == (gold[t].to_numpy() > 0.5)).mean() for t in TARGETS
        ])
        print(f"  {name:44} {agree:.4f}")
    all_negative = float(np.mean([(gold[t] <= 0.5).mean() for t in TARGETS]))
    print(f"  {'(all-negative baseline)':44} {all_negative:.4f}")

    mine = tables["ours/rules (calibrated)"].to_numpy()
    best = max((k for k in tables if not k.startswith("ours")), key=lambda k: macro[k], default=None)
    if best:
        mean, low, high = bootstrap_gap(mine, tables[best].to_numpy(), truth)
        print(f"\n{best} minus ours: {mean:+.3f} macro AUC  95% CI [{low:+.3f}, {high:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
