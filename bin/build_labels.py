#!/usr/bin/env python3
"""Assemble the training label matrix: soft targets plus a per-cell weight.

Three sources, used for what each is actually good for:

- The **public label table** supplies the soft targets. Measured at 0.893 macro
  AUC against the 58 gold studies, it is the best supervision available and
  nothing we have improves on it (see docs/competitive-landscape.md).
- The **58 gold studies** override it where they exist. They are the only
  ground truth in the dataset, and they carry full weight.
- Our **rule reader** supplies confidence, not labels. It scores 0.748 on its
  own and blending it into the targets makes them worse — but it can say
  whether the report *addressed* a finding at all, and a label the report never
  mentioned deserves less weight in the loss than one the report states.

The weighting is a hypothesis, not a result: it has not yet been shown to help
an OOF score. Train with and without it and keep whichever wins.

    python3 bin/build_labels.py --data DIR --public TABLE.csv --out cache/
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

SILENT_WEIGHT = 0.6
CONTRADICTED_WEIGHT = 0.5
GOLD_WEIGHT = 8.0


def build(data: Path, public: Path):
    train = pd.read_csv(data / "train.csv").set_index("StudyInstanceUID")
    table = pd.read_csv(public).set_index("StudyInstanceUID")
    targets = pd.DataFrame(
        table.reindex(train.index)[list(TARGETS)].to_numpy(dtype=float),
        index=train.index,
        columns=list(TARGETS),
    )
    if targets.isna().any().any():
        n = int(targets.isna().any(axis=1).sum())
        raise SystemExit(f"{n} studies are absent from {public.name}; refusing to guess them")

    gold_mask = train[list(TARGETS)].notna().all(axis=1)
    gold = train.loc[gold_mask, list(TARGETS)]
    targets.loc[gold.index] = gold.to_numpy(dtype=float)
    print(f"{len(gold)} gold studies override the table; {len(train) - len(gold)} keep it")

    states, _ = rules.read_corpus(train["Report"])
    silent = (states == R.LabelState.UNMENTIONED.value).to_numpy()
    says_negative = (states == R.LabelState.NEGATIVE.value).to_numpy()
    says_positive = (states == R.LabelState.POSITIVE.value).to_numpy()
    table_positive = targets.to_numpy() > 0.5

    weights = np.ones(targets.shape)
    weights[silent] = SILENT_WEIGHT
    contradicted = (says_negative & table_positive) | (says_positive & ~table_positive)
    weights[contradicted] = CONTRADICTED_WEIGHT
    weights[gold_mask.to_numpy(), :] = GOLD_WEIGHT

    weight_frame = pd.DataFrame(weights, index=train.index, columns=list(TARGETS))
    print(f"\nper-cell weights: {SILENT_WEIGHT} where the report is silent "
          f"({silent.mean():.1%} of cells), {CONTRADICTED_WEIGHT} where our reader "
          f"contradicts the table ({contradicted.mean():.1%}), {GOLD_WEIGHT} for gold studies")
    print("\nsoft target distribution per finding (mean, and share above 0.5):")
    summary = pd.DataFrame(
        {"mean": targets.mean().round(3), "share>0.5": (targets > 0.5).mean().round(3)}
    )
    print(summary.to_string())
    return targets, weight_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    targets, weights = build(args.data, args.public)
    args.out.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.out / "labels.csv")
    weights.to_csv(args.out / "label_weights.csv")
    print(f"\nwrote {args.out / 'labels.csv'} and {args.out / 'label_weights.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
