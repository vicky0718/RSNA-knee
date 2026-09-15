#!/usr/bin/env python3
"""Audit the competition CSVs before trusting anything built on top of them.

Checks the assumptions the pipeline is built on — slot coverage, the two series
flags, how many studies carry gold labels, and how much of the training set
shares a report. Run it after any data refresh; the numbers are cheap and being
wrong about them is not.

    python3 bin/audit_data.py --data DIR [--out DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import folds  # noqa: E402
from knee.constants import SLOTS, TARGETS  # noqa: E402


def audit(data: Path, out: Path | None) -> int:
    train = pd.read_csv(data / "train.csv")
    series = pd.read_csv(data / "train_series.csv")
    n_studies = train["StudyInstanceUID"].nunique()
    problems = 0

    print(f"train.csv {train.shape}  train_series.csv {series.shape}")
    labels = train[list(TARGETS)]
    fully = labels.notna().all(axis=1)
    print(f"\ngold-labelled studies: {fully.sum()} of {len(train)}")
    print(f"studies with a report: {train['Report'].notna().sum()} of {len(train)}")
    if train["Report"].isna().any():
        print("  ! some studies have no report and can only be supervised by images")
        problems += 1

    print("\nprevalence among gold-labelled studies (%):")
    print((labels[fully].mean() * 100).round(1).sort_values(ascending=False).to_string())

    print("\nslot coverage over studies:")
    for name, plane, fat in SLOTS:
        hit = (series["Anatomical_Plane"] == plane) & (series["Fat_Suppression"] == int(fat))
        have = series[hit]["StudyInstanceUID"].nunique()
        flag = "  <- mostly empty" if have / n_studies < 0.5 else ""
        print(f"  {name:10} {have:>5} {have / n_studies:6.3f}{flag}")

    same = (series["Fluid_Sensitive"] == series["Fat_Suppression"]).mean()
    print(f"\nFluid_Sensitive == Fat_Suppression on {same:.1%} of series")
    if same == 1.0:
        print("  (identical in training: slot selection keys on fat-suppression alone)")

    clusters = folds.report_clusters(train["Report"])
    sizes = clusters.value_counts()
    shared = int(sizes[sizes > 1].sum())
    print(f"\nreport clusters: {clusters.nunique()} for {len(train)} studies")
    print(f"  studies sharing a report with another: {shared} ({shared / len(train):.1%})")
    print(f"  largest cluster: {int(sizes.iloc[0])} studies")
    print("  -> these must not straddle a fold boundary (folds.combine_groups)")

    if out:
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"StudyInstanceUID": train["StudyInstanceUID"], "report_cluster": clusters}
        ).to_csv(out / "report_clusters.csv", index=False)
        print(f"\nwrote {out / 'report_clusters.csv'}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="directory holding train.csv")
    parser.add_argument("--out", type=Path, default=None, help="where to write derived tables")
    args = parser.parse_args()
    return audit(args.data, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
