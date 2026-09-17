#!/usr/bin/env python3
"""Build the leak-safe fold assignment and write it where training can read it.

Two leaks are known here. Studies share reports — 250 of 4,407 share one with
another study, in clusters of up to 37 — and studies share scanners. A split
that ignores either reports a CV number better than the model deserves.

Report clusters we can compute from train.csv alone. Scanner identity lives in
the DICOM headers, so `--scanner-keys` takes a CSV produced on Kaggle
(StudyInstanceUID, scanner_key); without it the split guards the report leak
only, and says so rather than implying otherwise.

Stratification needs a label for every study, and only 58 have gold ones, so it
uses a label table (the public one by default) binarised at 0.5.

    python3 bin/build_folds.py --data DIR --labels TABLE.csv --out cache/folds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import folds as F  # noqa: E402
from knee.constants import TARGETS  # noqa: E402


def build(
    data: Path,
    labels_path: Path | None,
    scanner_path: Path | None,
    guard: str,
    n_splits: int,
    seed: int,
):
    train = pd.read_csv(data / "train.csv").set_index("StudyInstanceUID")

    if labels_path is not None:
        table = pd.read_csv(labels_path).set_index("StudyInstanceUID")
        labels = (table.reindex(train.index)[list(TARGETS)] > 0.5).astype(float)
        source = labels_path.name
    else:
        labels = train[list(TARGETS)].fillna(0.0)
        source = "train.csv gold columns (58 studies; weak stratification)"
    print(f"stratifying on {source}")

    clusters = F.report_clusters(train["Report"])
    clusters.index = train.index
    available = {"report": clusters}

    if scanner_path is not None:
        scanner = pd.read_csv(scanner_path).set_index("StudyInstanceUID")["scanner_key"]
        scanner = scanner.reindex(train.index).fillna(pd.Series(train.index, index=train.index))
        scanner.name = "scanner_key"
        available["scanner"] = scanner
    elif guard in ("scanner", "both"):
        raise SystemExit(f"--guard {guard} needs --scanner-keys (run notebooks/kaggle/extract_headers.py)")

    print("\ngrouping diagnostics:")
    print(F.group_diagnostics(*available.values()).to_string(index=False))

    # The two guards do not compose on this data: report boilerplate bridges
    # scanners, so the transitive closure of both collapses to 37 groups with the
    # largest holding 42% of studies. Pick one deliberately rather than letting a
    # fallback pick for you.
    wanted = list(available.values()) if guard == "both" else [available[guard]]
    print(f"\nguarding: {guard}")
    groups = F.combine_groups(*wanted) if len(wanted) > 1 else wanted[0].rename("group")
    fold = F.make_folds(labels, groups, n_splits=n_splits, seed=seed)
    F.assert_no_leak(fold, groups, *wanted)
    print(f"\nfold sizes: {fold.value_counts().sort_index().to_dict()}")

    prevalence = pd.DataFrame(
        {f"fold{k}": labels[fold == k].mean() for k in sorted(fold.unique())}
    )
    spread = (prevalence.max(axis=1) - prevalence.min(axis=1)).sort_values(ascending=False)
    print("\nworst prevalence spread across folds (rare findings are the ones to watch):")
    print(spread.head(4).round(4).to_string())

    return pd.DataFrame(
        {"StudyInstanceUID": train.index, "fold": fold.to_numpy(), "report_cluster": clusters.to_numpy()}
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None, help="label table for stratification")
    parser.add_argument("--scanner-keys", type=Path, default=None)
    parser.add_argument(
        "--guard",
        choices=("report", "scanner", "both"),
        default="report",
        help="which leak to group on; 'both' collapses on this dataset (see docs/results.md)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = build(args.data, args.labels, args.scanner_keys, args.guard, args.splits, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(frame)} studies)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
