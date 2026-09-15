"""Leak-safe cross-validation.

Two leaks are known in this dataset: studies that share a report (duplicated or
near-duplicated text) and studies that share a scanner. A split that ignores
either reports an optimistic CV, which is how a plateau of teams ends up tuning
against the public leaderboard instead of their own validation.

`make_folds` groups on both and stratifies on the rarest positive finding.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import TARGETS

_PUNCT = re.compile(r"[^\w\s]+", re.UNICODE)
_SPACE = re.compile(r"\s+")
_DIGIT = re.compile(r"\d+")


def normalise_report(text: str) -> str:
    """Casefold, strip accents, punctuation and digits, collapse whitespace.

    Digits go because dates and measurements differ between otherwise identical
    template reports; keeping them would hide the duplicates we are hunting.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = _PUNCT.sub(" ", text.casefold())
    text = _DIGIT.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def _shingles(text: str, k: int = 5) -> set[str]:
    tokens = text.split()
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


class _Union:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, a: int) -> int:
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def report_clusters(reports: pd.Series, threshold: float = 0.9, k: int = 5) -> pd.Series:
    """Cluster id per study: exact duplicates plus near-duplicates by Jaccard.

    Candidate pairs come from an inverted shingle index, so this stays linear-ish
    in corpus size instead of comparing all 4,407 x 4,407 pairs.
    """
    texts = [normalise_report(t) for t in reports]
    n = len(texts)
    union = _Union(n)

    by_hash: dict[str, int] = {}
    for i, t in enumerate(texts):
        h = hashlib.sha1(t.encode()).hexdigest()
        if h in by_hash:
            union.union(by_hash[h], i)
        else:
            by_hash[h] = i

    sets = [_shingles(t, k) for t in texts]
    index: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sets):
        for sh in s:
            index[sh].append(i)

    candidates: set[tuple[int, int]] = set()
    for members in index.values():
        if 1 < len(members) <= 50:  # a shingle shared by everything carries no signal
            for a_idx in range(len(members)):
                for b_idx in range(a_idx + 1, len(members)):
                    candidates.add((members[a_idx], members[b_idx]))

    for a, b in candidates:
        if union.find(a) == union.find(b):
            continue
        sa, sb = sets[a], sets[b]
        if not sa or not sb:
            continue
        inter = len(sa & sb)
        if inter / (len(sa) + len(sb) - inter) >= threshold:
            union.union(a, b)

    roots = [union.find(i) for i in range(n)]
    remap = {r: c for c, r in enumerate(dict.fromkeys(roots))}
    return pd.Series([remap[r] for r in roots], index=reports.index, name="report_cluster")


def scanner_keys(series: pd.DataFrame, columns: tuple[str, ...] = ()) -> pd.Series:
    """One scanner/site key per study, from whichever header columns exist.

    Falls back to the study id when no header identifies the scanner, which makes
    the study its own group and is the safe direction to be wrong in.
    """
    default = ("Manufacturer", "ManufacturerModelName", "MagneticFieldStrength", "StationName")
    columns = columns or default
    present = [c for c in columns if c in series.columns]
    if not present:
        return pd.Series(series.index.astype(str), index=series.index, name="scanner_key")
    key = series[present].astype(str).agg("|".join, axis=1)
    key.name = "scanner_key"
    return key


def combine_groups(*group_series: pd.Series, max_group_fraction: float = 0.25) -> pd.Series:
    """Merge several grouping keys transitively.

    Two studies sharing *either* a report cluster or a scanner land in one group:
    guarding both leaks means the union of the groupings, not the intersection.

    The union can collapse. If report clusters bridge scanners, transitive closure
    chains every study into one group and cross-validation becomes impossible.
    That is a real finding about the data, not a bug to paper over, so it raises
    with the numbers needed to choose a fallback (usually: group on scanner, and
    measure the report leak separately).
    """
    frames = [s.reset_index(drop=True) for s in group_series]
    n = len(frames[0])
    union = _Union(n)
    for s in frames:
        first: dict[object, int] = {}
        for i, v in enumerate(s):
            if v in first:
                union.union(first[v], i)
            else:
                first[v] = i
    roots = [union.find(i) for i in range(n)]
    remap = {r: c for c, r in enumerate(dict.fromkeys(roots))}
    out = pd.Series([remap[r] for r in roots], index=group_series[0].index, name="group")
    largest = out.value_counts().iloc[0] / n
    if largest > max_group_fraction:
        names = [s.name or f"key{i}" for i, s in enumerate(group_series)]
        raise ValueError(
            f"grouping on {names} collapses: the largest group holds {largest:.1%} of studies "
            f"({out.nunique()} groups for {n} studies). Guard the leaks separately."
        )
    return out


def group_diagnostics(*group_series: pd.Series) -> pd.DataFrame:
    """Group count and largest-group share for each key and for their union.

    Run this before committing to a split; it is how you find out that two guards
    cannot coexist before a fold assignment silently becomes one fold.
    """
    rows = []
    for s in group_series:
        counts = s.value_counts()
        rows.append(
            {
                "key": s.name or "unnamed",
                "n_groups": int(s.nunique()),
                "largest_share": float(counts.iloc[0] / len(s)),
            }
        )
    try:
        merged = combine_groups(*group_series, max_group_fraction=1.0)
        counts = merged.value_counts()
        rows.append(
            {
                "key": "union",
                "n_groups": int(merged.nunique()),
                "largest_share": float(counts.iloc[0] / len(merged)),
            }
        )
    except ValueError:  # pragma: no cover - union is computed with the cap lifted
        pass
    return pd.DataFrame(rows)


def rarest_positive(labels: pd.DataFrame, targets: tuple[str, ...] = TARGETS) -> pd.Series:
    """Stratification key: the rarest finding a study is positive for, else -1.

    Standard multi-label trick. Keeps the scarce findings (Fracture, Synovitis)
    spread across folds instead of pooling in one.
    """
    present = [t for t in targets if t in labels.columns]
    values = labels[present].to_numpy(dtype=float)
    positives = np.nan_to_num(values) > 0.5
    order = np.argsort(positives.sum(axis=0))  # rarest column first
    out = np.full(len(labels), -1, dtype=int)
    for rank, col in enumerate(order):
        hit = positives[:, col] & (out == -1)
        out[hit] = rank
    return pd.Series(out, index=labels.index, name="stratum")


def make_folds(
    labels: pd.DataFrame,
    groups: pd.Series,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.Series:
    """Grouped, stratified fold assignment."""
    strata = rarest_positive(labels)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=labels.index, name="fold", dtype=int)
    for k, (_, valid) in enumerate(splitter.split(labels, strata, groups)):
        fold.iloc[valid] = k
    if (fold < 0).any():
        raise RuntimeError("some studies were never assigned a fold")
    return fold


def assert_no_leak(fold: pd.Series, *group_series: pd.Series) -> None:
    """Raise if any group spans two folds. Call this before trusting any CV number."""
    for s in group_series:
        spread = pd.DataFrame({"g": s.to_numpy(), "f": fold.to_numpy()}).groupby("g")["f"].nunique()
        bad = spread[spread > 1]
        if len(bad):
            raise AssertionError(
                f"{s.name or 'group'}: {len(bad)} group(s) span >1 fold, e.g. {bad.index[0]!r}"
            )


def inflation(
    y_true: np.ndarray,
    oof_guarded: np.ndarray,
    oof_unguarded: np.ndarray,
) -> float:
    """How many macro-AUC points a leaky split invents.

    Run the same model under both splits and subtract. The number is worth
    knowing before reading anyone else's CV, our own included.
    """
    from .metrics import macro_auc

    return macro_auc(y_true, oof_unguarded) - macro_auc(y_true, oof_guarded)
