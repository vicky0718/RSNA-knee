"""Scoring, per-target reporting, and the diagnostics that catch silent failures.

The competition metric is the macro average of twelve ROC AUCs. Two failure modes
are specific to this setup and both are checked here rather than discovered on the
leaderboard: a target column that is single-class (AUC undefined), and a block of
identical predictions, which under ROC-AUC scores half credit against everything
it ties with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import TARGETS


def target_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC for one target, or NaN when the column holds a single class."""
    y_true = np.asarray(y_true)
    finite = np.isfinite(y_true) & np.isfinite(np.asarray(y_score, dtype=float))
    y_true, y_score = y_true[finite], np.asarray(y_score, dtype=float)[finite]
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mean of the per-target AUCs, skipping undefined columns.

    Skipping is a reporting convenience for local CV. The competition's own test
    set is not ours to inspect, so never assume a column will be scoreable.
    """
    aucs = [target_auc(y_true[:, j], y_score[:, j]) for j in range(y_true.shape[1])]
    valid = [a for a in aucs if np.isfinite(a)]
    return float(np.mean(valid)) if valid else float("nan")


def per_target_table(
    y_true: np.ndarray, y_score: np.ndarray, targets: tuple[str, ...] = TARGETS
) -> pd.DataFrame:
    """Per-target AUC, positive count and prevalence, sorted worst first."""
    rows = []
    for j, name in enumerate(targets):
        col = y_true[:, j]
        finite = np.isfinite(col)
        rows.append(
            {
                "target": name,
                "auc": target_auc(col, y_score[:, j]),
                "n_pos": int(np.nansum(col[finite] > 0.5)),
                "n_labelled": int(finite.sum()),
                "prevalence": float(np.nanmean(col[finite] > 0.5)) if finite.any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("auc", na_position="first").reset_index(drop=True)


def headroom(y_true: np.ndarray, y_score: np.ndarray, ceiling: float = 0.95) -> pd.DataFrame:
    """Macro-AUC points each target would contribute if lifted to `ceiling`.

    With a twelve-way macro average, one target at 0.65 is worth more than every
    backbone swap combined: lifting it to 0.75 is +0.008 overall.
    """
    table = per_target_table(y_true, y_score)
    table["gain_if_at_ceiling"] = (ceiling - table["auc"]).clip(lower=0) / len(TARGETS)
    return table.sort_values("gain_if_at_ceiling", ascending=False).reset_index(drop=True)


@dataclass(frozen=True)
class TieReport:
    """How much of a prediction column is stuck in identical blocks."""

    target: str
    largest_tie_block: int
    tied_fraction: float

    @property
    def suspicious(self) -> bool:
        return self.tied_fraction > 0.01


def tie_report(y_score: np.ndarray, targets: tuple[str, ...] = TARGETS) -> list[TieReport]:
    """Find constant blocks in each prediction column.

    A study whose slots are all empty can only be given a constant. Those studies
    tie with each other and earn half credit; the bug is invisible in the mean
    prediction and obvious here.
    """
    out = []
    for j, name in enumerate(targets):
        col = np.asarray(y_score[:, j], dtype=float)
        _, counts = np.unique(col[np.isfinite(col)], return_counts=True)
        largest = int(counts.max()) if counts.size else 0
        tied = int(counts[counts > 1].sum()) if counts.size else 0
        out.append(TieReport(name, largest, tied / max(col.size, 1)))
    return out


def arm_correlation(arms: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean per-target Spearman correlation between ensemble arms.

    The public 0.941 is 41 checkpoints of one family; a 42nd correlated member
    buys nothing. Reject an arm above ~0.97 here regardless of its solo score.
    """
    names = list(arms)
    out = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = arms[names[a]], arms[names[b]]
            per_target = [
                np.corrcoef(_rankdata(x[:, j]), _rankdata(y[:, j]))[0, 1]
                for j in range(x.shape[1])
            ]
            rho = float(np.nanmean(per_target))
            out.iloc[a, b] = out.iloc[b, a] = rho
    return out


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks, so tied blocks do not fake a correlation."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(x.size, dtype=float)
    # average ranks within tied groups
    sorted_x = x[order]
    start = 0
    for i in range(1, x.size + 1):
        if i == x.size or sorted_x[i] != sorted_x[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks
