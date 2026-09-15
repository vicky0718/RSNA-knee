"""Ensemble fusion, fitted on out-of-fold predictions rather than the leaderboard.

The public plateau tuned its per-finding blend by submitting probes: "probe #22
raised five findings, and put Lateral Meniscus at 1.00 ... That bought +0.002 on
the public split." That is twelve correlated columns fitted to 30% of the test
set with no held-out check, shared by hundreds of teams.

This module does the same fitting against OOF predictions and, crucially,
reports the *honest* gain: weights fitted on K-1 folds and scored on the fold
they never saw. If per-target routing does not survive that, it is noise, and
`shrink_to_global` pulls it back toward one global weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import TARGETS
from .metrics import _rankdata, macro_auc, target_auc


def rank_normalise(x: np.ndarray) -> np.ndarray:
    """Per-column ranks scaled to [0, 1].

    The metric is rank-based, so arms must be fused in rank space; averaging raw
    probabilities lets an over-confident arm dominate a better-ordered one.
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        r = _rankdata(x[:, j])
        out[:, j] = r / max(r.size - 1, 1)
    return out


def blend(arms: dict[str, np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Weighted rank mean. `weights` is (n_arms, n_targets), columns summing to 1."""
    names = list(arms)
    if weights.shape[0] != len(names):
        raise ValueError(f"expected weights for {len(names)} arms, got {weights.shape[0]}")
    ranked = np.stack([rank_normalise(arms[n]) for n in names])  # (arm, sample, target)
    w = weights[:, None, :]
    return (ranked * w).sum(axis=0)


def _fit_one_target(
    ranked: np.ndarray, y: np.ndarray, passes: int = 3, grid: int = 11
) -> np.ndarray:
    """Coordinate ascent on the simplex for a single target.

    Deliberately coarse: a fine grid on 58 gold studies or a noisy OOF column
    fits the noise, which is the failure this whole module exists to avoid.
    """
    n_arms = ranked.shape[0]
    w = np.full(n_arms, 1.0 / n_arms)
    finite = np.isfinite(y)
    if finite.sum() == 0 or len(np.unique(y[finite])) < 2:
        return w
    best = target_auc(y[finite], (ranked[:, finite] * w[:, None]).sum(axis=0))
    for _ in range(passes):
        for a in range(n_arms):
            for candidate in np.linspace(0.0, 1.0, grid):
                trial = w.copy()
                trial[a] = candidate
                total = trial.sum()
                if total <= 0:
                    continue
                trial /= total
                score = target_auc(y[finite], (ranked[:, finite] * trial[:, None]).sum(axis=0))
                if np.isfinite(score) and score > best + 1e-9:
                    best, w = score, trial
    return w


def fit_weights(arms: dict[str, np.ndarray], y: np.ndarray, grid: int = 11) -> pd.DataFrame:
    """Per-target arm weights fitted on the supplied (out-of-fold) predictions."""
    names = list(arms)
    ranked = np.stack([rank_normalise(arms[n]) for n in names])
    out = np.zeros((len(names), y.shape[1]))
    for j in range(y.shape[1]):
        out[:, j] = _fit_one_target(ranked[:, :, j], y[:, j], grid=grid)
    return pd.DataFrame(out, index=names, columns=list(TARGETS[: y.shape[1]]))


def fit_global_weights(arms: dict[str, np.ndarray], y: np.ndarray, grid: int = 11) -> pd.Series:
    """One weight vector shared by all targets — the honest baseline to beat."""
    names = list(arms)
    ranked = np.stack([rank_normalise(arms[n]) for n in names])
    n_arms = len(names)
    w = np.full(n_arms, 1.0 / n_arms)
    best = macro_auc(y, (ranked * w[:, None, None]).sum(axis=0))
    for _ in range(3):
        for a in range(n_arms):
            for candidate in np.linspace(0.0, 1.0, grid):
                trial = w.copy()
                trial[a] = candidate
                total = trial.sum()
                if total <= 0:
                    continue
                trial /= total
                score = macro_auc(y, (ranked * trial[:, None, None]).sum(axis=0))
                if np.isfinite(score) and score > best + 1e-9:
                    best, w = score, trial
    return pd.Series(w, index=names, name="global")


def shrink_to_global(per_target: pd.DataFrame, global_w: pd.Series, alpha: float) -> pd.DataFrame:
    """Pull per-target weights toward the global vector. alpha=0 keeps per-target, 1 discards it."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return per_target.mul(1 - alpha).add(global_w * alpha, axis=0)


@dataclass(frozen=True)
class HonestGain:
    """Per-target routing measured on folds its weights never saw."""

    in_sample: float
    out_of_sample: float
    global_baseline: float

    @property
    def real(self) -> bool:
        """True when routing beats one global weight on held-out folds."""
        return self.out_of_sample > self.global_baseline

    def __str__(self) -> str:
        verdict = "REAL" if self.real else "NOISE — use the global weight"
        return (
            f"per-target routing: in-sample {self.in_sample:.4f}, "
            f"held-out {self.out_of_sample:.4f}, global {self.global_baseline:.4f} -> {verdict}"
        )


def honest_gain(
    arms: dict[str, np.ndarray], y: np.ndarray, fold: np.ndarray, grid: int = 11
) -> HonestGain:
    """Fit weights on K-1 folds, score on the held-out fold, compare with a global weight.

    Run this before shipping any per-finding weight. It is the check the public
    plateau skipped.
    """
    fold = np.asarray(fold)
    names = list(arms)
    blended = np.zeros_like(y, dtype=float)
    for k in np.unique(fold):
        train, valid = fold != k, fold == k
        fitted = fit_weights({n: arms[n][train] for n in names}, y[train], grid=grid)
        ranked = np.stack([rank_normalise(arms[n][valid]) for n in names])
        blended[valid] = (ranked * fitted.to_numpy()[:, None, :]).sum(axis=0)

    in_sample_w = fit_weights(arms, y, grid=grid)
    in_sample = macro_auc(y, blend(arms, in_sample_w.to_numpy()))
    global_w = fit_global_weights(arms, y, grid=grid)
    baseline = macro_auc(
        y, blend(arms, np.repeat(global_w.to_numpy()[:, None], y.shape[1], axis=1))
    )
    return HonestGain(in_sample, macro_auc(y, blended), baseline)
