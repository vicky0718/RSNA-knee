"""Assembling out-of-fold predictions, and refusing to score a broken one.

Every number this project trusts — fusion weights, arm selection, whether a
change helped — comes from an OOF matrix. So the failure that matters is not a
bad score but a *quietly wrong* one: a fold that predicted rows it also trained
on, a study that never got a prediction and silently defaulted to zero, a run
that scored 4,400 of 4,407 studies and reported the mean anyway.

`run_cv` builds the matrix and `OOF.check` refuses to hand it over until every
row is filled exactly once by a fold that did not train on it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import TARGETS
from .metrics import macro_auc, per_target_table, tie_report


@dataclass
class OOF:
    """Out-of-fold predictions with the fold that produced each row."""

    predictions: pd.DataFrame  # index: study, columns: targets
    fold: pd.Series

    def check(self) -> OOF:
        """Raise unless every study has exactly one prediction from one fold."""
        missing = self.predictions.isna().any(axis=1)
        if missing.any():
            raise AssertionError(
                f"{int(missing.sum())} studies have no OOF prediction "
                f"(first: {self.predictions.index[missing][0]!r})"
            )
        if not self.predictions.index.equals(self.fold.index):
            raise AssertionError("predictions and fold assignment are not aligned")
        if self.predictions.index.has_duplicates:
            raise AssertionError("a study appears more than once in the OOF matrix")
        return self

    def score(self, truth: pd.DataFrame) -> float:
        """Macro AUC against whichever labels you are validating on."""
        self.check()
        common = self.predictions.index.intersection(truth.index)
        return macro_auc(
            truth.loc[common, list(TARGETS)].to_numpy(dtype=float) > 0.5,
            self.predictions.loc[common, list(TARGETS)].to_numpy(dtype=float),
        )

    def report(self, truth: pd.DataFrame) -> pd.DataFrame:
        self.check()
        common = self.predictions.index.intersection(truth.index)
        return per_target_table(
            truth.loc[common, list(TARGETS)].to_numpy(dtype=float) > 0.5,
            self.predictions.loc[common, list(TARGETS)].to_numpy(dtype=float),
        )

    def warn_on_ties(self) -> list[str]:
        """Names of targets with a suspicious block of identical predictions.

        A tie block is how the empty-slot leak shows up at the far end of the
        pipeline: studies that fell through the slot logic all get one constant.
        """
        return [t.target for t in tie_report(self.predictions.to_numpy()) if t.suspicious]


def run_cv(
    fit_predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
    fold: pd.Series,
    targets: tuple[str, ...] = TARGETS,
) -> OOF:
    """Run a fold loop and collect out-of-fold predictions.

    `fit_predict(train_positions, valid_positions)` is handed integer positions
    into `fold` and returns predictions for the validation rows only — it never
    sees the validation labels, because it is never given them.
    """
    index = fold.index
    out = pd.DataFrame(np.nan, index=index, columns=list(targets), dtype=float)
    positions = np.arange(len(index))
    for k in sorted(fold.unique()):
        valid = positions[(fold == k).to_numpy()]
        train = positions[(fold != k).to_numpy()]
        predictions = np.asarray(fit_predict(train, valid), dtype=float)
        if predictions.shape != (len(valid), len(targets)):
            raise ValueError(
                f"fold {k}: expected {(len(valid), len(targets))} predictions, "
                f"got {predictions.shape}"
            )
        out.iloc[valid] = predictions
    return OOF(out, fold).check()
