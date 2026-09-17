#!/usr/bin/env python3
"""Measure what the leak guard actually buys, with a real model in the harness.

The claim behind our fold design is that studies sharing a report inflate CV.
This puts a number on it: the same model, the same data, scored out-of-fold
under a random split and under the report-grouped split. The difference is CV
that a leaky split invents.

The model is deliberately a text model — TF-IDF over the report, predicting the
public label table. It is the cheapest thing that can exploit a duplicated
report, which makes it the sharpest instrument for finding the leak, and it runs
in seconds with no GPU. It is not the image model, so read the number as "what
the report leak is worth to a model that can see the report", which is an upper
bound for the image pipeline rather than its exact exposure.

    python3 bin/measure_leak.py --data DIR --labels TABLE.csv --folds cache/folds.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from knee import folds as F, oof as O  # noqa: E402
from knee.constants import TARGETS  # noqa: E402


def text_model(reports: pd.Series, labels: pd.DataFrame):
    """A fit_predict closure over TF-IDF character n-grams.

    Character n-grams rather than words: the corpus spans ten languages and
    several alphabets, and a word-level vectoriser would model the language, not
    the finding.
    """
    binary = (labels[list(TARGETS)].to_numpy(dtype=float) > 0.5).astype(int)

    def fit_predict(train: np.ndarray, valid: np.ndarray) -> np.ndarray:
        vectoriser = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=60000)
        x_train = vectoriser.fit_transform(reports.iloc[train])
        x_valid = vectoriser.transform(reports.iloc[valid])
        out = np.zeros((len(valid), len(TARGETS)))
        for j in range(len(TARGETS)):
            y = binary[train, j]
            if len(np.unique(y)) < 2:
                out[:, j] = float(y.mean())
                continue
            model = LogisticRegression(max_iter=1000, C=1.0)
            model.fit(x_train, y)
            out[:, j] = model.predict_proba(x_valid)[:, 1]
        return out

    return fit_predict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--folds",
        nargs="+",
        required=True,
        metavar="NAME=PATH",
        help="one or more fold files to compare, e.g. report=cache/folds.csv scanner=cache/folds_scanner.csv",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train = pd.read_csv(args.data / "train.csv").set_index("StudyInstanceUID")
    labels = pd.read_csv(args.labels).set_index("StudyInstanceUID").reindex(train.index)
    gold = train[train[list(TARGETS)].notna().all(axis=1)][list(TARGETS)]
    fit_predict = text_model(train["Report"], labels)

    rng = np.random.default_rng(args.seed)
    splits = {"random": pd.Series(rng.permutation(np.arange(len(train)) % 5), index=train.index)}
    for spec in args.folds:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).stem, spec
        frame = pd.read_csv(path).set_index("StudyInstanceUID")
        splits[name] = frame["fold"].reindex(train.index)

    scores = {}
    for name, fold in splits.items():
        result = O.run_cv(fit_predict, fold)
        scores[name] = result.score(labels)
        print(f"{name:22} OOF macro AUC vs public labels: {scores[name]:.4f}")
        print(f"{' ':22} OOF macro AUC vs the 58 gold:   {result.score(gold):.4f}")

    print("\ninflation each guard removes, against the random split:")
    for name, score in scores.items():
        if name == "random":
            continue
        print(f"  {name:20} {scores['random'] - score:+.4f} macro AUC")
    print("\nA guard worth keeping shows a positive number here: the random split was scoring")
    print("higher than the model deserved. A number near zero means the guard costs nothing")
    print("and protects against nothing measurable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
