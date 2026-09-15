import numpy as np
import pandas as pd
import pytest

from knee import oof as O
from knee.constants import TARGETS


@pytest.fixture
def setup():
    rng = np.random.default_rng(0)
    index = [f"s{i}" for i in range(50)]
    fold = pd.Series(np.arange(50) % 5, index=index)
    truth = pd.DataFrame(
        (rng.random((50, 12)) < 0.3).astype(float), index=index, columns=list(TARGETS)
    )
    return fold, truth


def test_run_cv_fills_every_row_exactly_once(setup):
    fold, truth = setup
    seen = []

    def fit_predict(train, valid):
        assert not set(train) & set(valid)  # a fold never predicts what it trained on
        seen.extend(valid.tolist())
        return np.full((len(valid), 12), 0.5)

    result = O.run_cv(fit_predict, fold)
    assert sorted(seen) == list(range(50))
    assert not result.predictions.isna().any().any()


def test_wrong_prediction_shape_is_caught(setup):
    fold, _ = setup
    with pytest.raises(ValueError, match="expected"):
        O.run_cv(lambda train, valid: np.zeros((len(valid), 3)), fold)


def test_missing_predictions_refuse_to_score(setup):
    fold, truth = setup
    predictions = pd.DataFrame(0.5, index=fold.index, columns=list(TARGETS))
    predictions.iloc[3] = np.nan
    with pytest.raises(AssertionError, match="no OOF prediction"):
        O.OOF(predictions, fold).score(truth)


def test_tie_blocks_are_surfaced(setup):
    """A constant column is the empty-slot leak arriving at the far end."""
    fold, _ = setup
    predictions = pd.DataFrame(
        np.random.default_rng(1).random((50, 12)), index=fold.index, columns=list(TARGETS)
    )
    predictions.iloc[:20, 0] = 0.5
    assert TARGETS[0] in O.OOF(predictions, fold).warn_on_ties()


def test_score_and_report_agree(setup):
    fold, truth = setup
    rng = np.random.default_rng(2)
    predictions = pd.DataFrame(
        truth.to_numpy() * 0.6 + rng.random((50, 12)) * 0.4, index=fold.index, columns=list(TARGETS)
    )
    result = O.OOF(predictions, fold)
    assert result.score(truth) == pytest.approx(result.report(truth)["auc"].mean())
