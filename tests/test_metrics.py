import numpy as np
import pytest

from knee import metrics
from knee.constants import TARGETS


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    y = (rng.random((300, 12)) < 0.3).astype(float)
    signal = y * 2 - 1
    return y, signal + rng.normal(0, 1.2, y.shape)


def test_macro_auc_matches_mean_of_per_target(scores):
    y, s = scores
    table = metrics.per_target_table(y, s)
    assert metrics.macro_auc(y, s) == pytest.approx(table["auc"].mean())


def test_single_class_column_is_skipped_not_fatal(scores):
    y, s = scores
    y[:, 3] = 0.0
    assert np.isnan(metrics.target_auc(y[:, 3], s[:, 3]))
    assert np.isfinite(metrics.macro_auc(y, s))


def test_tie_report_finds_constant_blocks(scores):
    _, s = scores
    s[:50, 0] = 0.5
    report = {t.target: t for t in metrics.tie_report(s)}
    assert report["ACL"].largest_tie_block == 50
    assert report["ACL"].suspicious


def test_headroom_ranks_worst_target_first(scores):
    y, s = scores
    s[:, 5] = 0.5 + np.random.default_rng(1).normal(0, 0.01, len(s))  # ruin one target
    table = metrics.headroom(y, s)
    assert table.iloc[0]["target"] == TARGETS[5]
    assert table.iloc[0]["gain_if_at_ceiling"] > 0


def test_arm_correlation_is_symmetric_with_unit_diagonal(scores):
    _, s = scores
    rng = np.random.default_rng(2)
    corr = metrics.arm_correlation({"a": s, "b": s + rng.normal(0, 3, s.shape)})
    assert corr.loc["a", "a"] == pytest.approx(1.0)
    assert corr.loc["a", "b"] == pytest.approx(corr.loc["b", "a"])
    assert corr.loc["a", "b"] < 0.97
